"""Run-scoped SQLite storage for querying previous tool results."""

from __future__ import annotations

import json
import logging
import sqlite3
import tempfile
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from threading import RLock
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence
from uuid import uuid4

from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel, ConfigDict, Field

from .common import clamped_int
from .sql import _readonly_sql_authorizer, _validate_readonly_sql

logger = logging.getLogger(__name__)

MAX_SAVED_RESULT_QUERY_ROWS = 100

# These tools read SQLite-backed project facts and return tabular rows. Results
# from graph, visualization and static SQL parsing tools are intentionally not
# materialized here.
SQLITE_RESULT_TOOL_NAMES = frozenset(
    {
        "get_excel_row",
        "get_file_description",
        "list_column_catalog",
        "list_columns",
        "list_file_sheet_headers",
        "list_files",
        "list_s2t_table_names",
        "list_s2t_transformations",
        "list_sheets",
        "query_saved_result",
        "resolve_file",
        "run_sql",
        "search_column_catalog",
        "search_excel_values",
        "search_s2t_transformations",
        "semantic_search_descriptions",
        "summarize_s2t_tables",
        "summarize_table_descriptions",
    }
)


class SavedResultColumn(BaseModel):
    """One physical column exposed by a saved result relation."""

    model_config = ConfigDict(extra="forbid")

    name: str
    sqlite_type: str


class SavedResultDescriptor(BaseModel):
    """Opaque reference and query schema for one saved tool result."""

    model_config = ConfigDict(extra="forbid")

    result_ref: str
    source_tool: str
    row_count: int = Field(ge=0)
    source_total: Optional[int] = Field(default=None, ge=0)
    truncated: bool = False
    columns: List[SavedResultColumn] = Field(default_factory=list)


def _quote_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def _json_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bytes)):
        return value
    if isinstance(value, bool):
        return int(value)
    return json.dumps(value, ensure_ascii=False, default=str, separators=(",", ":"))


def _sqlite_type(values: Iterable[Any]) -> str:
    kinds = set()
    for value in values:
        if value is None:
            continue
        if isinstance(value, bool) or isinstance(value, int):
            kinds.add("INTEGER")
        elif isinstance(value, float):
            kinds.add("REAL")
        elif isinstance(value, bytes):
            kinds.add("BLOB")
        else:
            kinds.add("TEXT")
    if not kinds:
        return "TEXT"
    if kinds <= {"INTEGER", "REAL"}:
        return "REAL" if "REAL" in kinds else "INTEGER"
    return kinds.pop() if len(kinds) == 1 else "TEXT"


def _normalized_columns(
    rows: Sequence[Mapping[str, Any]],
    declared_columns: Sequence[Any],
) -> List[str]:
    columns: List[str] = []
    for value in declared_columns:
        name = str(value or "").strip()
        if name and name not in columns:
            columns.append(name)
    for row in rows:
        for value in row:
            name = str(value or "").strip()
            if name and name not in columns:
                columns.append(name)
    return columns


def _tabular_payload(payload: Any) -> Optional[Dict[str, Any]]:
    if isinstance(payload, list):
        raw_rows = payload
        declared_columns: Sequence[Any] = ()
        metadata: Mapping[str, Any] = {}
    elif isinstance(payload, Mapping):
        metadata = payload
        if isinstance(payload.get("rows"), list):
            raw_rows = payload["rows"]
        elif isinstance(payload.get("preview_rows"), list):
            raw_rows = payload["preview_rows"]
        else:
            return None
        declared_columns = (
            payload.get("columns")
            if isinstance(payload.get("columns"), list)
            else ()
        )
    else:
        return None

    rows: List[Dict[str, Any]] = []
    for item in raw_rows:
        if isinstance(item, Mapping):
            rows.append({str(key): value for key, value in item.items()})
        else:
            rows.append({"value": item})

    columns = _normalized_columns(rows, declared_columns)
    if not columns:
        return None

    source_total: Optional[int] = None
    for key in ("total", "row_count"):
        value = metadata.get(key)
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            source_total = value
            break
    truncated = bool(metadata.get("truncated"))
    if source_total is not None and source_total > len(rows):
        truncated = True
    return {
        "columns": columns,
        "rows": rows,
        "source_total": source_total,
        "truncated": truncated,
    }


def _decode_tool_content(content: Any) -> Optional[Any]:
    if isinstance(content, (dict, list)):
        return content
    if not isinstance(content, str):
        return None
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return None


class SavedResultStore:
    """Temporary SQLite database containing relations from one chat run."""

    def __init__(self) -> None:
        self._temp_dir = tempfile.TemporaryDirectory(
            prefix="etl_agent_saved_results_"
        )
        self.path = Path(self._temp_dir.name) / "saved_results.db"
        self._lock = RLock()
        self._descriptors: Dict[str, SavedResultDescriptor] = {}
        self._tables: Dict[str, str] = {}
        sqlite3.connect(self.path).close()

    def close(self) -> None:
        self._temp_dir.cleanup()

    def descriptors(self) -> List[SavedResultDescriptor]:
        with self._lock:
            return [item.model_copy(deep=True) for item in self._descriptors.values()]

    def descriptor(self, result_ref: str) -> Optional[SavedResultDescriptor]:
        with self._lock:
            item = self._descriptors.get(str(result_ref or "").strip())
            return item.model_copy(deep=True) if item is not None else None

    def save_payload(
        self,
        *,
        source_tool: str,
        payload: Any,
    ) -> Optional[SavedResultDescriptor]:
        tabular = _tabular_payload(payload)
        if tabular is None:
            return None

        rows: List[Dict[str, Any]] = tabular["rows"]
        column_names: List[str] = tabular["columns"]
        column_types = [
            _sqlite_type(row.get(name) for row in rows)
            for name in column_names
        ]
        result_ref = f"saved_{uuid4().hex}"

        with self._lock:
            table_name = f"saved_result_{len(self._descriptors) + 1}"
            definitions = ", ".join(
                f"{_quote_identifier(name)} {sqlite_type}"
                for name, sqlite_type in zip(column_names, column_types)
            )
            conn = sqlite3.connect(self.path)
            try:
                conn.execute(
                    f"CREATE TABLE {_quote_identifier(table_name)} ({definitions})"
                )
                if rows:
                    placeholders = ", ".join("?" for _ in column_names)
                    insert_sql = (
                        f"INSERT INTO {_quote_identifier(table_name)} "
                        f"VALUES ({placeholders})"
                    )
                    conn.executemany(
                        insert_sql,
                        [
                            tuple(_json_value(row.get(name)) for name in column_names)
                            for row in rows
                        ],
                    )
                conn.commit()
            finally:
                conn.close()

            descriptor = SavedResultDescriptor(
                result_ref=result_ref,
                source_tool=str(source_tool or "unknown_tool"),
                row_count=len(rows),
                source_total=tabular["source_total"],
                truncated=bool(tabular["truncated"]),
                columns=[
                    SavedResultColumn(name=name, sqlite_type=sqlite_type)
                    for name, sqlite_type in zip(column_names, column_types)
                ],
            )
            self._tables[result_ref] = table_name
            self._descriptors[result_ref] = descriptor
            logger.info(
                "Saved SQLite tool result: ref=%s source_tool=%s rows=%s "
                "truncated=%s columns=%s",
                result_ref,
                descriptor.source_tool,
                descriptor.row_count,
                descriptor.truncated,
                [column.name for column in descriptor.columns],
            )
            return descriptor.model_copy(deep=True)

    def query(
        self,
        *,
        result_ref: str,
        query: str,
        preview_limit: int,
    ) -> Dict[str, Any]:
        clean_ref = str(result_ref or "").strip()
        text = str(query or "").strip()
        validation_error = _validate_readonly_sql(text)
        if validation_error:
            return {
                "error": validation_error,
                "result_ref": clean_ref,
                "query": text,
            }

        with self._lock:
            descriptor = self._descriptors.get(clean_ref)
            table_name = self._tables.get(clean_ref)
        if descriptor is None or table_name is None:
            return {
                "error": "Saved result not found in the current coordinator run",
                "result_ref": clean_ref,
                "query": text,
            }

        limit = clamped_int(preview_limit, 20, 0, MAX_SAVED_RESULT_QUERY_ROWS)
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute(
                "CREATE TEMP VIEW result AS SELECT * FROM "
                + _quote_identifier(table_name)
            )
            conn.execute("PRAGMA query_only = ON")
            if hasattr(conn, "set_authorizer"):
                allowed_tables = {"result", table_name}

                def authorizer(
                    action_code: int,
                    arg1: Optional[str],
                    arg2: Optional[str],
                    database_name: Optional[str],
                    trigger_name: Optional[str],
                ) -> int:
                    base = _readonly_sql_authorizer(
                        action_code,
                        arg1,
                        arg2,
                        database_name,
                        trigger_name,
                    )
                    if base != sqlite3.SQLITE_OK:
                        return base
                    if (
                        action_code == sqlite3.SQLITE_READ
                        and str(arg1 or "") not in allowed_tables
                    ):
                        return sqlite3.SQLITE_DENY
                    return sqlite3.SQLITE_OK

                conn.set_authorizer(authorizer)

            cursor = conn.execute(text)
            columns = [item[0] for item in (cursor.description or [])]
            fetched = cursor.fetchmany(limit + 1)
            truncated = len(fetched) > limit
            rows = [dict(row) for row in fetched[:limit]]
            logger.info(
                "Queried saved SQLite result: ref=%s returned_rows=%s "
                "truncated=%s",
                clean_ref,
                len(rows),
                truncated,
            )
            return {
                "result_ref": clean_ref,
                "query": text,
                "input_row_count": descriptor.row_count,
                "input_truncated": descriptor.truncated,
                "columns": columns,
                "rows": rows,
                "returned_rows": len(rows),
                "truncated": truncated,
                "max_inline_rows": limit,
            }
        except sqlite3.Error as exc:
            logger.exception("Saved result SQL execution failed")
            return {
                "error": "Saved result SQL query failed",
                "error_message": str(exc),
                "result_ref": clean_ref,
                "query": text,
            }
        finally:
            conn.close()


_ACTIVE_SAVED_RESULT_STORE: ContextVar[Optional[SavedResultStore]] = ContextVar(
    "active_saved_result_store",
    default=None,
)


@contextmanager
def saved_result_store_scope() -> Iterator[SavedResultStore]:
    """Create and clean one isolated store for a coordinator execution."""
    store = SavedResultStore()
    token = _ACTIVE_SAVED_RESULT_STORE.set(store)
    try:
        yield store
    finally:
        _ACTIVE_SAVED_RESULT_STORE.reset(token)
        store.close()


def get_active_saved_result_store() -> Optional[SavedResultStore]:
    return _ACTIVE_SAVED_RESULT_STORE.get()


def persist_sqlite_tool_message(message: ToolMessage) -> ToolMessage:
    """Materialize a successful SQLite tool message and attach its schema."""
    store = get_active_saved_result_store()
    if store is None or str(message.name or "") not in SQLITE_RESULT_TOOL_NAMES:
        return message
    if getattr(message, "status", None) == "error":
        return message

    payload = _decode_tool_content(message.content)
    if not isinstance(payload, (dict, list)):
        return message
    if isinstance(payload, dict) and payload.get("error"):
        return message
    if isinstance(payload, dict) and isinstance(payload.get("saved_result"), dict):
        return message

    descriptor = store.save_payload(
        source_tool=str(message.name or "unknown_tool"),
        payload=payload,
    )
    if descriptor is None:
        return message

    if isinstance(payload, dict):
        enriched = dict(payload)
    else:
        enriched = {"rows": payload}
    enriched["saved_result"] = descriptor.model_dump(mode="json")
    return message.model_copy(
        update={
            "content": json.dumps(
                enriched,
                ensure_ascii=False,
                default=str,
                separators=(",", ":"),
            )
        }
    )


def _descriptor_catalog(
    descriptors: Sequence[SavedResultDescriptor],
) -> str:
    items = []
    for descriptor in descriptors:
        columns = ", ".join(
            f"{_quote_identifier(column.name)} {column.sqlite_type}"
            for column in descriptor.columns
        )
        total = (
            f", source_total={descriptor.source_total}"
            if descriptor.source_total is not None
            else ""
        )
        items.append(
            f"- result_ref={descriptor.result_ref}; "
            f"source_tool={descriptor.source_tool}; "
            f"stored_rows={descriptor.row_count}{total}; "
            f"truncated={str(descriptor.truncated).lower()}; "
            f"schema: CREATE TABLE result ({columns})"
        )
    return "\n".join(items)


def bind_saved_result_schemas(
    tools: Sequence[BaseTool],
    task: str,
) -> tuple[BaseTool, ...]:
    """Bind matching saved-result schemas into the SQL tool description."""
    store = get_active_saved_result_store()
    if store is None:
        return tuple(tools)
    descriptors = store.descriptors()
    if not descriptors:
        return tuple(tools)

    clean_task = str(task or "")
    referenced = [
        item for item in descriptors if item.result_ref in clean_task
    ]
    visible = referenced or descriptors
    catalog = _descriptor_catalog(visible)
    bound: List[BaseTool] = []
    for item in tools:
        if item.name != "query_saved_result":
            bound.append(item)
            continue
        bound.append(
            item.model_copy(
                update={
                    "description": (
                        f"{item.description}\n\n"
                        "Доступные сохранённые результаты текущего coordinator-"
                        "запуска. Для выбранного result_ref SQL видит только "
                        "таблицу `result` с указанной схемой:\n"
                        f"{catalog}"
                    )
                }
            )
        )
    return tuple(bound)


@tool(parse_docstring=True)
def query_saved_result(
    result_ref: str,
    query: str,
    preview_limit: int = 20,
) -> Dict[str, Any]:
    """Выполнить произвольный read-only SQL по сохранённому результату tool.

    Используй только когда task содержит точный result_ref предыдущего worker.
    Схема доступного результата передаётся вместе с description этого tool.
    В SQL выбранный результат всегда называется `result`; другие таблицы и
    основная SQLite-база недоступны. Если schema помечена truncated=true, запрос
    анализирует только сохранённые строки preview и не доказывает свойства
    полного исходного набора. Поддерживается ровно один SELECT, WITH или EXPLAIN
    QUERY PLAN без мутаций.

    Args:
        result_ref: Точный непрозрачный идентификатор сохранённого результата.
        query: Один read-only SQL-запрос к таблице result по переданной схеме.
        preview_limit: Максимум возвращаемых строк, от 0 до 100.
    """
    store = get_active_saved_result_store()
    if store is None:
        return {
            "error": "No active saved-result store",
            "result_ref": str(result_ref or "").strip(),
            "query": str(query or "").strip(),
        }
    return store.query(
        result_ref=result_ref,
        query=query,
        preview_limit=preview_limit,
    )


__all__ = [
    "MAX_SAVED_RESULT_QUERY_ROWS",
    "SQLITE_RESULT_TOOL_NAMES",
    "SavedResultColumn",
    "SavedResultDescriptor",
    "SavedResultStore",
    "bind_saved_result_schemas",
    "get_active_saved_result_store",
    "persist_sqlite_tool_message",
    "query_saved_result",
    "saved_result_store_scope",
]
