"""Build a semantic catalog and generate summary text for one uploaded file."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Sequence, Tuple

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser

from storage.database import (
    get_db_connection,
    update_file_description,
    update_file_summary,
)
from .agent import chat_model

try:
    from langfuse import observe
    from .observability import get_callback_handler
except ImportError:
    def observe(*args, **kwargs):
        def decorator(func):
            return func

        return decorator

    def get_callback_handler():
        return None


MAX_SUBJECT_AREAS = 12
MAX_TABLE_DESCRIPTIONS = 20
MAX_VIEW_DESCRIPTIONS = 20
MAX_ATTRIBUTE_DESCRIPTIONS = 25
MAX_FIELD_DESCRIPTIONS = 20
MAX_METRIC_DESCRIPTIONS = 10
SUMMARY_TEXT_CHAR_LIMIT = 300

SYSTEM_PROMPT = (
    "Сделай краткое бизнес-саммари и описание на русском языке по каталогу "
    "описаний таблиц, представлений, атрибутов и полей."
)
SUMMARY_OUTPUT_REQUIREMENTS = """
Саммари: один цельный абзац из 3–5 предложений, не более 1200 символов.
Описание: один короткий абзац из 2–3 предложений, не более 500 символов.
Опирайся только на переданные описания таблиц, представлений, атрибутов и полей.
Сформулируй предметные области, сущности и бизнес-процессы спецификации.
Не перечисляй типы артефактов (S2T-строки, SQL, внешние ключи, исключённые листы),
не описывай структуру документа и не придумывай отсутствующие факты.
Не упоминай JSON, Excel, файл, документ, листы, выборку строк или процесс анализа.
Верни только JSON с непустыми строками summary и description, без Markdown.
""".strip()
SUMMARY_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "business_summary",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "summary": {"type": "string", "minLength": 1, "maxLength": 1200},
                "description": {"type": "string", "minLength": 1, "maxLength": 500},
            },
            "required": ["summary", "description"],
            "additionalProperties": False,
        },
    },
}

SUBJECT_AREA_COLUMNS = ("Предметная область",)
VIEW_NAME_COLUMNS = ("Таблица", "Представление")
VIEW_DESCRIPTION_COLUMNS = ("Описание таблицы",)
TARGET_TABLE_COLUMNS = ("Таблица-приемник",)
TARGET_TABLE_DESCRIPTION_COLUMNS = ("Описание целевой таблицы",)
FIELD_NAME_COLUMNS = ("Поле приемника", "Поле", "Атрибут")
FIELD_DESCRIPTION_COLUMNS = (
    "Описание поля приемника",
    "Описание поля источника",
    "Описание поля",
    "Описание атрибута",
)
ENTITY_COLUMNS = ("Сущность",)
ATTRIBUTE_NAME_COLUMNS = ("Атрибут",)
METRIC_CODE_COLUMNS = ("Код выборки данных",)
METRIC_DESCRIPTION_COLUMNS = ("Описание",)


def _invoke(messages: List[BaseMessage], *, structured: bool = False) -> str:
    model = chat_model
    handler = get_callback_handler()
    if handler:
        model = model.with_config({"callbacks": [handler]})
    kwargs = (
        {"response_format": SUMMARY_RESPONSE_FORMAT}
        if structured and getattr(chat_model, "supports_json_schema", False) is True
        else {}
    )
    return StrOutputParser().invoke(model.invoke(messages, **kwargs)).strip()


def call_gigachat(user_content: str) -> str:
    """Invoke the configured chat model for a standalone text rewrite."""
    return _invoke([HumanMessage(content=user_content)])


def _file_text_fields(file_id: int) -> Dict[str, Any]:
    conn = get_db_connection()
    try:
        row = conn.execute(
            "SELECT file_id, filename, summary, description FROM files WHERE file_id = ?",
            (file_id,),
        ).fetchone()
    finally:
        conn.close()
    if not row:
        raise ValueError(f"File {file_id} not found")
    return dict(row)


def _sheet_columns(headers_json: Optional[str]) -> List[Dict[str, Any]]:
    try:
        headers = json.loads(headers_json or "[]")
    except (TypeError, json.JSONDecodeError):
        headers = []
    columns = []
    for position, item in enumerate(headers):
        if not isinstance(item, dict):
            continue
        path = item.get("path") if isinstance(item.get("path"), list) else []
        name = str(item.get("flat") or "").strip() or " > ".join(
            str(part) for part in path if part is not None and str(part).strip()
        )
        if not name:
            continue
        try:
            index = int(item.get("index", position))
        except (TypeError, ValueError):
            index = position
        columns.append({"index": index, "name": name, "column_id": index + 1})
    return sorted(columns, key=lambda column: column["index"])


def _load_rows(
    cursor: Any,
    file_id: int,
    sheet_name: str,
) -> List[Dict[int, Any]]:
    rows: Dict[int, Dict[int, Any]] = {}
    for item in cursor.execute(
        """
        SELECT row_num, column_id, value
        FROM data
        WHERE file_id = ? AND table_name = ? COLLATE NOCASE
        ORDER BY row_num, id
        """,
        (file_id, sheet_name),
    ).fetchall():
        rows.setdefault(int(item["row_num"]), {})[int(item["column_id"])] = item["value"]
    return list(rows.values())


def _pick_column(column_ids: Dict[str, int], candidates: Sequence[str]) -> Optional[int]:
    return next((column_ids[name] for name in candidates if name in column_ids), None)


def _compact(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).replace("\r\n", "\n").strip()
    if not text:
        return None
    return text if len(text) <= SUMMARY_TEXT_CHAR_LIMIT else f"{text[:299]}…"


def _value(row: Dict[int, Any], column_id: Optional[int]) -> Optional[str]:
    return _compact(row.get(column_id)) if column_id is not None else None


def _dedupe(records: List[Any], key_fields: Sequence[str] = ()) -> List[Any]:
    seen, result = set(), []
    for record in records:
        key = (
            tuple(str(record.get(field) or "") for field in key_fields)
            if isinstance(record, dict)
            else str(record or "")
        )
        empty = not any(key) if isinstance(key, tuple) else not key
        if empty:
            continue
        if key not in seen:
            seen.add(key)
            result.append(record)
    return result


def _evenly_spaced_items(items: List[Any], limit: int) -> List[Any]:
    if len(items) <= limit:
        return items
    if limit <= 1:
        return items[:1]
    last = len(items) - 1
    indexes = {round(index * last / (limit - 1)) for index in range(limit)}
    return [item for index, item in enumerate(items) if index in indexes]


def _extract_sheet_semantics(
    columns: List[Dict[str, Any]], rows: List[Dict[int, Any]]
) -> Dict[str, List[Any]]:
    ids = {column["name"]: column["column_id"] for column in columns}
    area_id = _pick_column(ids, SUBJECT_AREA_COLUMNS)
    view_id = _pick_column(ids, VIEW_NAME_COLUMNS)
    view_description_id = _pick_column(ids, VIEW_DESCRIPTION_COLUMNS)
    table_id = _pick_column(ids, TARGET_TABLE_COLUMNS)
    table_description_id = _pick_column(ids, TARGET_TABLE_DESCRIPTION_COLUMNS)
    field_id = _pick_column(ids, FIELD_NAME_COLUMNS)
    field_description_id = _pick_column(ids, FIELD_DESCRIPTION_COLUMNS)
    entity_id = _pick_column(ids, ENTITY_COLUMNS)
    attribute_id = _pick_column(ids, ATTRIBUTE_NAME_COLUMNS)
    metric_id = _pick_column(ids, METRIC_CODE_COLUMNS)
    metric_description_id = _pick_column(ids, METRIC_DESCRIPTION_COLUMNS)
    result: Dict[str, List[Any]] = {
        "subject_areas": [], "views": [], "tables": [],
        "attributes": [], "fields": [], "metrics": [],
    }
    for row in rows:
        area = _value(row, area_id)
        if area:
            result["subject_areas"].append(area)

        view, view_description = _value(row, view_id), _value(row, view_description_id)
        if view and view_description and table_id is None:
            result["views"].append({"name": view, "description": view_description})

        table, table_description = _value(row, table_id), _value(row, table_description_id)
        if table and table_description:
            record = {"name": table, "description": table_description}
            if area:
                record["subject_area"] = area
            result["tables"].append(record)

        field, field_description = _value(row, field_id), _value(row, field_description_id)
        if table_description_id is not None and field and field_description:
            record = {"field": field, "description": field_description}
            if table:
                record["table"] = table
            result["fields"].append(record)

        entity, attribute = _value(row, entity_id), _value(row, attribute_id)
        if entity and attribute and field_description:
            result["attributes"].append(
                {"entity": entity, "attribute": attribute, "description": field_description}
            )

        metric, metric_description = _value(row, metric_id), _value(row, metric_description_id)
        if metric and metric_description:
            result["metrics"].append(
                {"code": metric, "description": metric_description}
            )
    return result


def _limit_catalog(catalog: Dict[str, List[Any]]) -> None:
    catalog["subject_areas"] = _dedupe(catalog["subject_areas"])[:MAX_SUBJECT_AREAS]
    catalog["views"] = _dedupe(catalog["views"], ("name", "description"))[:MAX_VIEW_DESCRIPTIONS]
    catalog["tables"] = _evenly_spaced_items(
        _dedupe(catalog["tables"], ("name", "description")), MAX_TABLE_DESCRIPTIONS
    )
    catalog["attributes"] = _evenly_spaced_items(
        _dedupe(catalog["attributes"], ("entity", "attribute", "description")),
        MAX_ATTRIBUTE_DESCRIPTIONS,
    )
    catalog["fields"] = _evenly_spaced_items(
        _dedupe(catalog["fields"], ("table", "field", "description")),
        MAX_FIELD_DESCRIPTIONS,
    )
    catalog["metrics"] = _dedupe(catalog["metrics"], ("code", "description"))[:MAX_METRIC_DESCRIPTIONS]


def fetch_file_data(file_id: int) -> Dict[str, Any]:
    """Return the semantic catalog used by the summary model."""
    catalog: Dict[str, List[Any]] = {
        "subject_areas": [], "views": [], "tables": [], "attributes": [],
        "fields": [], "metrics": [], "catalog_tables": [],
    }
    conn = get_db_connection()
    try:
        file_row = conn.execute(
            "SELECT filename FROM files WHERE file_id = ?", (file_id,)
        ).fetchone()
        if not file_row:
            raise ValueError(f"File {file_id} not found")
        sheets = conn.execute(
            """
            SELECT sheet_name, headers_json
            FROM file_sheet_headers AS headers
            WHERE file_id = ?
              AND EXISTS (
                  SELECT 1
                  FROM data
                  WHERE data.file_id = headers.file_id
                    AND data.table_name = headers.sheet_name COLLATE NOCASE
              )
            ORDER BY sheet_name
            """,
            (file_id,),
        ).fetchall()
        for sheet in sheets:
            extracted = _extract_sheet_semantics(
                _sheet_columns(sheet["headers_json"]),
                _load_rows(conn, file_id, str(sheet["sheet_name"])),
            )
            for key, values in extracted.items():
                catalog[key].extend(values)

        persisted = conn.execute(
            """
            SELECT 'target_tables' AS catalog, table_name, description, row_num
            FROM target_tables WHERE file_id = ?
            UNION ALL
            SELECT 'source_tables', table_name, description, row_num
            FROM source_tables WHERE file_id = ?
            ORDER BY row_num
            """,
            (file_id, file_id),
        ).fetchall()
        catalog["catalog_tables"] = _dedupe(
            [
                {
                    "name": _compact(row["table_name"]) or "",
                    "description": _compact(row["description"]) or "",
                    "catalog": row["catalog"],
                }
                for row in persisted
                if _compact(row["table_name"]) and _compact(row["description"])
            ],
            ("name", "description"),
        )
    finally:
        conn.close()
    _limit_catalog(catalog)
    return {"filename": file_row["filename"], "semantic_catalog": catalog}


def build_summary_payload(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """Build the compact JSON payload sent to the summarizer LLM."""
    catalog = snapshot.get("semantic_catalog") or {}
    return {
        "focus": "table_and_attribute_descriptions",
        "filename": snapshot.get("filename"),
        **{
            key: catalog.get(key) or []
            for key in (
                "subject_areas", "views", "tables", "attributes",
                "fields", "metrics", "catalog_tables",
            )
        },
    }


@observe()
def generate_analysis_texts(file_id: int) -> Tuple[str, str]:
    payload = json.dumps(build_summary_payload(fetch_file_data(file_id)), ensure_ascii=False)
    messages = [
        SystemMessage(content=f"{SYSTEM_PROMPT}\n\n{SUMMARY_OUTPUT_REQUIREMENTS}"),
        HumanMessage(content=payload),
    ]
    try:
        result = json.loads(_invoke(messages, structured=True))
    except json.JSONDecodeError as exc:
        raise ValueError("LLM returned invalid summary JSON") from exc
    if not isinstance(result, dict):
        raise ValueError("LLM summary response must be a JSON object")
    summary = str(result.get("summary") or "").strip()
    description = str(result.get("description") or "").strip()
    if not summary or not description:
        raise ValueError("LLM returned empty summary or description")
    return summary, description


def generate_summary(file_id: int) -> str:
    return generate_analysis_texts(file_id)[0]


def summarize_file(file_id: int, save: bool = True) -> str:
    summary, description = generate_analysis_texts(file_id)
    if save:
        update_file_summary(file_id, summary)
        update_file_description(file_id, description)
    return summary


def generate_description_from_summary(summary: str) -> str:
    return call_gigachat(
        "Сформируй по бизнес-саммари краткое описание на русском языке: 2–3 "
        "предложения об областях, сущностях и процессах из описаний таблиц и "
        "атрибутов. Не упоминай структуру документа, S2T-артефакты, SQL, Excel "
        f"или файл. Верни только описание.\n\nБизнес-саммари:\n{summary}"
    ).strip()


def ensure_file_description(
    file_id: int,
    refresh: bool = False,
    save: bool = True,
    summary_override: Optional[str] = None,
) -> str:
    fields = _file_text_fields(file_id)
    cached = str(fields.get("description") or "").strip()
    if cached and not refresh:
        return cached
    summary = str(summary_override or fields.get("summary") or "").strip()
    if not summary:
        summary, description = generate_analysis_texts(file_id)
        if save:
            update_file_summary(file_id, summary)
            update_file_description(file_id, description)
        return description
    description = generate_description_from_summary(summary)
    if save:
        update_file_description(file_id, description)
    return description


def generate_description_update_from_user_query(
    current_description: str,
    summary: str,
    user_query: str,
) -> str:
    return call_gigachat(
        "Обнови краткое описание по уточнению пользователя. Верни только один "
        "абзац на русском языке из 2–4 предложений. Опирайся на описание, саммари "
        "и запрос; не упоминай структуру документа, S2T-артефакты, SQL, Excel "
        "или файл и не придумывай факты.\n\n"
        f"Описание:\n{current_description}\n\nСаммари:\n{summary}\n\n"
        f"Запрос пользователя:\n{user_query}"
    ).strip()


def update_file_description_from_user_query(
    file_id: int,
    user_query: str,
    save: bool = True,
) -> str:
    request_text = str(user_query or "").strip()
    if not request_text:
        raise ValueError("user_query must be non-empty")
    current = ensure_file_description(file_id, refresh=False, save=save)
    summary = str(_file_text_fields(file_id).get("summary") or "").strip()
    updated = generate_description_update_from_user_query(
        current_description=current,
        summary=summary,
        user_query=request_text,
    )
    if save:
        update_file_description(file_id, updated)
    return updated
