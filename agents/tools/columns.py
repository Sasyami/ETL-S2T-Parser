"""Read-only tools for structured source/target column catalogs."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

from langchain_core.tools import tool

from .common import clamped_int


ColumnScope = Literal["all_tables", "source_columns", "target_columns"]

COLUMN_RESULT_FIELDS = (
    "record_id",
    "column_role",
    "file_id",
    "filename",
    "sheet_name",
    "row_num",
    "table_name",
    "column_name",
    "data_type",
    "primary_key",
    "not_null",
    "description",
)


def _catalog_branches(scope: ColumnScope) -> Tuple[Tuple[str, str], ...]:
    if scope == "source_columns":
        return (("source_columns", "source"),)
    if scope == "target_columns":
        return (("target_columns", "target"),)
    return (("source_columns", "source"), ("target_columns", "target"))


def _catalog_cte(scope: ColumnScope) -> str:
    return "\nUNION ALL\n".join(
        f"""
        SELECT catalog.id AS record_id, '{role}' AS column_role,
               catalog.file_id, files.filename, catalog.sheet_name,
               catalog.row_num, catalog.table_name, catalog.column_name,
               catalog.data_type, catalog.primary_key, catalog.not_null,
               catalog.description
        FROM {table_name} AS catalog
        LEFT JOIN files ON files.file_id = catalog.file_id
        """.strip()
        for table_name, role in _catalog_branches(scope)
    )


def _clean_exact_filter(value: Optional[str], name: str) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if len(text) > 300:
        raise ValueError(f"{name} too long")
    return text


def _subset_conditions(
    *,
    file_id: Optional[int],
    table_name: Optional[str],
    column_name: Optional[str],
    data_type: Optional[str],
    primary_key: Optional[bool],
    not_null: Optional[bool],
    needle: Optional[str] = None,
) -> Tuple[List[str], List[Any], Dict[str, Any]]:
    conditions: List[str] = []
    params: List[Any] = []
    filters: Dict[str, Any] = {}

    if file_id is not None:
        clean_file_id = int(file_id)
        if clean_file_id <= 0:
            raise ValueError("file_id must be positive")
        conditions.append("file_id = ?")
        params.append(clean_file_id)
        filters["file_id"] = clean_file_id

    for field, value in (
        ("table_name", table_name),
        ("column_name", column_name),
        ("data_type", data_type),
    ):
        clean_value = _clean_exact_filter(value, field)
        if clean_value is None:
            continue
        conditions.append(f"TRIM({field}) = ? COLLATE NOCASE")
        params.append(clean_value)
        filters[field] = clean_value

    for field, value in (("primary_key", primary_key), ("not_null", not_null)):
        if value is None:
            continue
        conditions.append(f"{field} = ?")
        params.append(1 if value else 0)
        filters[field] = bool(value)

    if needle is not None:
        pattern = f"%{needle}%"
        conditions.append(
            "(" + " OR ".join(
                f"COALESCE({field}, '') LIKE ? COLLATE NOCASE"
                for field in (
                    "table_name",
                    "column_name",
                    "data_type",
                    "description",
                )
            ) + ")"
        )
        params.extend([pattern] * 4)
    return conditions, params, filters


def _query_catalog(
    *,
    scope: ColumnScope,
    limit: int,
    columns: Optional[Sequence[str]],
    file_id: Optional[int],
    table_name: Optional[str],
    column_name: Optional[str],
    data_type: Optional[str],
    primary_key: Optional[bool],
    not_null: Optional[bool],
    needle: Optional[str] = None,
) -> Dict[str, Any]:
    selected = list(columns or COLUMN_RESULT_FIELDS)
    if not selected:
        return {"error": "columns must not be empty", "rows": []}
    unknown = [field for field in selected if field not in COLUMN_RESULT_FIELDS]
    if unknown:
        return {
            "error": f"unknown columns: {', '.join(unknown)}",
            "allowed_columns": list(COLUMN_RESULT_FIELDS),
            "rows": [],
        }
    selected = list(dict.fromkeys(selected))
    try:
        conditions, params, filters = _subset_conditions(
            file_id=file_id,
            table_name=table_name,
            column_name=column_name,
            data_type=data_type,
            primary_key=primary_key,
            not_null=not_null,
            needle=needle,
        )
    except (TypeError, ValueError) as exc:
        return {"error": str(exc), "rows": []}

    where_sql = "WHERE " + " AND ".join(conditions) if conditions else ""
    clean_limit = clamped_int(limit, 50, 1, 100)
    select_sql = ", ".join(f'"{field}"' for field in selected)
    query = f"""
        WITH catalog AS (
            {_catalog_cte(scope)}
        ), matched AS (
            SELECT * FROM catalog
            {where_sql}
        )
        SELECT {select_sql}, COUNT(*) OVER () AS __total_matches
        FROM matched
        ORDER BY column_role, file_id, table_name, column_name, record_id
        LIMIT ?
    """
    params.append(clean_limit)

    from storage.database import get_db_connection

    conn = get_db_connection()
    try:
        fetched = conn.execute(query, params).fetchall()
    finally:
        conn.close()
    total_matches = int(fetched[0]["__total_matches"]) if fetched else 0
    rows = []
    for row in fetched:
        item = dict(row)
        item.pop("__total_matches", None)
        rows.append(item)
    return {
        "scope": scope,
        "filters": filters,
        "columns": selected,
        "total_matches": total_matches,
        "returned_rows": len(rows),
        "rows": rows,
    }


@tool(parse_docstring=True)
def list_column_catalog(
    scope: ColumnScope,
    limit: int = 50,
    columns: Optional[List[str]] = None,
    file_id: Optional[int] = None,
    table_name: Optional[str] = None,
    column_name: Optional[str] = None,
    data_type: Optional[str] = None,
    primary_key: Optional[bool] = None,
    not_null: Optional[bool] = None,
) -> Dict[str, Any]:
    """Получить подвыборку каталога колонок по точным структурным фильтрам.

    Используй для известной роли и точных значений table_name/column_name либо
    для выборки по file_id, типу, PK и not-null. Обязательный scope
    all_tables объединяет source_columns и target_columns; ролевые scope читают
    только один каталог. Выбирай all_tables только если task явно требует обе
    стороны либо не ограничивает роль колонки. Один вызов применяет одинаковые table_name и
    column_name ко всему выбранному scope: разные source/target пары получай
    отдельными вызовами, не ищи target по идентификаторам source. Для фрагмента имени или текста сначала используй
    search_column_catalog, для смысла описания — semantic_search_descriptions.
    Передавай все относящиеся к операции точные фильтры, явно названные в task:
    если названы file_id, table_name и not_null, нельзя опускать table_name и
    расширять результат до всех таблиц файла.
    Без columns возвращает все публичные поля, но никогда не возвращает BLOB.

    Args:
        scope: Обязательная область: all_tables, source_columns или target_columns; all_tables выбирай только для обеих сторон или неизвестной роли.
        limit: Максимальное число строк, от 1 до 100.
        columns: Опциональная подвыборка возвращаемых публичных полей.
        file_id: Опциональный точный идентификатор явно выбранной загрузки.
        table_name: Опциональное точное полное имя таблицы; обязательно передай, если оно явно задано task.
        column_name: Опциональное точное имя колонки.
        data_type: Опциональный точный тип данных.
        primary_key: Опциональный фильтр признака первичного ключа.
        not_null: Опциональный фильтр обязательности значения.
    """
    return _query_catalog(
        scope=scope,
        limit=limit,
        columns=columns,
        file_id=file_id,
        table_name=table_name,
        column_name=column_name,
        data_type=data_type,
        primary_key=primary_key,
        not_null=not_null,
    )


@tool(parse_docstring=True)
def search_column_catalog(
    needle: str,
    scope: ColumnScope,
    limit: int = 50,
    file_id: Optional[int] = None,
    table_name: Optional[str] = None,
    data_type: Optional[str] = None,
    primary_key: Optional[bool] = None,
    not_null: Optional[bool] = None,
) -> Dict[str, Any]:
    """Найти колонки по подстроке внутри ограниченной структурной подвыборки.

    Ищет needle в table_name, column_name, data_type и description. Используй
    для неполного имени или неизвестной роли; scope и
    остальные фильтры сначала ограничивают подвыборку. После разрешения точных
    имён передавай их в list_column_catalog. Это не семантический поиск.

    Args:
        needle: Непустая искомая подстрока длиной до 300 символов.
        scope: Обязательная область: all_tables, source_columns или target_columns; all_tables выбирай только для обеих сторон или неизвестной роли.
        limit: Максимальное число строк, от 1 до 100.
        file_id: Опциональный точный идентификатор явно выбранной загрузки.
        table_name: Опциональное точное имя таблицы для ограничения поиска.
        data_type: Опциональный точный тип данных.
        primary_key: Опциональный фильтр признака первичного ключа.
        not_null: Опциональный фильтр обязательности значения.
    """
    text = str(needle or "").strip()
    if not text:
        return {"error": "needle must be non-empty", "rows": []}
    if len(text) > 300:
        return {"error": "needle too long", "rows": []}
    result = _query_catalog(
        scope=scope,
        limit=limit,
        columns=None,
        file_id=file_id,
        table_name=table_name,
        column_name=None,
        data_type=data_type,
        primary_key=primary_key,
        not_null=not_null,
        needle=text,
    )
    result["query"] = text
    return result


__all__ = [
    "COLUMN_RESULT_FIELDS",
    "list_column_catalog",
    "search_column_catalog",
]
