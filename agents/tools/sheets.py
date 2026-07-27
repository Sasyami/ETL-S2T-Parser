"""Tools for workbook sheets, headers, groups, and columns."""

import json
from typing import Any, Dict, List, Optional

from langchain_core.tools import tool

@tool(parse_docstring=True)
def list_sheets(file_id: int) -> List[str]:
    """Получить полный список имён Excel-листов одной сохранённой загрузки.

    Читает file_sheet_headers и возвращает только реальные имена листов, включая
    пропущенные при анализе листы, если они были сохранены в метаданных.

    Args:
        file_id: Числовой идентификатор загрузки из UI или resolve_file.
    """
    from storage.database import get_db_connection
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT sheet_name FROM file_sheet_headers WHERE file_id = ? ORDER BY sheet_name", (file_id,))
    rows = cursor.fetchall()
    conn.close()
    return [row["sheet_name"] for row in rows]

@tool(parse_docstring=True)
def list_file_sheet_headers(file_id: int) -> List[Dict[str, Any]]:
    """Получить подробные метаданные листов и распознанных Excel-заголовков.

    Для каждого листа возвращает sheet_id, статус пропуска, положение и глубину
    заголовка, число колонок, плоские названия и разобранные пути колонок из
    file_sheet_headers.headers_json.

    Args:
        file_id: Числовой идентификатор загрузки из UI или resolve_file.
    """
    from storage.database import get_db_connection

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT file_id, sheet_id, sheet_name, skipped, skip_reason,
               header_start_row, header_rows_count, nested_structure,
               columns_count, headers_json, headers_flat
        FROM file_sheet_headers
        WHERE file_id = ?
        ORDER BY sheet_name
        """,
        (file_id,),
    )
    rows = []
    for row in cursor.fetchall():
        item = dict(row)
        try:
            item["headers"] = json.loads(item.get("headers_json") or "[]")
        except json.JSONDecodeError:
            item["headers"] = []
        rows.append(item)
    conn.close()
    return rows


@tool(parse_docstring=True)
def list_sheet_group_classifications(file_id: int) -> Dict[str, Any]:
    """Получить сохранённо-детерминированную классификацию групп Excel-листов.

    Показывает сопоставление листов с группами из sheet_groups.json, направления
    ETL-слоёв, несопоставленные листы и проверочный отчёт. Инструментальный вызов
    не обращается к LLM и не записывает новые алиасы.

    Args:
        file_id: Числовой идентификатор загрузки, листы которой нужно классифицировать.
    """
    from agents.sheet_group_classifier import classify_file_sheet_groups

    return classify_file_sheet_groups(file_id, use_llm=False, persist_aliases=False)


# ----------------------------------------------------------------------
# Tool: list_columns
# ----------------------------------------------------------------------
@tool(parse_docstring=True)
def list_columns(sheet_id: int | str) -> Dict[str, Any]:
    """Получить распознанные колонки одного Excel-листа в сохранённом порядке.

    Возвращает структурированный объект с разрешённым числовым sheet_id,
    количеством колонок и массивом columns. Для каждой колонки показывает
    стабильный column_id, плоское сохранённое имя и исходную позицию. Если имя
    листа неоднозначно, не выбирает файл самовольно, а возвращает совпадения.

    Args:
        sheet_id: Числовой ID листа или его уникальное сохранённое имя без учёта регистра.
    """
    from storage.database import get_columns_by_sheet, get_db_connection

    requested = str(sheet_id).strip()
    if not requested:
        return {"error": "sheet_id must be non-empty", "columns": []}

    conn = get_db_connection()
    try:
        resolved_sheet_id: Optional[int] = None
        if requested.isdigit():
            resolved_sheet_id = int(requested)
        else:
            matches = conn.execute(
                """
                SELECT sheet_id, file_id, sheet_name
                FROM file_sheet_headers
                WHERE sheet_name = ? COLLATE NOCASE
                ORDER BY file_id, sheet_id
                """,
                (requested,),
            ).fetchall()
            if not matches:
                return {
                    "error": "Sheet not found",
                    "sheet": requested,
                    "columns": [],
                }
            if len(matches) > 1:
                return {
                    "error": "Multiple sheets have this name",
                    "sheet": requested,
                    "matches": [dict(row) for row in matches],
                    "columns": [],
                }
            resolved_sheet_id = int(matches[0]["sheet_id"])

        rows = get_columns_by_sheet(resolved_sheet_id)
        columns = [
            {
                "column_id": row["column_id"],
                "name": row["column_name_flat"],
                "index": row["column_index"],
            }
            for row in rows
        ]
        return {
            "sheet_id": resolved_sheet_id,
            "columns": columns,
            "column_count": len(columns),
        }
    finally:
        conn.close()
