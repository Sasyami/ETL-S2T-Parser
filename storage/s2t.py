"""Persistence API for validated S2T transformation records."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from .database import S2T_FIELDS, _sql_identifier, get_db_connection

logger = logging.getLogger(__name__)


def refresh_s2t_transformations(file_id: int) -> int:
    """Back-compatible public entrypoint: return only written row count."""
    from sheet_skills.s2t import run_s2t_extraction_subagent

    report = run_s2t_extraction_subagent(file_id)
    logger.info(
        "Refreshed %s S2T transformation rows for file %s via useful-column subagent",
        report.get("verification", {}).get("count", 0),
        file_id,
    )
    return int(report.get("verification", {}).get("count", 0))


def replace_s2t_transformations(file_id: int, records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Transactionally replace the global S2T result with validated records."""
    insert_columns = (
        "file_id",
        "sheet_id",
        "sheet_name",
        "row_num",
        *S2T_FIELDS,
        "raw_json",
    )
    columns_sql = ", ".join(_sql_identifier(column) for column in insert_columns)
    placeholders = ", ".join("?" for _ in insert_columns)
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("BEGIN")
        cursor.execute("DELETE FROM s2t_transformations")
        cursor.executemany(
            f"""
            INSERT INTO s2t_transformations
            ({columns_sql})
            VALUES ({placeholders})
            """,
            [
                (
                    row["file_id"],
                    row["sheet_id"],
                    row["sheet_name"],
                    row["row_num"],
                    *(row.get(field) for field in S2T_FIELDS),
                    row["raw_json"],
                )
                for row in records
            ],
        )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
    return {"file_id": file_id, "count": len(records)}


def clear_s2t_transformations(file_id: int) -> int:
    """Delete generated S2T transformation rows for one workbook."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) AS n FROM s2t_transformations WHERE file_id = ?", (file_id,))
    deleted = int(cursor.fetchone()["n"])
    cursor.execute("DELETE FROM s2t_transformations WHERE file_id = ?", (file_id,))
    conn.commit()
    conn.close()
    logger.info("Cleared %s S2T transformation rows for file %s", deleted, file_id)
    return deleted


def list_s2t_transformations(
    file_id: Optional[int] = None,
    limit: int = 200,
    q: Optional[str] = None,
) -> Dict[str, Any]:
    """Return minimal stored S2T transformations for UI/API browsing."""
    clean_limit = max(1, min(int(limit or 200), 1000))
    params: List[Any] = []
    where = ["IFNULL(row_num, 0) >= 0"]
    if file_id is not None:
        where.insert(0, "file_id = ?")
        params.append(int(file_id))
    if q:
        pattern = f"%{q.strip()}%"
        where.append(
            "(" + " OR ".join(f"{_sql_identifier(field)} LIKE ?" for field in S2T_FIELDS) + ")"
        )
        params.extend([pattern] * len(S2T_FIELDS))
    where_sql = " AND ".join(where)
    selected_fields_sql = ", ".join(_sql_identifier(field) for field in S2T_FIELDS)

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(f"SELECT COUNT(*) AS n FROM s2t_transformations WHERE {where_sql}", params)
    total = int(cursor.fetchone()["n"])
    cursor.execute(
        f"""
        SELECT row_num, {selected_fields_sql}
        FROM s2t_transformations
        WHERE {where_sql}
        ORDER BY id
        LIMIT ?
        """,
        params + [clean_limit],
    )
    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    result = {
        "scope": "global" if file_id is None else "file",
        "total": total,
        "limit": clean_limit,
        "rows": rows,
    }
    if file_id is not None:
        result["file_id"] = int(file_id)
    return result


def verify_s2t_transformations(file_id: int, limit: int = 5) -> Dict[str, Any]:
    """Return the stored row count and a small verification sample."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT COUNT(*) AS n
        FROM s2t_transformations
        WHERE file_id = ? AND IFNULL(row_num, 0) >= 0
        """,
        (file_id,),
    )
    count = int(cursor.fetchone()["n"])
    cursor.execute(
        f"""
        SELECT row_num, {", ".join(_sql_identifier(field) for field in S2T_FIELDS)}
        FROM s2t_transformations
        WHERE file_id = ? AND IFNULL(row_num, 0) >= 0
        ORDER BY row_num
        LIMIT ?
        """,
        (file_id, limit),
    )
    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return {"file_id": file_id, "count": count, "rows": rows}


def summarize_s2t_transformations(
    group_by: str = "target",
    file_id: Optional[int] = None,
    min_related_tables: int = 1,
    limit: int = 100,
) -> Dict[str, Any]:
    """Aggregate mapping counts and related tables by source or target table."""
    if group_by not in {"source", "target"}:
        return {"error": "group_by must be 'source' or 'target'"}

    expected_fields = {
        "source_table",
        "source_field",
        "target_table",
        "target_field",
        "transformation_rule",
    }
    missing_fields = sorted(expected_fields - set(S2T_FIELDS))
    if missing_fields:
        return {
            "error": "Configured s2t_transformations fields do not support table summary",
            "missing_fields": missing_fields,
        }

    clean_file_id = int(file_id) if file_id is not None else None
    clean_min_related = max(1, min(int(min_related_tables or 1), 1000))
    clean_limit = max(1, min(int(limit or 100), 200))
    table_column = "source_table" if group_by == "source" else "target_table"
    mapped_field = "source_field" if group_by == "source" else "target_field"
    related_table_column = "target_table" if group_by == "source" else "source_table"

    scope_sql = ""
    params: List[Any] = []
    if clean_file_id is not None:
        scope_sql = "AND file_id = ?"
        params.append(clean_file_id)
    params.extend([clean_min_related, clean_limit])

    query = f"""
        SELECT
            {table_column} AS table_name,
            COUNT(*) AS mapping_count,
            COUNT(DISTINCT NULLIF(TRIM({mapped_field}), '')) AS field_count,
            COUNT(DISTINCT NULLIF(TRIM({related_table_column}), '')) AS related_table_count,
            GROUP_CONCAT(DISTINCT NULLIF(TRIM({related_table_column}), '')) AS related_tables,
            SUM(
                CASE
                    WHEN NULLIF(TRIM(transformation_rule), '') IS NOT NULL THEN 1
                    ELSE 0
                END
            ) AS mappings_with_rule
        FROM s2t_transformations
        WHERE NULLIF(TRIM({table_column}), '') IS NOT NULL
          {scope_sql}
        GROUP BY {table_column}
        HAVING COUNT(DISTINCT NULLIF(TRIM({related_table_column}), '')) >= ?
        ORDER BY related_table_count DESC, mapping_count DESC, table_name
        LIMIT ?
    """

    conn = get_db_connection()
    try:
        rows = conn.execute(query, params).fetchall()
    finally:
        conn.close()

    groups = []
    for row in rows:
        item = dict(row)
        related_tables = str(item.pop("related_tables") or "")
        item["related_tables"] = sorted(name for name in related_tables.split(",") if name)
        item["rule_coverage"] = (
            item["mappings_with_rule"] / item["mapping_count"]
            if item["mapping_count"]
            else 0.0
        )
        groups.append(item)

    return {
        "group_by": group_by,
        "file_id": clean_file_id,
        "min_related_tables": clean_min_related,
        "groups": groups,
        "group_count": len(groups),
    }


__all__ = [
    "clear_s2t_transformations",
    "list_s2t_transformations",
    "refresh_s2t_transformations",
    "replace_s2t_transformations",
    "summarize_s2t_transformations",
    "verify_s2t_transformations",
]
