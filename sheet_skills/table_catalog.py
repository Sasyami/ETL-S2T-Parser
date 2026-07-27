"""Sheet skill for source/target table catalogs from configured workbook sheets."""

from __future__ import annotations

import json
from difflib import SequenceMatcher
from typing import Any, Dict, Iterable, List, Optional, Tuple

from config.column_mapping import (
    add_field_aliases,
    get_field_aliases,
    header_matches_alias,
    normalize_column_alias,
)
from storage.database import _sql_identifier, get_columns_by_sheet, get_db_connection
from config.useful_columns import get_usefull_col_extraction_target


TABLE_CATALOG_TARGETS = ("source_tables", "target_tables")
FUZZY_MATCH_THRESHOLD = 0.70


def _parse_header(header_json: Optional[str], flat_name: str) -> List[Any]:
    if header_json:
        try:
            parsed = json.loads(header_json)
            if isinstance(parsed, list):
                return parsed
            return [parsed]
        except (TypeError, json.JSONDecodeError):
            pass
    return [part.strip() for part in str(flat_name or "").split(">") if part.strip()]


def _header_candidates(column: Dict[str, Any]) -> List[str]:
    header = [
        str(part)
        for part in column.get("column_header") or []
        if part is not None and str(part).strip()
    ]
    candidates = [str(column.get("column_name_flat") or ""), " ".join(header)]
    candidates.extend(header)
    if header:
        candidates.append(header[-1])
    return [candidate for candidate in candidates if candidate.strip()]


def _best_alias_score(
    column: Dict[str, Any], aliases: Iterable[str]
) -> Tuple[float, Optional[str], Optional[str]]:
    best_score = 0.0
    best_alias: Optional[str] = None
    best_candidate: Optional[str] = None
    for candidate in _header_candidates(column):
        normalized_candidate = normalize_column_alias(candidate)
        if not normalized_candidate:
            continue
        for alias in aliases:
            normalized_alias = normalize_column_alias(alias)
            if not normalized_alias:
                continue
            if normalized_alias == normalized_candidate:
                return 1.0, alias, candidate
            score = SequenceMatcher(None, normalized_candidate, normalized_alias).ratio()
            if score > best_score:
                best_score = score
                best_alias = alias
                best_candidate = candidate
    return best_score, best_alias, best_candidate


def _load_columns(sheet_id: int) -> List[Dict[str, Any]]:
    columns: List[Dict[str, Any]] = []
    for row in get_columns_by_sheet(sheet_id):
        column = dict(row)
        column["column_header"] = _parse_header(
            column.get("column_header"),
            column.get("column_name_flat") or "",
        )
        columns.append(column)
    return columns


def _field_aliases(target_config: Dict[str, Any], output_field: str) -> List[str]:
    return get_field_aliases(target_config["sheet_group"], output_field)


def _map_catalog_columns(
    sheet: Dict[str, Any],
    target_name: str,
    target_config: Dict[str, Any],
) -> Dict[str, Any]:
    columns = _load_columns(sheet["sheet_id"])
    fields = target_config["fields"]
    selected: Dict[str, Optional[int]] = {field: None for field in fields}
    evidence: Dict[str, Dict[str, Any]] = {}
    used_column_ids: set[int] = set()

    for output_field in fields:
        aliases = _field_aliases(target_config, output_field)
        for alias in aliases:
            exact_match = next(
                (
                    column
                    for column in columns
                    if column.get("column_id") not in used_column_ids
                    and header_matches_alias(
                        column.get("column_header") or [],
                        column.get("column_name_flat") or "",
                        [alias],
                    )
                ),
                None,
            )
            if exact_match:
                column_id = int(exact_match["column_id"])
                selected[output_field] = column_id
                used_column_ids.add(column_id)
                evidence[output_field] = {
                    "column_id": column_id,
                    "column_name": exact_match.get("column_name_flat"),
                    "matched_alias": alias,
                    "confidence": 1.0,
                    "method": "exact",
                }
                break
        if selected[output_field]:
            continue

        scored: List[Tuple[float, int, Dict[str, Any], Optional[str], Optional[str]]] = []
        for column in columns:
            column_id = int(column["column_id"])
            if column_id in used_column_ids:
                continue
            score, alias, candidate = _best_alias_score(column, aliases)
            scored.append((score, int(column.get("column_index") or 0), column, alias, candidate))
        scored.sort(key=lambda item: (-item[0], item[1]))
        if scored and scored[0][0] >= FUZZY_MATCH_THRESHOLD:
            score, _, column, alias, candidate = scored[0]
            column_id = int(column["column_id"])
            selected[output_field] = column_id
            used_column_ids.add(column_id)
            evidence[output_field] = {
                "column_id": column_id,
                "column_name": column.get("column_name_flat"),
                "matched_header_candidate": candidate,
                "matched_alias": alias,
                "confidence": round(score, 3),
                "method": "fuzzy",
            }

    for output_field, field_evidence in evidence.items():
        if field_evidence["method"] == "exact":
            continue
        alias = field_evidence.get("matched_header_candidate") or field_evidence.get("column_name")
        if alias:
            add_field_aliases(
                target_config["sheet_group"],
                output_field,
                [alias],
            )

    return {
        "sheet_id": sheet["sheet_id"],
        "sheet_name": sheet["sheet_name"],
        "field_column_ids": selected,
        "evidence": evidence,
    }


def _load_row_values(sheet_id: int) -> Dict[int, Dict[int, Any]]:
    conn = get_db_connection()
    try:
        rows = conn.execute(
            """
            SELECT row_num, column_id, value
            FROM data
            WHERE sheet_id = ?
            ORDER BY row_num, id
            """,
            (sheet_id,),
        ).fetchall()
    finally:
        conn.close()

    values_by_row: Dict[int, Dict[int, Any]] = {}
    for row in rows:
        values_by_row.setdefault(int(row["row_num"]), {})[row["column_id"]] = row["value"]
    return values_by_row


def _clean_value(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _records_from_mapping(
    file_id: int,
    fields: List[str],
    mapping: Dict[str, Any],
) -> List[Dict[str, Any]]:
    column_ids = mapping["field_column_ids"]
    records: List[Dict[str, Any]] = []
    for row_num, row_values in _load_row_values(mapping["sheet_id"]).items():
        selected_values = {
            field: _clean_value(row_values.get(column_ids.get(field)))
            for field in fields
        }
        if not any(value is not None for value in selected_values.values()):
            continue
        records.append(
            {
                "file_id": file_id,
                "sheet_id": mapping["sheet_id"],
                "sheet_name": mapping["sheet_name"],
                "row_num": row_num,
                **selected_values,
            }
        )
    return records


def _candidate_sheets(
    file_id: int,
    sheet_group_analysis: Dict[str, Any],
    sheet_group: str,
) -> List[Dict[str, Any]]:
    classified_sheet_ids = [
        int(row["sheet_id"])
        for row in sheet_group_analysis.get("classifications") or []
        if row.get("group") == sheet_group and row.get("sheet_id")
    ]
    conn = get_db_connection()
    try:
        rows = conn.execute(
            """
            SELECT sheet_id, sheet_name
            FROM file_sheet_headers
            WHERE file_id = ? AND IFNULL(skipped, 0) = 0
            ORDER BY sheet_name, sheet_id
            """,
            (file_id,),
        ).fetchall()
    finally:
        conn.close()
    return [dict(row) for row in rows if int(row["sheet_id"]) in classified_sheet_ids]


def extract_table_catalogs(
    file_id: int,
    sheet_group_analysis: Dict[str, Any],
) -> Dict[str, Any]:
    """Rebuild both configured table catalogs while preserving equal input rows."""
    from services.embeddings import embed_descriptions

    target_reports: Dict[str, Dict[str, Any]] = {}
    records_by_target: Dict[str, List[Dict[str, Any]]] = {}
    target_configs: Dict[str, Dict[str, Any]] = {}

    for target_name in TABLE_CATALOG_TARGETS:
        target_config = get_usefull_col_extraction_target(target_name)
        target_configs[target_name] = target_config
        sheets = _candidate_sheets(
            file_id,
            sheet_group_analysis,
            target_config["sheet_group"],
        )
        mappings = [
            _map_catalog_columns(sheet, target_name, target_config)
            for sheet in sheets
        ]
        records = [
            record
            for mapping in mappings
            for record in _records_from_mapping(
                file_id,
                target_config["fields"],
                mapping,
            )
        ]
        embeddings = embed_descriptions([row["description"] for row in records])
        for row, embedding in zip(records, embeddings):
            row["description_embedding"] = embedding
        records_by_target[target_name] = records
        target_reports[target_name] = {
            "sheet_group": target_config["sheet_group"],
            "sheet_count": len(sheets),
            "count": len(records),
            "sheet_mappings": mappings,
        }

    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("BEGIN")
        for target_name in TABLE_CATALOG_TARGETS:
            fields = target_configs[target_name]["fields"]
            insert_columns = (
                "file_id",
                "sheet_id",
                "sheet_name",
                "row_num",
                *fields,
                "description_embedding",
            )
            columns_sql = ", ".join(
                _sql_identifier(column) for column in insert_columns
            )
            placeholders = ", ".join("?" for _ in insert_columns)
            cursor.execute(f"DELETE FROM {target_name} WHERE file_id = ?", (file_id,))
            cursor.executemany(
                f"""
                INSERT INTO {target_name}
                ({columns_sql})
                VALUES ({placeholders})
                """,
                [
                    (
                        row["file_id"],
                        row["sheet_id"],
                        row["sheet_name"],
                        row["row_num"],
                        *(row.get(field) for field in fields),
                        row["description_embedding"],
                    )
                    for row in records_by_target[target_name]
                ],
            )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

    return {
        "status": "ok",
        "file_id": file_id,
        "count": sum(report["count"] for report in target_reports.values()),
        "targets": target_reports,
    }


__all__ = ["TABLE_CATALOG_TARGETS", "extract_table_catalogs"]
