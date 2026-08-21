"""Build source/target column metadata from catalog sheets and raw S2T rows."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from config.column_mapping import get_field_aliases, normalize_column_alias
from config.useful_columns import get_usefull_col_extraction_target
from sheet_skills.column_matching import (
    FUZZY_MATCH_THRESHOLD,
    best_alias_match,
    candidate_sheets,
    clean_value,
    load_columns,
    load_row_values,
    match_fields,
)
from storage.database import _sql_identifier, get_db_connection


COLUMN_CATALOG_TARGETS = ("source_columns", "target_columns")
COLUMN_VALUE_FIELDS = (
    "data_type",
    "primary_key",
    "not_null",
    "description",
)
S2T_METADATA_ROLES = {
    "source_columns": {
        "role": "source",
        "table_field": "source_table",
        "column_field": "source_field",
        "metadata_fields": {
            "data_type": "source_field_data_type",
            "primary_key": "source_primary_key",
            "not_null": "source_not_null",
            "description": "source_description",
        },
    },
    "target_columns": {
        "role": "target",
        "table_field": "target_table",
        "column_field": "target_field",
        "metadata_fields": {
            "data_type": "target_field_data_type",
            "primary_key": "target_primary_key",
            "not_null": "target_not_null",
            "description": "target_description",
        },
    },
}


def _identity_part(value: Any) -> str:
    return " ".join(str(value or "").strip().split()).casefold()


def _identity_key(table_name: Any, column_name: Any) -> Tuple[str, str]:
    return _identity_part(table_name), _identity_part(column_name)


def _is_null_question(header: Any) -> bool:
    normalized = normalize_column_alias(str(header or ""))
    return (
        "null" in normalized
        and "not null" not in normalized
        and "notnull" not in normalized
    )


def _normalise_flag(value: Any, header: Any = None) -> Tuple[Optional[int], bool]:
    text = clean_value(value)
    if text is None:
        return None, False
    normalized = normalize_column_alias(text)
    if normalized in {"not null", "notnull", "non null", "nonnull", "required"}:
        return 1, False
    if normalized in {"null", "nullable"}:
        return 0, False

    positive = {"1", "true", "yes", "y", "да", "+", "x", "pk"}
    negative = {"0", "false", "no", "n", "нет", "-"}
    if normalized in positive:
        result = 1
    elif normalized in negative:
        result = 0
    else:
        return None, True
    if _is_null_question(header):
        result = 1 - result
    return result, False


def _virtual_header_columns(
    columns: Sequence[Dict[str, Any]],
    row_values: Dict[int, Any],
) -> List[Dict[str, Any]]:
    recovered = []
    for column in columns:
        value = clean_value(row_values.get(int(column["column_id"])))
        if value is None:
            recovered.append(dict(column))
            continue
        recovered.append(
            {
                **column,
                "column_name_flat": value,
                "column_header": [value],
                "leaf_header": value,
            }
        )
    return recovered


def _description_column_ids(
    columns: Sequence[Dict[str, Any]],
    sheet_group: str,
    selected_id: Optional[int],
) -> List[int]:
    aliases = get_field_aliases(sheet_group, "description")
    matches = []
    for column in columns:
        score, _, _, _ = best_alias_match(column, aliases)
        if score >= FUZZY_MATCH_THRESHOLD:
            matches.append(
                (int(column.get("column_index") or 0), int(column["column_id"]))
            )
    ordered = [column_id for _, column_id in sorted(matches)]
    if selected_id in ordered:
        ordered.remove(int(selected_id))
        ordered.insert(0, int(selected_id))
    return ordered


def _specialized_sheet_rows(
    file_id: int,
    sheet_group_analysis: Dict[str, Any],
    target_name: str,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    config = get_usefull_col_extraction_target(target_name)
    fields = tuple(config["fields"])
    records: List[Dict[str, Any]] = []
    mappings: List[Dict[str, Any]] = []
    incomplete_rows: List[Dict[str, Any]] = []
    invalid_flags: List[Dict[str, Any]] = []

    for sheet in candidate_sheets(
        file_id, sheet_group_analysis, config["sheet_group"]
    ):
        sheet_name = sheet["sheet_name"]
        columns = load_columns(file_id, sheet_name)
        rows = load_row_values(file_id, sheet_name)
        mapping = match_fields(
            {**sheet, "columns": columns}, config["sheet_group"], fields
        )
        recovered_header_row: Optional[int] = None
        if not (
            mapping["field_column_ids"].get("table_name")
            and mapping["field_column_ids"].get("column_name")
        ) and rows:
            first_row_num = min(rows)
            recovered_columns = _virtual_header_columns(columns, rows[first_row_num])
            recovered = match_fields(
                {**sheet, "columns": recovered_columns},
                config["sheet_group"],
                fields,
            )
            if (
                recovered["field_column_ids"].get("table_name")
                and recovered["field_column_ids"].get("column_name")
            ):
                columns = recovered_columns
                mapping = recovered
                recovered_header_row = first_row_num

        mapping["header_recovered"] = recovered_header_row is not None
        mapping["header_row_num"] = recovered_header_row
        mappings.append(mapping)
        selected = mapping["field_column_ids"]
        description_ids = _description_column_ids(
            columns, config["sheet_group"], selected.get("description")
        )
        last_table_name: Optional[str] = None
        not_null_header = (mapping.get("evidence", {}).get("not_null") or {}).get(
            "matched_header_candidate"
        )
        for row_num, row_values in rows.items():
            if row_num == recovered_header_row:
                continue
            table_name = clean_value(row_values.get(selected.get("table_name")))
            column_name = clean_value(row_values.get(selected.get("column_name")))
            if table_name:
                last_table_name = table_name
            elif column_name:
                table_name = last_table_name

            raw_primary_key = row_values.get(selected.get("primary_key"))
            raw_not_null = row_values.get(selected.get("not_null"))
            primary_key, invalid_primary_key = _normalise_flag(raw_primary_key)
            not_null, invalid_not_null = _normalise_flag(
                raw_not_null, not_null_header
            )
            if selected.get("primary_key") and clean_value(raw_primary_key) is None:
                primary_key = 0
            if selected.get("not_null") and clean_value(raw_not_null) is None:
                not_null = 0
            description = next(
                (
                    value
                    for value in (
                        clean_value(row_values.get(column_id))
                        for column_id in description_ids
                    )
                    if value is not None
                ),
                None,
            )
            values = {
                "data_type": clean_value(row_values.get(selected.get("data_type"))),
                "primary_key": primary_key,
                "not_null": not_null,
                "description": description,
            }
            if not table_name and not column_name and not any(
                value is not None for value in values.values()
            ):
                continue
            if not table_name or not column_name:
                incomplete_rows.append(
                    {
                        "sheet_name": sheet_name,
                        "row_num": row_num,
                        "table_name": table_name,
                        "column_name": column_name,
                    }
                )
                continue
            if invalid_primary_key or invalid_not_null:
                invalid_flags.append(
                    {
                        "sheet_name": sheet_name,
                        "row_num": row_num,
                        "primary_key": clean_value(raw_primary_key)
                        if invalid_primary_key
                        else None,
                        "not_null": clean_value(raw_not_null)
                        if invalid_not_null
                        else None,
                    }
                )
            records.append(
                {
                    "file_id": file_id,
                    "sheet_name": sheet_name,
                    "row_num": row_num,
                    "table_name": table_name,
                    "column_name": column_name,
                    **values,
                }
            )

    return records, {
        "sheet_group": config["sheet_group"],
        "sheet_count": len(mappings),
        "sheet_mappings": mappings,
        "source_row_count": len(records),
        "incomplete_rows": incomplete_rows,
        "invalid_flags": invalid_flags,
    }


def _column_indices(columns: Sequence[Dict[str, Any]]) -> Dict[int, int]:
    return {
        int(column["column_id"]): int(column.get("column_index") or 0)
        for column in columns
    }


def _anchor_indices(
    core_mapping: Dict[str, Any],
    indices: Dict[int, int],
    role: str,
) -> List[int]:
    selected = core_mapping.get("field_column_ids") or {}
    names = (
        ("source_table", "source_field")
        if role == "source"
        else ("target_table", "target_field")
    )
    return [
        indices[int(selected[name])]
        for name in names
        if selected.get(name) and int(selected[name]) in indices
    ]


def _distance(index: int, anchors: Sequence[int]) -> int:
    return min((abs(index - anchor) for anchor in anchors), default=10**6)


def _s2t_metadata_mapping(
    columns: Sequence[Dict[str, Any]],
    core_mapping: Dict[str, Any],
) -> Dict[str, Dict[str, Optional[int]]]:
    indices = _column_indices(columns)
    result = {
        "source_columns": {field: None for field in COLUMN_VALUE_FIELDS},
        "target_columns": {field: None for field in COLUMN_VALUE_FIELDS},
    }
    used: set[int] = set()
    block_starts = {
        role: min(_anchor_indices(core_mapping, indices, role), default=None)
        for role in ("source", "target")
    }

    def side_for_index(index: int) -> Optional[str]:
        ordered = sorted(
            (start, role)
            for role, start in block_starts.items()
            if start is not None
        )
        if not ordered:
            return None
        preceding = [item for item in ordered if item[0] <= index]
        return (preceding[-1] if preceding else ordered[0])[1]

    for target_name in COLUMN_CATALOG_TARGETS:
        spec = S2T_METADATA_ROLES[target_name]
        role = spec["role"]
        own_anchors = _anchor_indices(core_mapping, indices, role)
        if not own_anchors:
            continue
        other_target = (
            "target_columns" if target_name == "source_columns" else "source_columns"
        )
        for value_field, alias_field in spec["metadata_fields"].items():
            aliases = get_field_aliases("s2t", alias_field)
            other_alias_field = S2T_METADATA_ROLES[other_target]["metadata_fields"][
                value_field
            ]
            shared_aliases = {
                normalize_column_alias(alias)
                for alias in aliases
            } & {
                normalize_column_alias(alias)
                for alias in get_field_aliases("s2t", other_alias_field)
            }
            candidates = []
            for column in columns:
                column_id = int(column["column_id"])
                if column_id in used:
                    continue
                score, alias, _, _ = best_alias_match(column, aliases)
                if score < FUZZY_MATCH_THRESHOLD:
                    continue
                index = indices[column_id]
                own_distance = _distance(index, own_anchors)
                shared = normalize_column_alias(alias or "") in shared_aliases
                if shared and side_for_index(index) not in {None, role}:
                    continue
                candidates.append((score, -own_distance, -index, column_id))
            if not candidates:
                continue
            column_id = max(candidates) [-1]
            result[target_name][value_field] = column_id
            used.add(column_id)
    return result


def _fallback_core_mappings(
    file_id: int,
    sheet_group_analysis: Dict[str, Any],
) -> List[Dict[str, Any]]:
    fields = get_usefull_col_extraction_target("s2t_transformations")["fields"]
    return [
        match_fields(
            {
                **sheet,
                "columns": load_columns(file_id, sheet["sheet_name"]),
            },
            "s2t",
            fields,
        )
        for sheet in candidate_sheets(file_id, sheet_group_analysis, "s2t")
    ]


def _raw_s2t_records(
    file_id: int,
    sheet_group_analysis: Dict[str, Any],
    core_mappings: Optional[Sequence[Dict[str, Any]]],
) -> Dict[str, List[Dict[str, Any]]]:
    mappings = list(core_mappings or _fallback_core_mappings(
        file_id, sheet_group_analysis
    ))
    mapping_by_sheet = {
        str(mapping.get("sheet_name") or "").casefold(): mapping
        for mapping in mappings
    }
    raw_by_sheet = {
        sheet_name: load_row_values(file_id, mapping["sheet_name"])
        for sheet_name, mapping in mapping_by_sheet.items()
    }
    metadata_by_sheet = {}
    headers_by_sheet = {}
    for sheet_name, mapping in mapping_by_sheet.items():
        columns = load_columns(file_id, mapping["sheet_name"])
        metadata_by_sheet[sheet_name] = _s2t_metadata_mapping(columns, mapping)
        headers_by_sheet[sheet_name] = {
            int(column["column_id"]): column.get("leaf_header")
            for column in columns
        }

    conn = get_db_connection()
    try:
        transformations = conn.execute(
            """
            SELECT sheet_name, row_num, source_table, source_field,
                   target_table, target_field
            FROM s2t_transformations
            WHERE file_id = ?
            ORDER BY id
            """,
            (int(file_id),),
        ).fetchall()
    finally:
        conn.close()

    result = {target_name: [] for target_name in COLUMN_CATALOG_TARGETS}
    for stored in transformations:
        sheet_key = str(stored["sheet_name"] or "").casefold()
        if sheet_key not in mapping_by_sheet:
            continue
        row_num = int(stored["row_num"])
        raw_values = raw_by_sheet[sheet_key].get(row_num, {})
        for target_name in COLUMN_CATALOG_TARGETS:
            spec = S2T_METADATA_ROLES[target_name]
            table_name = clean_value(stored[spec["table_field"]])
            column_name = clean_value(stored[spec["column_field"]])
            if not table_name or not column_name:
                continue
            selected = metadata_by_sheet[sheet_key][target_name]
            raw_primary_key = raw_values.get(selected.get("primary_key"))
            raw_not_null = raw_values.get(selected.get("not_null"))
            primary_key, _ = _normalise_flag(raw_primary_key)
            not_null_id = selected.get("not_null")
            not_null, _ = _normalise_flag(
                raw_not_null,
                headers_by_sheet[sheet_key].get(not_null_id) if not_null_id else None,
            )
            if selected.get("primary_key") and clean_value(raw_primary_key) is None:
                primary_key = 0
            if not_null_id and clean_value(raw_not_null) is None:
                not_null = 0
            result[target_name].append(
                {
                    "file_id": file_id,
                    "sheet_name": stored["sheet_name"],
                    "row_num": row_num,
                    "table_name": table_name,
                    "column_name": column_name,
                    "data_type": clean_value(raw_values.get(selected.get("data_type"))),
                    "primary_key": primary_key,
                    "not_null": not_null,
                    "description": clean_value(
                        raw_values.get(selected.get("description"))
                    ),
                }
            )
    return result


def _merge_catalog(
    target_name: str,
    s2t_records: Sequence[Dict[str, Any]],
    sheet_records: Sequence[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    catalog: Dict[Tuple[str, str], Dict[str, Any]] = {}
    conflicts: List[Dict[str, Any]] = []
    s2t_tables: Dict[str, str] = {}

    def add_s2t(record: Dict[str, Any]) -> None:
        key = _identity_key(record["table_name"], record["column_name"])
        s2t_tables.setdefault(key[0], record["table_name"])
        current = catalog.get(key)
        if current is None:
            current = {
                **record,
                "_s2t_fields": set(),
                "_sheet_fields": set(),
            }
            catalog[key] = current
        for field in COLUMN_VALUE_FIELDS:
            value = record.get(field)
            if value is None:
                continue
            if current.get(field) is None:
                current[field] = value
            elif current[field] != value:
                conflicts.append(
                    {
                        "table_name": current["table_name"],
                        "column_name": current["column_name"],
                        "field": field,
                        "selected_source": "first_s2t_row",
                        "selected_value": current[field],
                        "other_value": value,
                    }
                )
            current["_s2t_fields"].add(field)

    for record in s2t_records:
        add_s2t(record)

    unlinked_rows: List[Dict[str, Any]] = []
    for record in sheet_records:
        key = _identity_key(record["table_name"], record["column_name"])
        current = catalog.get(key)
        if current is None:
            canonical_table_name = s2t_tables.get(key[0])
            if canonical_table_name is None:
                unlinked_rows.append(
                    {
                        "sheet_name": record["sheet_name"],
                        "row_num": record["row_num"],
                        "table_name": record["table_name"],
                        "column_name": record["column_name"],
                    }
                )
            current = {
                **record,
                "table_name": canonical_table_name or record["table_name"],
                "_s2t_fields": set(),
                "_sheet_fields": set(),
            }
            catalog[key] = current
        had_sheet_values = bool(current["_sheet_fields"])
        current["sheet_name"] = record["sheet_name"]
        current["row_num"] = record["row_num"]
        for field in COLUMN_VALUE_FIELDS:
            value = record.get(field)
            if value is None:
                continue
            if current.get(field) is not None and current[field] != value:
                conflicts.append(
                    {
                        "table_name": current["table_name"],
                        "column_name": current["column_name"],
                        "field": field,
                        "selected_source": "columns_sheet",
                        "selected_value": current[field] if had_sheet_values else value,
                        "other_value": value if had_sheet_values else current[field],
                    }
                )
            if not had_sheet_values or current.get(field) is None:
                current[field] = value
            current["_sheet_fields"].add(field)

    rows = []
    source_counts = {"columns_sheet": 0, "s2t": 0, "mixed": 0}
    for current in catalog.values():
        if current["_sheet_fields"] and current["_s2t_fields"]:
            metadata_source = "mixed"
        elif current["_sheet_fields"]:
            metadata_source = "columns_sheet"
        else:
            metadata_source = "s2t"
        source_counts[metadata_source] += 1
        rows.append(
            {
                key: value
                for key, value in current.items()
                if not key.startswith("_")
            }
            | {"metadata_source": metadata_source}
        )
    rows.sort(
        key=lambda row: (
            _identity_part(row["table_name"]),
            _identity_part(row["column_name"]),
        )
    )
    return rows, {
        "target": target_name,
        "count": len(rows),
        "source_counts": source_counts,
        "unlinked_rows": unlinked_rows,
        "conflicts": conflicts,
        "missing": {
            field: sum(row.get(field) is None for row in rows)
            for field in COLUMN_VALUE_FIELDS
        },
    }


def _replace_catalog_rows(
    file_id: int,
    rows_by_target: Dict[str, Sequence[Dict[str, Any]]],
) -> None:
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("BEGIN")
        for target_name in COLUMN_CATALOG_TARGETS:
            cursor.execute(
                f"DELETE FROM {_sql_identifier(target_name)} WHERE file_id = ?",
                (int(file_id),),
            )
            fields = get_usefull_col_extraction_target(target_name)["fields"]
            columns = (
                "file_id",
                "sheet_name",
                "row_num",
                *fields,
                "metadata_source",
            )
            columns_sql = ", ".join(_sql_identifier(column) for column in columns)
            placeholders = ", ".join("?" for _ in columns)
            cursor.executemany(
                f"INSERT INTO {_sql_identifier(target_name)} ({columns_sql}) "
                f"VALUES ({placeholders})",
                [
                    tuple(row.get(column) for column in columns)
                    for row in rows_by_target[target_name]
                ],
            )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def extract_column_catalogs(
    file_id: int,
    sheet_group_analysis: Dict[str, Any],
    *,
    s2t_sheet_mappings: Optional[Sequence[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Merge dedicated column sheets with metadata read from raw S2T rows."""
    raw_s2t = _raw_s2t_records(
        file_id, sheet_group_analysis, s2t_sheet_mappings
    )
    rows_by_target: Dict[str, List[Dict[str, Any]]] = {}
    target_reports: Dict[str, Dict[str, Any]] = {}
    for target_name in COLUMN_CATALOG_TARGETS:
        sheet_rows, sheet_report = _specialized_sheet_rows(
            file_id, sheet_group_analysis, target_name
        )
        rows, merge_report = _merge_catalog(
            target_name, raw_s2t[target_name], sheet_rows
        )
        rows_by_target[target_name] = rows
        target_reports[target_name] = {**sheet_report, **merge_report}
    _replace_catalog_rows(file_id, rows_by_target)
    return {
        "status": "ok",
        "file_id": file_id,
        "count": sum(len(rows) for rows in rows_by_target.values()),
        "targets": target_reports,
    }


__all__ = ["COLUMN_CATALOG_TARGETS", "extract_column_catalogs"]
