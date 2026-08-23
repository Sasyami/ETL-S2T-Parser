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
    persist_mapping_aliases,
)
from sheet_skills.s2t import resolve_configured_sheet_mapping
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


class ColumnCatalogExtractionError(RuntimeError):
    """Raised when a configured column-catalog sheet cannot be mapped safely."""


def _identity_part(value: Any) -> str:
    return " ".join(str(value or "").strip().split()).casefold()


def _identity_key(table_name: Any, column_name: Any) -> Tuple[str, str]:
    return _identity_part(table_name), _identity_part(column_name)


def _is_null_question(header: Any) -> bool:
    normalized = normalize_column_alias(str(header or ""))
    if not normalized:
        return False
    not_null_headers = {
        "not null",
        "notnull",
        "non null",
        "nonnull",
        "required",
        "mandatory",
        "обязательное поле",
        "обязательность заполнения",
        "null не допускается",
        "не допускается null",
    }
    if normalized in not_null_headers:
        return False
    nullable_headers = {
        "null",
        "nullable",
        "allows null",
        "allow null",
        "null allowed",
        "column null option",
        "допускается null",
        "null допускается",
        "может быть null",
        "допускается пустое значение",
        "может быть пустым",
        "пустое значение допустимо",
    }
    if normalized in nullable_headers:
        return True
    if "null" not in normalized:
        return False
    return not any(
        marker in normalized
        for marker in (
            "not null",
            "notnull",
            "non null",
            "nonnull",
            "null не допускается",
            "не допускается null",
        )
    )


def _normalise_flag(value: Any, header: Any = None) -> Tuple[Optional[int], bool]:
    text = clean_value(value)
    if text is None:
        return None, False
    normalized = normalize_column_alias(text)
    if normalized in {
        "not null",
        "notnull",
        "non null",
        "nonnull",
        "not nullable",
        "required",
        "mandatory",
        "обязательно",
        "обязательный",
        "обязательная",
        "обязательное",
        "не допускается null",
        "null не допускается",
    }:
        return 1, False
    if normalized in {
        "null",
        "nullable",
        "optional",
        "не обязательно",
        "необязательно",
        "необязательный",
        "необязательная",
        "необязательное",
        "допускается null",
        "null допускается",
    }:
        return 0, False

    positive = {"1", "true", "yes", "y", "да", "истина", "+", "x", "pk"}
    negative = {"0", "false", "no", "n", "нет", "ложь", "-"}
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
    if selected_id:
        if int(selected_id) in ordered:
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
    missing_table_rows: List[Dict[str, Any]] = []
    invalid_flags: List[Dict[str, Any]] = []
    attempts = 0

    for sheet in candidate_sheets(
        file_id, sheet_group_analysis, config["sheet_group"]
    ):
        sheet_name = sheet["sheet_name"]
        columns = load_columns(file_id, sheet_name, sample_limit=5)
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
            if recovered["field_column_ids"].get("column_name"):
                columns = recovered_columns
                mapping = recovered
                recovered_header_row = first_row_num

        resolved = resolve_configured_sheet_mapping(
            file_id,
            {**sheet, "columns": columns},
            sheet_group=config["sheet_group"],
            fields=fields,
            required_fields=("column_name",),
        )
        attempts += int(resolved["attempts"])
        if resolved.get("mapping") is None:
            raise ColumnCatalogExtractionError(
                f"{sheet_name}: не удалось сопоставить колонки каталога "
                f"{target_name}: {resolved.get('error') or 'unknown mapping error'}"
            )
        mapping = resolved["mapping"]
        mapping["method"] = resolved["method"]
        mapping["attempts"] = int(resolved["attempts"])
        mapping["header_recovered"] = recovered_header_row is not None
        mapping["header_row_num"] = recovered_header_row
        mappings.append(mapping)
        selected = mapping["field_column_ids"]
        description_ids = _description_column_ids(
            columns, config["sheet_group"], selected.get("description")
        )
        not_null_header = (mapping.get("evidence", {}).get("not_null") or {}).get(
            "matched_header_candidate"
        )
        for row_num, row_values in rows.items():
            if row_num == recovered_header_row:
                continue
            table_name = clean_value(row_values.get(selected.get("table_name")))
            column_name = clean_value(row_values.get(selected.get("column_name")))

            raw_primary_key = row_values.get(selected.get("primary_key"))
            raw_not_null = row_values.get(selected.get("not_null"))
            primary_key, invalid_primary_key = _normalise_flag(raw_primary_key)
            not_null, invalid_not_null = _normalise_flag(
                raw_not_null, not_null_header
            )
            if selected.get("primary_key") and clean_value(raw_primary_key) is None:
                primary_key = 0
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
            if not column_name:
                incomplete_rows.append(
                    {
                        "sheet_name": sheet_name,
                        "row_num": row_num,
                        "table_name": table_name,
                        "column_name": column_name,
                    }
                )
                continue
            if not table_name:
                missing_table_rows.append(
                    {
                        "sheet_name": sheet_name,
                        "row_num": row_num,
                        "column_name": column_name,
                    }
                )
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

    aliases_added = persist_mapping_aliases(config["sheet_group"], mappings)
    return records, {
        "sheet_group": config["sheet_group"],
        "sheet_count": len(mappings),
        "attempts": attempts,
        "aliases_added": aliases_added,
        "sheet_mappings": mappings,
        "source_row_count": len(records),
        "incomplete_rows": incomplete_rows,
        "missing_table_rows": missing_table_rows,
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
    s2t_by_column: Dict[str, Dict[Tuple[str, str], str]] = {}

    def merge_description(
        current: Dict[str, Any],
        value: Any,
        *,
        prefer_new: bool,
        selected_source: str,
    ) -> None:
        description = clean_value(value)
        if description is None:
            return
        selected = clean_value(current.get("description"))
        if selected is None:
            current["description"] = description
            return
        if _identity_part(selected) == _identity_part(description):
            return
        selected_value = description if prefer_new else selected
        other_value = selected if prefer_new else description
        conflicts.append(
            {
                "table_name": current["table_name"],
                "column_name": current["column_name"],
                "field": "description",
                "selected_source": selected_source,
                "selected_value": selected_value,
                "other_value": other_value,
            }
        )
        if prefer_new:
            current["description"] = description

    def add_s2t(record: Dict[str, Any]) -> None:
        key = _identity_key(record["table_name"], record["column_name"])
        s2t_tables.setdefault(key[0], record["table_name"])
        s2t_by_column.setdefault(key[1], {}).setdefault(key, record["table_name"])
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
            if field == "description":
                merge_description(
                    current,
                    value,
                    prefer_new=False,
                    selected_source="first_s2t_row",
                )
            elif current.get(field) is None:
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
    resolved_missing_tables: List[Dict[str, Any]] = []
    unresolved_missing_tables: List[Dict[str, Any]] = []
    ambiguous_missing_tables: List[Dict[str, Any]] = []
    for source_record in sheet_records:
        record = dict(source_record)
        table_name = clean_value(record.get("table_name"))
        column_key = _identity_part(record["column_name"])
        if table_name is None:
            candidates = s2t_by_column.get(column_key, {})
            if len(candidates) == 1:
                _, resolved_table_name = next(iter(candidates.items()))
                record["table_name"] = resolved_table_name
                resolved_missing_tables.append(
                    {
                        "sheet_name": record["sheet_name"],
                        "row_num": record["row_num"],
                        "column_name": record["column_name"],
                        "table_name": resolved_table_name,
                    }
                )
            elif candidates:
                ambiguous_missing_tables.append(
                    {
                        "sheet_name": record["sheet_name"],
                        "row_num": record["row_num"],
                        "column_name": record["column_name"],
                        "candidate_tables": sorted(set(candidates.values())),
                    }
                )
            else:
                unresolved_missing_tables.append(
                    {
                        "sheet_name": record["sheet_name"],
                        "row_num": record["row_num"],
                        "column_name": record["column_name"],
                    }
                )
        if clean_value(record.get("table_name")) is None:
            key = (
                "__missing_table__:"
                + _identity_part(record["sheet_name"])
                + f":{record['row_num']}",
                column_key,
            )
        else:
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
        current["sheet_name"] = record["sheet_name"]
        current["row_num"] = record["row_num"]
        for field in COLUMN_VALUE_FIELDS:
            value = record.get(field)
            if value is None:
                continue
            had_sheet_field = field in current["_sheet_fields"]
            if field == "description":
                merge_description(
                    current,
                    value,
                    prefer_new=not had_sheet_field,
                    selected_source="columns_sheet",
                )
            elif had_sheet_field:
                if current.get(field) != value:
                    conflicts.append(
                        {
                            "table_name": current["table_name"],
                            "column_name": current["column_name"],
                            "field": field,
                            "selected_source": "columns_sheet",
                            "selected_value": current[field],
                            "other_value": value,
                        }
                    )
            else:
                if current.get(field) is not None and current[field] != value:
                    conflicts.append(
                        {
                            "table_name": current["table_name"],
                            "column_name": current["column_name"],
                            "field": field,
                            "selected_source": "columns_sheet",
                            "selected_value": value,
                            "other_value": current[field],
                        }
                    )
                current[field] = value
            current["_sheet_fields"].add(field)

    rows = []
    source_counts = {"columns_sheet": 0, "s2t": 0, "mixed": 0}
    for current in catalog.values():
        if current["_sheet_fields"] and current["_s2t_fields"]:
            source_kind = "mixed"
        elif current["_sheet_fields"]:
            source_kind = "columns_sheet"
        else:
            source_kind = "s2t"
        source_counts[source_kind] += 1
        row = {
            key: value
            for key, value in current.items()
            if not key.startswith("_")
        }
        rows.append(row)
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
        "resolved_missing_tables": resolved_missing_tables,
        "unresolved_missing_tables": unresolved_missing_tables,
        "ambiguous_missing_tables": ambiguous_missing_tables,
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
                "description_embedding",
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


def _embed_catalog_descriptions(
    rows_by_target: Dict[str, Sequence[Dict[str, Any]]],
) -> int:
    """Embed each column's technical name and optional description in one batch."""
    from services.embeddings import embed_descriptions

    pending: List[Tuple[Dict[str, Any], str]] = []
    for target_name in COLUMN_CATALOG_TARGETS:
        for row in rows_by_target[target_name]:
            row["description_embedding"] = None
            semantic_text = _column_semantic_text(row)
            if semantic_text:
                pending.append((row, semantic_text))
    if not pending:
        return 0
    embeddings = embed_descriptions([description for _, description in pending])
    for (row, _), embedding in zip(pending, embeddings):
        row["description_embedding"] = embedding
    return len(pending)


def _column_semantic_text(row: Dict[str, Any]) -> str:
    """Build stable semantic text without source-sheet header aliases."""
    column_name = clean_value(row.get("column_name"))
    description = clean_value(row.get("description"))
    parts = []
    if column_name:
        parts.append(f"Название колонки: {column_name}")
    if description:
        parts.append(f"Описание: {description}")
    return "\n".join(parts)


def backfill_column_description_embeddings(
    file_id: Optional[int] = None,
) -> Dict[str, Any]:
    """Embed stored column names/descriptions missing a semantic embedding."""
    from services.embeddings import embed_descriptions

    conn = get_db_connection()
    try:
        candidates: List[Dict[str, Any]] = []
        for table_name in COLUMN_CATALOG_TARGETS:
            where = (
                "column_name IS NOT NULL AND trim(column_name) <> '' "
                "AND description_embedding IS NULL"
            )
            params: Tuple[Any, ...] = ()
            if file_id is not None:
                where += " AND file_id = ?"
                params = (int(file_id),)
            candidates.extend(
                {
                    "table_name": table_name,
                    "id": int(row["id"]),
                    "description": _column_semantic_text(dict(row)),
                }
                for row in conn.execute(
                    f"SELECT id, column_name, description "
                    f"FROM {_sql_identifier(table_name)} "
                    f"WHERE {where} ORDER BY id",
                    params,
                ).fetchall()
            )
        if not candidates:
            return {"file_id": file_id, "candidates": 0, "updated": 0}

        embeddings = embed_descriptions(
            [candidate["description"] for candidate in candidates]
        )
        cursor = conn.cursor()
        cursor.execute("BEGIN")
        for candidate, embedding in zip(candidates, embeddings):
            cursor.execute(
                f"UPDATE {_sql_identifier(candidate['table_name'])} "
                "SET description_embedding = ? WHERE id = ?",
                (embedding, candidate["id"]),
            )
        conn.commit()
        return {
            "file_id": file_id,
            "candidates": len(candidates),
            "updated": len(candidates),
        }
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
    embedded_count = _embed_catalog_descriptions(rows_by_target)
    _replace_catalog_rows(file_id, rows_by_target)
    return {
        "status": "ok",
        "file_id": file_id,
        "count": sum(len(rows) for rows in rows_by_target.values()),
        "embedded_description_count": embedded_count,
        "targets": target_reports,
    }


__all__ = [
    "COLUMN_CATALOG_TARGETS",
    "ColumnCatalogExtractionError",
    "backfill_column_description_embeddings",
    "extract_column_catalogs",
]
