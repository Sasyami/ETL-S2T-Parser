import logging
import json
import re
from typing import Dict, Any, List, Optional
from db_storage import (
    get_db_connection, generate_id, add_relationship,
    get_column_id_by_name, get_sheet_hash
)

try:
    from langfuse import observe
    LANGFUSE_AVAILABLE = True
except ImportError:
    LANGFUSE_AVAILABLE = False
    def observe(*args, **kwargs):
        def decorator(func): return func
        return decorator

logger = logging.getLogger(__name__)

# Sheets that only describe target columns (no source system) still map to column_mappings;
# we store them with a synthetic source table for referential consistency.
PLACEHOLDER_SOURCE_TABLE = "(target catalog)"


def _normalize_cell(val: Any) -> Any:
    """Strip strings; treat blank as None (DB often stores text)."""
    if val is None:
        return None
    if isinstance(val, str):
        s = val.strip()
        return s if s else None
    return val


def _norm_col_key(s: str) -> str:
    return " ".join((s or "").split()).lower()


def _mapping_column_description(record: Dict[str, Any]) -> Optional[str]:
    """Business description for a target column (S2T catalog Excel)."""
    priority = (
        "target_column_description",
        "column_description",
        "description",
        "comment",
        "column_comment",
        "назначение",
        "описание целевого поля",
        "описание назначения",
        "описание целевого атрибута",
        "Описание",
        "описание",
        "Комментарий",
        "комментарий",
        "Комментарий к полю назначения",
        "комментарий к полю назначения",
    )
    for key in priority:
        v = record.get(key)
        if v is None:
            continue
        norm = _normalize_cell(v)
        if isinstance(norm, str) and norm.strip():
            return norm.strip()
    return None


def _row_value_for_excel_header(row: Dict[str, Any], header: Optional[Any]) -> Any:
    """
    Map similarity-report Excel column names to row dict keys.
    Handles case/whitespace differences and nested headers (leaf segment match).
    """
    if header is None:
        return None
    if not isinstance(header, str):
        header = str(header)
    h = header.strip()
    if not h:
        return None
    hn = _norm_col_key(h)

    if h in row:
        return _normalize_cell(row[h])

    hl = h.lower()
    for k, v in row.items():
        if k is None:
            continue
        ks = str(k).strip()
        if ks.lower() == hl:
            return _normalize_cell(v)

    for k, v in row.items():
        if k is None:
            continue
        if _norm_col_key(str(k)) == hn:
            return _normalize_cell(v)

    segment_matches: List[Any] = []
    for k, v in row.items():
        if k is None:
            continue
        ks = str(k).strip()
        parts = re.split(r"\s*>\s*", ks)
        for p in parts:
            if _norm_col_key(p) == hn:
                segment_matches.append(v)
                break

    if len(segment_matches) == 1:
        return _normalize_cell(segment_matches[0])
    if len(segment_matches) > 1:
        logger.debug(
            "Ambiguous excel header %r on row keys %s",
            h,
            list(row.keys()),
        )

    return None


def get_data_rows(file_hash: str, sheet_name: str) -> List[Dict[str, Any]]:
    """Retrieve data rows for a sheet as list of dicts (column_name -> value)."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT sheet_hash FROM sheets WHERE file_hash = ? AND sheet_name = ?", (file_hash, sheet_name))
    sheet_row = cursor.fetchone()
    if not sheet_row:
        conn.close()
        return []
    sheet_hash = sheet_row["sheet_hash"]

    cursor.execute("SELECT column_hash, column_name_flat FROM columns WHERE sheet_hash = ? ORDER BY column_index",
                   (sheet_hash,))
    columns = cursor.fetchall()
    col_map = {col["column_hash"]: col["column_name_flat"] for col in columns}

    cursor.execute("SELECT column_hash, row_num, value FROM data WHERE sheet_hash = ? ORDER BY row_num, column_hash",
                   (sheet_hash,))
    rows_data = cursor.fetchall()
    conn.close()

    rows_dict = {}
    for row in rows_data:
        row_num = row["row_num"]
        if row_num not in rows_dict:
            rows_dict[row_num] = {}
        col_name = col_map.get(row["column_hash"], "unknown")
        rows_dict[row_num][col_name] = row["value"]
    return list(rows_dict.values())


def get_or_create_source_table(name: str, description: Optional[str] = None, system_code: Optional[str] = None) -> str:
    if not name:
        raise ValueError("Source table name cannot be empty or None")
    name_lower = name.lower()
    table_id = generate_id(name_lower)
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM source_tables WHERE id = ?", (table_id,))
    if cursor.fetchone():
        conn.close()
        return table_id
    cursor.execute("""
        INSERT INTO source_tables (id, name, description, system_code)
        VALUES (?, ?, ?, ?)
    """, (table_id, name, description, system_code))
    conn.commit()
    conn.close()
    return table_id


def get_or_create_target_table(name: str, description: Optional[str] = None) -> str:
    if not name:
        raise ValueError("Target table name cannot be empty or None")
    name_lower = name.lower()
    table_id = generate_id(name_lower)
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM target_tables WHERE id = ?", (table_id,))
    if cursor.fetchone():
        conn.close()
        return table_id
    cursor.execute("""
        INSERT INTO target_tables (id, name, description)
        VALUES (?, ?, ?)
    """, (table_id, name, description))
    conn.commit()
    conn.close()
    return table_id


def insert_column_mapping(target_table_name: str, target_column: str,
                          source_table_name: str, source_column: Optional[str],
                          transformation_rule: Optional[str], data_type: Optional[str],
                          is_primary_key: bool,
                          column_description: Optional[str] = None) -> str:
    if not target_table_name or not target_column or not source_table_name:
        raise ValueError(f"Missing required fields: target_table_name='{target_table_name}', "
                         f"target_column='{target_column}', source_table_name='{source_table_name}'")
    target_id = get_or_create_target_table(target_table_name)
    source_id = get_or_create_source_table(source_table_name)
    mapping_id = generate_id(target_id, target_column, source_id, source_column)
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT OR IGNORE INTO column_mappings
        (id, target_table_id, target_column, source_table_id, source_column,
         transformation_rule, data_type, is_primary_key, column_description)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (mapping_id, target_id, target_column, source_id, source_column,
          transformation_rule, data_type, 1 if is_primary_key else 0,
          column_description))
    conn.commit()
    conn.close()
    return mapping_id


def insert_addition(table_name: Optional[str], table_description: Optional[str],
                    source_tables_name: Optional[str], sql: Optional[str],
                    description: Optional[str]) -> str:
    add_id = generate_id(table_name, source_tables_name, sql)
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT OR IGNORE INTO additions (id, table_name, table_description, source_tables_name, sql, description)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (add_id, table_name, table_description, source_tables_name, sql, description))
    conn.commit()
    conn.close()
    return add_id


@observe()
def load_data_from_similarity_report(file_hash: str, similarity_report: Dict[str, Any],
                                     include_medium: bool = True,
                                     min_similarity: str = "low") -> Dict[str, int]:
    """
    Load data into target tables based on the similarity report.
    Creates graph relationships for every column mapping.
    """
    similarity_rank = {"high": 3, "medium": 2, "low": 1}
    min_rank = similarity_rank.get(min_similarity, 0)

    inserted_counts = {}

    for suggestion in similarity_report.get("mapping_suggestions", []):
        sim = suggestion.get("similarity", "low")
        sim_rank = similarity_rank.get(sim, 0)

        if sim_rank < min_rank:
            continue
        if sim == "medium" and not include_medium:
            continue

        target_table = suggestion["target_table"]
        sheet_name = suggestion["excel_sheet"]
        column_mapping = suggestion.get("column_mapping", {})

        if not target_table:
            logger.info(f"Skipping sheet '{sheet_name}' - no target table matched")
            continue

        required_cols = {
            "source_tables": ["name"],
            "target_tables": ["name"],
            # source_* optional: target-column catalog sheets often have no source side
            "column_mappings": ["target_table_name", "target_column"],
            "additions": []
        }
        required = required_cols.get(target_table, [])
        missing = [col for col in required if col not in column_mapping]
        if missing:
            logger.warning(f"Skipping sheet '{sheet_name}' -> {target_table}: missing required columns {missing}. Mapping: {column_mapping}")
            continue

        rows = get_data_rows(file_hash, sheet_name)
        if not rows:
            logger.info(f"No data rows for sheet '{sheet_name}'")
            continue

        # Get sheet_hash for lineage
        sheet_hash = get_sheet_hash(file_hash, sheet_name)
        if not sheet_hash:
            logger.error(f"Cannot find sheet_hash for sheet '{sheet_name}' (file {file_hash})")
            continue

        count = 0
        for row_idx, row in enumerate(rows):
            record = {}
            for target_col, excel_col in column_mapping.items():
                record[target_col] = _row_value_for_excel_header(row, excel_col)

            try:
                if target_table == "source_tables":
                    name = record.get("name")
                    if not name:
                        logger.warning(f"Sheet '{sheet_name}', row {row_idx}: skipping - missing 'name'")
                        continue
                    get_or_create_source_table(
                        name=name,
                        description=record.get("description"),
                        system_code=record.get("system_code")
                    )
                    count += 1

                elif target_table == "target_tables":
                    name = record.get("name")
                    if not name:
                        logger.warning(f"Sheet '{sheet_name}', row {row_idx}: skipping - missing 'name'")
                        continue
                    get_or_create_target_table(
                        name=name,
                        description=record.get("description")
                    )
                    count += 1

                elif target_table == "column_mappings":
                    target_table_name_val = _normalize_cell(record.get("target_table_name"))
                    target_column = _normalize_cell(record.get("target_column"))
                    source_table_name_val = _normalize_cell(record.get("source_table_name"))
                    source_column_val = _normalize_cell(record.get("source_column"))
                    if not target_table_name_val or not target_column:
                        empty_fields = [
                            n
                            for n, v in (
                                ("target_table_name", target_table_name_val),
                                ("target_column", target_column),
                            )
                            if not v
                        ]
                        logger.warning(
                            "Sheet '%s', row %s: skipping column_mappings — "
                            "empty fields %s. Mapped excel headers: %s. "
                            "Sample row keys: %s",
                            sheet_name,
                            row_idx,
                            empty_fields,
                            {k: column_mapping.get(k) for k in column_mapping},
                            list(row.keys())[:25],
                        )
                        continue

                    st_name = source_table_name_val or PLACEHOLDER_SOURCE_TABLE
                    mapping_id = insert_column_mapping(
                        target_table_name=target_table_name_val,
                        target_column=target_column,
                        source_table_name=st_name,
                        source_column=source_column_val,
                        transformation_rule=record.get("transformation_rule"),
                        data_type=record.get("data_type"),
                        is_primary_key=str(record.get("is_primary_key")).lower()
                        in ("yes", "true", "1", "y", "да"),
                        column_description=_mapping_column_description(record),
                    )

                    # Lineage: resolve the *Excel column header* from the mapping, not the cell value.
                    src_excel_header = column_mapping.get("source_column")
                    if src_excel_header is not None and str(src_excel_header).strip():
                        hdr = str(src_excel_header).strip()
                        source_col_hash = get_column_id_by_name(sheet_hash, hdr)
                        if source_col_hash:
                            metadata = {"transformation": record.get("transformation_rule", "direct")}
                            add_relationship(source_col_hash, mapping_id, "MAPS_TO", json.dumps(metadata))
                            logger.debug("MAPS_TO: Excel column %r -> mapping %s", hdr, mapping_id)
                        else:
                            logger.debug(
                                "No column hash for excel header %r (sheet %s); lineage skipped",
                                hdr,
                                sheet_name,
                            )
                    count += 1

                elif target_table == "additions":
                    if any(record.values()):
                        insert_addition(
                            table_name=record.get("table_name"),
                            table_description=record.get("table_description"),
                            source_tables_name=record.get("source_tables_name"),
                            sql=record.get("sql"),
                            description=record.get("description")
                        )
                        count += 1
                    else:
                        logger.debug(f"Sheet '{sheet_name}', row {row_idx}: all fields empty, skipping")

            except Exception as e:
                logger.error(f"Sheet '{sheet_name}', row {row_idx}: failed to insert: {e}")

        inserted_counts[target_table] = inserted_counts.get(target_table, 0) + count
        logger.info(f"Loaded {count} rows from sheet '{sheet_name}' into '{target_table}'")

    return inserted_counts