"""Shared helpers for agent tools."""

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def clamped_int(value: Any, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, min(parsed, maximum))


def pack_tabular_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    columns: Sequence[str],
    dictionary_columns: Sequence[str] = (),
) -> Dict[str, Any]:
    """Losslessly pack mapping rows for a compact model-visible payload.

    Every input row remains one positional output row in the same order.
    Repeated values in explicitly selected columns are interned by exact value
    and type, never by similarity. Dictionary indexes are transport references,
    not domain identifiers or logical grouping keys.
    """
    clean_columns = [str(column or "").strip() for column in columns]
    if not clean_columns or any(not column for column in clean_columns):
        raise ValueError("columns must contain non-empty names")
    if len(set(clean_columns)) != len(clean_columns):
        raise ValueError("columns must be unique")

    dictionary_names = [
        str(column or "").strip() for column in dictionary_columns
    ]
    unknown = [
        column for column in dictionary_names if column not in clean_columns
    ]
    if unknown:
        raise ValueError(
            "dictionary columns must be declared: " + ", ".join(unknown)
        )
    dictionary_names = list(dict.fromkeys(dictionary_names))
    dictionaries: Dict[str, List[Any]] = {
        column: [] for column in dictionary_names
    }

    def exact_index(values: List[Any], value: Any) -> int:
        for index, current in enumerate(values):
            if type(current) is type(value) and current == value:
                return index
        values.append(value)
        return len(values) - 1

    packed_rows: List[List[Any]] = []
    for row in rows:
        packed_row: List[Any] = []
        for column in clean_columns:
            value = row.get(column)
            if column in dictionaries:
                value = exact_index(dictionaries[column], value)
            packed_row.append(value)
        packed_rows.append(packed_row)
    return {
        "row_format": "arrays_in_column_order",
        "columns": clean_columns,
        "dictionaries": dictionaries,
        "rows": packed_rows,
    }


def normalize_column_reference(
    table_name: str,
    column_name: Optional[str],
) -> Optional[str]:
    """Return a bare column name for an exact table.column reference."""
    if column_name is None:
        return None
    clean_column = str(column_name).strip()
    if not clean_column:
        return None

    prefix = f"{str(table_name).strip()}."
    if clean_column.casefold().startswith(prefix.casefold()):
        return clean_column[len(prefix):].strip()
    return clean_column


def file_meta(file_id: int) -> dict[str, Any]:
    """Return minimal metadata for one uploaded file."""
    from storage.database import get_db_connection

    conn = get_db_connection()
    try:
        row = conn.execute(
            "SELECT file_id, filename, upload_time FROM files WHERE file_id = ?",
            (file_id,),
        ).fetchone()
        return dict(row) if row else {"file_id": file_id}
    finally:
        conn.close()
