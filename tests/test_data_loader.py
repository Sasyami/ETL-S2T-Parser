import tempfile
import pytest

import db_storage
from db_storage import init_db, store_excel_data, get_db_connection, get_sheet_hash
from data_loader import (
    get_data_rows,
    get_or_create_source_table,
    get_or_create_target_table,
    load_data_from_similarity_report,
    _row_value_for_excel_header,
)


@pytest.fixture(autouse=True)
def _temp_db_path():
    original = db_storage.DB_PATH
    with tempfile.NamedTemporaryFile(suffix=".db") as tmp:
        db_storage.DB_PATH = tmp.name
        init_db()
        yield
        db_storage.DB_PATH = original


def test_get_data_rows_empty_when_unknown_sheet():
    assert get_data_rows("missing", "Sheet1") == []


def test_row_value_for_excel_header_case_and_nested():
    row = {"Target TABLE": "t1", "S > ColName": "c1"}
    assert _row_value_for_excel_header(row, "target table") == "t1"
    assert _row_value_for_excel_header(row, "ColName") == "c1"


def test_get_data_rows_roundtrip(sample_excel_bytes):
    sheets_info = [
        {
            "sheet_name": "Sheet1",
            "skipped": False,
            "ai_decision": {
                "header_start_row": 0,
                "header_rows_count": 1,
                "nested_structure": False,
            },
            "columns": ["Name", "Age"],
        }
    ]
    data_rows = {"Sheet1": [["Alice", "30"], ["Bob", "25"]]}
    fh = store_excel_data(
        sample_excel_bytes, "t.xlsx", "model", sheets_info, data_rows
    )
    rows = get_data_rows(fh, "Sheet1")
    assert len(rows) >= 2
    names = {r.get("Name") for r in rows}
    assert "Alice" in names and "Bob" in names


def test_get_or_create_source_table_idempotent():
    id1 = get_or_create_source_table("MyTable", "d1", "SYS")
    id2 = get_or_create_source_table("mytable", "d2", "SYS2")
    assert id1 == id2


def test_get_or_create_source_table_empty_name():
    with pytest.raises(ValueError, match="empty"):
        get_or_create_source_table("")


def test_get_or_create_target_table_idempotent():
    id1 = get_or_create_target_table("DimAcc", "desc")
    id2 = get_or_create_target_table("dimacc", "other")
    assert id1 == id2


def test_load_target_catalog_column_mappings_without_source(sample_excel_bytes):
    """Target-only sheets: LLM often omits source_* in column_mapping."""
    sheets_info = [
        {
            "sheet_name": "Target columns",
            "skipped": False,
            "ai_decision": {
                "header_start_row": 0,
                "header_rows_count": 1,
                "nested_structure": False,
            },
            "columns": ["ColumnTableName", "ColumnName", "Column Datatype"],
        }
    ]
    data_rows = {
        "Target columns": [["T1", "c1", "varchar"]],
    }
    fh = store_excel_data(
        sample_excel_bytes, "tc.xlsx", "model", sheets_info, data_rows
    )
    report = {
        "mapping_suggestions": [
            {
                "target_table": "column_mappings",
                "excel_sheet": "Target columns",
                "similarity": "high",
                "column_mapping": {
                    "target_table_name": "ColumnTableName",
                    "target_column": "ColumnName",
                },
            }
        ]
    }
    counts = load_data_from_similarity_report(fh, report)
    assert counts.get("column_mappings", 0) >= 1


def test_lineage_uses_excel_header_not_cell_value(sample_excel_bytes):
    """MAPS_TO must use the physical column from mapping, not the cell text (e.g. dto)."""
    from db_storage import get_lineage

    sheets_info = [
        {
            "sheet_name": "S2T",
            "skipped": False,
            "ai_decision": {
                "header_start_row": 0,
                "header_rows_count": 1,
                "nested_structure": False,
            },
            "columns": ["tgt_tbl", "tgt_col", "src_tbl", "src_col_attr"],
        }
    ]
    data_rows = {
        "S2T": [["dim", "x", "raw", "dto"]],
    }
    fh = store_excel_data(sample_excel_bytes, "s2t.xlsx", "m", sheets_info, data_rows)
    report = {
        "mapping_suggestions": [
            {
                "target_table": "column_mappings",
                "excel_sheet": "S2T",
                "similarity": "high",
                "column_mapping": {
                    "target_table_name": "tgt_tbl",
                    "target_column": "tgt_col",
                    "source_table_name": "src_tbl",
                    "source_column": "src_col_attr",
                },
            }
        ]
    }
    load_data_from_similarity_report(fh, report)

    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT column_hash FROM columns WHERE sheet_hash = ? AND column_name_flat = ?",
        (get_sheet_hash(fh, "S2T"), "src_col_attr"),
    )
    col_row = cur.fetchone()
    conn.close()
    assert col_row
    lineage = get_lineage(col_row["column_hash"])
    assert any(r["relation_type"] == "MAPS_TO" for r in lineage)


def test_insert_column_mapping_and_load_report(sample_excel_bytes):
    sheets_info = [
        {
            "sheet_name": "Map",
            "skipped": False,
            "ai_decision": {
                "header_start_row": 0,
                "header_rows_count": 1,
                "nested_structure": False,
            },
            "columns": [
                "target_table_name",
                "target_column",
                "source_table_name",
                "source_column",
            ],
        }
    ]
    data_rows = {
        "Map": [
            ["t1", "col_a", "src1", "excel_col"],
        ]
    }
    fh = store_excel_data(
        sample_excel_bytes, "m.xlsx", "model", sheets_info, data_rows
    )
    sh = get_sheet_hash(fh, "Map")
    assert sh

    report = {
        "mapping_suggestions": [
            {
                "target_table": "column_mappings",
                "excel_sheet": "Map",
                "similarity": "high",
                "column_mapping": {
                    "target_table_name": "target_table_name",
                    "target_column": "target_column",
                    "source_table_name": "source_table_name",
                    "source_column": "source_column",
                },
            }
        ]
    }
    counts = load_data_from_similarity_report(
        fh, report, include_medium=True, min_similarity="low"
    )
    assert counts.get("column_mappings", 0) >= 1

    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) AS n FROM column_mappings")
    n = cur.fetchone()["n"]
    conn.close()
    assert n >= 1
