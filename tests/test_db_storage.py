import pytest
import json
from db_storage import (
    init_db,
    store_excel_data,
    get_db_connection,
    update_file_summary,
    update_file_result_json,
    generate_id,
    add_relationship,
    get_lineage,
    get_column_id_by_name,
    get_sheet_hash,
)

def test_init_db(temp_db):
    # temp_db fixture provides a connection; tables should exist
    cursor = temp_db.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [row[0] for row in cursor.fetchall()]
    assert 'files' in tables
    assert 'sheets' in tables
    assert 'columns' in tables
    assert 'data' in tables

def test_store_excel_data(temp_db, sample_excel_bytes):
    sheets_info = [{
        "sheet_name": "Sheet1",
        "skipped": False,
        "ai_decision": {"header_start_row": 0, "header_rows_count": 1, "nested_structure": False},
        "columns": ["Name", "Age"]
    }]
    data_rows_by_sheet = {
        "Sheet1": [["Alice", 30], ["Bob", 25]]
    }
    file_hash = store_excel_data(sample_excel_bytes, "test.xlsx", "GigaChat-Pro", sheets_info, data_rows_by_sheet)
    assert file_hash is not None
    # Verify data was inserted
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT filename FROM files WHERE file_hash = ?", (file_hash,))
    row = cursor.fetchone()
    assert row["filename"] == "test.xlsx"
    conn.close()

def test_store_excel_preserves_summary_and_result_json_on_reupload(
    temp_db, sample_excel_bytes
):
    sheets_info = [
        {
            "sheet_name": "Sheet1",
            "skipped": False,
            "ai_decision": {
                "header_start_row": 0,
                "header_rows_count": 1,
                "nested_structure": False,
            },
            "columns": ["A"],
        }
    ]
    data_rows = {"Sheet1": [["1"]]}
    fh = store_excel_data(
        sample_excel_bytes, "t.xlsx", "m1", sheets_info, data_rows
    )
    update_file_summary(fh, "preserved summary")
    update_file_result_json(fh, '{"kept": true}')

    store_excel_data(sample_excel_bytes, "t.xlsx", "m2", sheets_info, data_rows)

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT summary, result_json, model_used FROM files WHERE file_hash = ?",
        (fh,),
    )
    row = cursor.fetchone()
    conn.close()
    assert row["summary"] == "preserved summary"
    assert json.loads(row["result_json"]) == {"kept": True}
    assert row["model_used"] == "m2"


def test_update_file_summary(temp_db):
    # First insert a file manually
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("INSERT INTO files (file_hash, filename, upload_time, model_used) VALUES (?, ?, ?, ?)",
                   ("hash123", "test.xlsx", "2025-01-01", "model"))
    conn.commit()
    conn.close()
    update_file_summary("hash123", "Test summary")
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT summary FROM files WHERE file_hash = ?", ("hash123",))
    row = cursor.fetchone()
    assert row["summary"] == "Test summary"
    conn.close()


def test_generate_id_deterministic():
    assert generate_id("a", "b") == generate_id("a", "b")
    assert generate_id("a", "b") != generate_id("a", "c")


def test_update_file_result_json(temp_db):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO files (file_hash, filename, upload_time, model_used) VALUES (?, ?, ?, ?)",
        ("h_rj", "f.xlsx", "2025-01-01", "m"),
    )
    conn.commit()
    conn.close()
    payload = {"ok": True, "n": 1}
    update_file_result_json("h_rj", json.dumps(payload))
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT result_json FROM files WHERE file_hash = ?", ("h_rj",))
    row = cursor.fetchone()
    conn.close()
    assert json.loads(row["result_json"]) == payload


def test_add_relationship_and_get_lineage(temp_db):
    rid = add_relationship("from-1", "to-2", "MAPS_TO", '{"k": 1}')
    assert rid == generate_id("from-1", "to-2", "MAPS_TO")
    lines = get_lineage("from-1")
    assert len(lines) == 1
    assert lines[0]["to_id"] == "to-2"
    assert lines[0]["relation_type"] == "MAPS_TO"


def test_get_column_id_by_name_and_nested_partial(temp_db):
    sheets_info = [
        {
            "sheet_name": "S1",
            "skipped": False,
            "ai_decision": {
                "header_start_row": 0,
                "header_rows_count": 1,
                "nested_structure": False,
            },
            "columns": ["Parent > Child", "Plain"],
        }
    ]
    fh = store_excel_data(
        b"",
        "minimal.xlsx",
        "model",
        sheets_info,
        {"S1": [["v1", "v2"]]},
        max_rows_per_sheet=10,
    )
    sh = get_sheet_hash(fh, "S1")
    assert get_column_id_by_name(sh, "Parent > Child") is not None
    assert get_column_id_by_name(sh, "zzz_nonexistent_column") is None
    cid = get_column_id_by_name(sh, "Child")
    assert cid is not None


def test_store_nested_column_headers(temp_db, sample_excel_bytes):
    sheets_info = [
        {
            "sheet_name": "Sheet1",
            "skipped": False,
            "ai_decision": {
                "header_start_row": 0,
                "header_rows_count": 2,
                "nested_structure": True,
            },
            "columns": [["H1", "a"], ["H1", "b"]],
        }
    ]
    file_hash = store_excel_data(
        sample_excel_bytes,
        "nest.xlsx",
        "model",
        sheets_info,
        {"Sheet1": [[1, 2]]},
    )
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT column_name_flat FROM columns WHERE sheet_hash = ? ORDER BY column_index",
        (get_sheet_hash(file_hash, "Sheet1"),),
    )
    flats = [r["column_name_flat"] for r in cursor.fetchall()]
    conn.close()
    assert "H1 > a" in flats
    assert "H1 > b" in flats