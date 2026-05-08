import tempfile
import pytest

import db_storage
from db_storage import init_db, get_db_connection


@pytest.fixture(autouse=True)
def _temp_db_path():
    original = db_storage.DB_PATH
    with tempfile.NamedTemporaryFile(suffix=".db") as tmp:
        db_storage.DB_PATH = tmp.name
        init_db()
        yield
        db_storage.DB_PATH = original


def test_run_sql_select():
    from load_skills_tools import run_sql

    conn = get_db_connection()
    conn.execute(
        "INSERT INTO files (file_hash, filename, upload_time) VALUES (?, ?, ?)",
        ("h1", "f.xlsx", "2024-01-01"),
    )
    conn.commit()
    conn.close()

    rows = run_sql("SELECT file_hash, filename FROM files WHERE file_hash = 'h1'")
    assert rows == [{"file_hash": "h1", "filename": "f.xlsx"}]


def test_run_sql_invalid_returns_error_dict():
    from load_skills_tools import run_sql

    out = run_sql("NOT A VALID STMT")
    assert isinstance(out, dict)
    assert "error" in out


def test_list_files_empty():
    from load_skills_tools import list_files

    assert list_files() == []


def test_list_sheets_and_columns_after_store(sample_excel_bytes):
    from db_storage import store_excel_data, get_sheet_hash
    from load_skills_tools import list_sheets, list_columns

    sheets_info = [
        {
            "sheet_name": "Sheet1",
            "skipped": False,
            "ai_decision": {
                "header_start_row": 0,
                "header_rows_count": 1,
                "nested_structure": False,
            },
            "columns": ["ColA"],
        }
    ]
    fh = store_excel_data(
        sample_excel_bytes, "x.xlsx", "m", sheets_info, {"Sheet1": [[1]]}
    )
    assert list_sheets(fh) == ["Sheet1"]
    sh = get_sheet_hash(fh, "Sheet1")
    cols = list_columns(sh)
    assert len(cols) == 1
    assert cols[0]["name"] == "ColA"


def test_list_columns_by_sheet_name(sample_excel_bytes):
    from db_storage import store_excel_data
    from load_skills_tools import list_columns

    sheets_info = [
        {
            "sheet_name": "OnlyOne",
            "skipped": False,
            "ai_decision": {
                "header_start_row": 0,
                "header_rows_count": 1,
                "nested_structure": False,
            },
            "columns": ["Z"],
        }
    ]
    store_excel_data(
        sample_excel_bytes, "y.xlsx", "m", sheets_info, {"OnlyOne": [["v"]]}
    )
    resolved = list_columns("OnlyOne")
    assert len(resolved) == 1
    assert resolved[0]["name"] == "Z"
