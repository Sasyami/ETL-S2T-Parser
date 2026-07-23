import tempfile
import pytest

import db_storage
from db_storage import init_db, get_db_connection


def test_load_tools_appends_sqlite_mapping_cheatsheet():
    from load_skills_tools import load_tools

    text = load_tools()
    assert "target_tables" in text
    assert "NOT `target_column_name`" in text or "NOT `table_name`" in text
    assert "`name`" in text


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


def test_mapping_overview_empty_db():
    from load_skills_tools import mapping_overview

    out = mapping_overview(limit=5)
    for t in ("source_tables", "target_tables", "column_mappings", "additions"):
        assert t in out
        assert out[t]["count"] == 0
        assert out[t]["sample"] == []
    assert out["relationships_total"] == 0
    assert out["relationships_by_type"] == []


def test_mapping_overview_with_rows():
    from load_skills_tools import mapping_overview

    conn = get_db_connection()
    conn.execute(
        "INSERT INTO source_tables (id, name) VALUES ('s1', 'src')"
    )
    conn.execute(
        "INSERT INTO target_tables (id, name) VALUES ('t1', 'tgt')"
    )
    conn.execute(
        """INSERT INTO column_mappings
        (id, target_table_id, target_column, source_table_id, source_column)
        VALUES ('m1', 't1', 'c_tgt', 's1', 'c_src')"""
    )
    conn.execute(
        """INSERT INTO additions
        (id, table_name, description) VALUES ('a1', 'new_tbl', 'desc')"""
    )
    conn.execute(
        """INSERT INTO relationships (id, from_id, to_id, relation_type, metadata)
        VALUES ('r1', 'x', 'y', 'semantic', '{}')"""
    )
    conn.commit()
    conn.close()

    out = mapping_overview(limit=10)
    assert out["source_tables"]["count"] == 1
    assert out["target_tables"]["count"] == 1
    assert out["column_mappings"]["count"] == 1
    assert out["additions"]["count"] == 1
    assert out["relationships_total"] == 1
    assert len(out["relationships_by_type"]) >= 1


def test_user_wants_stored_file_inventory():
    from load_skills_tools import user_wants_stored_file_inventory as wants

    assert wants("Give me all files stored")
    assert wants("list every uploaded file in the db")
    assert not wants("What files contain customer identifiers")
    assert not wants("What files map to target columns")


def test_format_stored_files_answer():
    from load_skills_tools import format_stored_files_answer

    conn = get_db_connection()
    conn.execute(
        "INSERT INTO files (file_hash, filename, upload_time) VALUES (?, ?, ?)",
        ("hinv", "inventory.xlsx", "2026-01-02"),
    )
    conn.commit()
    conn.close()

    text = format_stored_files_answer()
    assert "inventory.xlsx" in text
    assert "hinv" in text
    assert "list_files" in text


def test_extract_workbook_filenames_from_query():
    from load_skills_tools import extract_workbook_filenames_from_query as ex

    q = "what sheets in s2t_sbrf_pprb_305000042_dul_b_t_v049.xlsx stored"
    assert ex(q) == ["s2t_sbrf_pprb_305000042_dul_b_t_v049.xlsx"]


def test_user_wants_sheets_for_named_workbook():
    from load_skills_tools import user_wants_sheets_for_named_workbook as sw

    assert sw("what sheets in book.xlsx stored")
    assert sw("tabs in data.xls")
    assert not sw("what columns in book.xlsx")
    assert not sw("what is in book.xlsx")


def test_try_answer_sheets_for_workbook_query():
    from load_skills_tools import try_answer_sheets_for_workbook_query

    conn = get_db_connection()
    conn.execute(
        "INSERT INTO files (file_hash, filename, upload_time) VALUES (?, ?, ?)",
        ("fhsheet", "Book1.xlsx", "2026-01-01"),
    )
    conn.execute(
        """INSERT INTO sheets
        (sheet_hash, file_hash, sheet_name, header_start_row, header_rows_count, nested_structure)
        VALUES (?, ?, ?, 0, 1, 0)""",
        ("sh1", "fhsheet", "SheetA"),
    )
    conn.commit()
    conn.close()

    out = try_answer_sheets_for_workbook_query(
        "What sheets in Book1.xlsx are stored?"
    )
    assert out is not None
    assert "SheetA" in out
    assert "fhsheet" in out


def test_search_column_mappings_matches():
    from load_skills_tools import search_column_mappings

    conn = get_db_connection()
    conn.execute(
        "INSERT INTO target_tables (id, name) VALUES ('t_rate', 'RATE_TGT')"
    )
    conn.execute(
        "INSERT INTO source_tables (id, name) VALUES ('s_src', 'SRC')"
    )
    conn.execute(
        """INSERT INTO column_mappings
        (id, target_table_id, target_column, source_table_id, source_column)
        VALUES ('m1', 't_rate', 'b3050000420018_ratetariff', 's_src', 'oldcol')"""
    )
    conn.commit()
    conn.close()

    rows = search_column_mappings("b3050000420018_ratetariff")
    assert isinstance(rows, list)
    assert len(rows) == 1
    assert rows[0]["target_table_id"] == "t_rate"
    assert rows[0]["target_table_name"] == "RATE_TGT"
    assert rows[0]["target_column"] == "b3050000420018_ratetariff"


def test_try_answer_target_table_for_mapping_identifier():
    from load_skills_tools import try_answer_target_table_for_mapping_identifier

    conn = get_db_connection()
    conn.execute("INSERT INTO target_tables (id, name) VALUES ('t2', 'NM')")
    conn.execute(
        """INSERT INTO column_mappings
        (id, target_table_id, target_column, source_table_id, source_column)
        VALUES ('mx', 't2', 'b3050000420018_ratetariff', '', '')"""
    )
    conn.commit()
    conn.close()

    out = try_answer_target_table_for_mapping_identifier(
        "what target table uses b3050000420018_ratetariff"
    )
    assert out is not None
    assert "t2" in out
    assert "b3050000420018_ratetariff" in out


def test_list_target_table_columns_orphan_mappings():
    """Logical id only in column_mappings (no target_tables row yet)."""
    from load_skills_tools import list_target_table_columns

    conn = get_db_connection()
    conn.execute(
        """INSERT INTO column_mappings
        (id, target_table_id, target_column, source_table_id, source_column)
        VALUES ('mf', 't_agr_frame', 'col_alpha', '', '')"""
    )
    conn.commit()
    conn.close()

    out = list_target_table_columns("t_agr_frame")
    assert out.get("column_count") == 1
    assert out["columns"][0]["target_column"] == "col_alpha"


def test_try_answer_target_table_fields_query_russian():
    from load_skills_tools import try_answer_target_table_fields_query

    conn = get_db_connection()
    conn.execute(
        "INSERT INTO target_tables (id, name) VALUES ('t_agr_frame', 'Agr')"
    )
    conn.execute(
        """INSERT INTO column_mappings
        (id, target_table_id, target_column, source_table_id, source_column)
        VALUES ('m99', 't_agr_frame', 'fld1', '', '')"""
    )
    conn.commit()
    conn.close()

    ans = try_answer_target_table_fields_query("дай поля таблицы t_agr_frame")
    assert ans is not None
    assert "fld1" in ans


def test_search_column_mappings_by_description():
    from load_skills_tools import search_column_mappings

    conn = get_db_connection()
    conn.execute("INSERT INTO target_tables (id, name) VALUES ('txd', 'X')")
    conn.execute(
        """INSERT INTO column_mappings
        (id, target_table_id, target_column, column_description)
        VALUES ('mxd', 'txd', 'c1', 'Unique phrase XyZ')"""
    )
    conn.commit()
    conn.close()

    rows = search_column_mappings("XyZ")
    assert isinstance(rows, list)
    assert len(rows) == 1
    assert rows[0]["column_description"] == "Unique phrase XyZ"


def test_try_answer_target_table_fields_includes_descriptions():
    from load_skills_tools import try_answer_target_table_fields_query

    conn = get_db_connection()
    conn.execute("INSERT INTO target_tables (id, name) VALUES ('td', 'D')")
    conn.execute(
        """INSERT INTO column_mappings
        (id, target_table_id, target_column, column_description)
        VALUES ('md', 'td', 'f1', 'Описание поля')"""
    )
    conn.commit()
    conn.close()

    out = try_answer_target_table_fields_query("описания колонок таблицы td")
    assert out is not None
    assert "Описание поля" in out


def test_aggregate_catalog_columns_merges_duplicate_target_column():
    from load_skills_tools import _aggregate_catalog_columns

    rows = [
        {
            "target_column": "agr_frame_desc",
            "column_description": "",
            "source_column": "a",
            "data_type": "text",
            "is_primary_key": 0,
            "mapping_id": "id1",
            "transformation_rule": None,
        },
        {
            "target_column": "agr_frame_desc",
            "column_description": None,
            "source_column": "b",
            "data_type": "text",
            "is_primary_key": 0,
            "mapping_id": "id2",
            "transformation_rule": "x",
        },
    ]
    agg, merged = _aggregate_catalog_columns(rows)
    assert merged is True
    assert len(agg) == 1
    assert agg[0]["mapping_id"] == "id1, id2"
    assert "`id1`" not in agg[0]["mapping_id"]
    assert "a" in agg[0]["source_column"] and "b" in agg[0]["source_column"]

    from load_skills_tools import _format_mapping_ids_markdown

    assert _format_mapping_ids_markdown(agg[0]["mapping_id"]) == "`id1`, `id2`"


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
