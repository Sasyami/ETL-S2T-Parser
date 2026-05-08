"""Tests for insights_chat (delegates to shared ReAct loop)."""

from unittest.mock import MagicMock, patch


@patch("agents.agent.giga")
def test_insights_chat_final_answer(mock_giga):
    r1 = MagicMock()
    r1.choices[0].message.content = "Final Answer: Mapping looks consistent."
    mock_giga.chat.return_value = r1

    from agents.insights_agent import insights_chat

    out = insights_chat("Summarize mappings", max_steps=3)
    assert "consistent" in out


@patch("agents.insights_agent.react_tool_loop")
def test_insights_chat_scopes_file_hash(mock_loop):
    mock_loop.return_value = "ok"

    from agents.insights_agent import insights_chat

    insights_chat("List sheets", max_steps=2, file_hash="abc123")
    call_kw = mock_loop.call_args
    assert "abc123" in call_kw[0][0]


def test_insights_chat_file_inventory_from_db(tmp_path):
    import db_storage

    orig = db_storage.DB_PATH
    db_storage.DB_PATH = str(tmp_path / "ins.db")
    try:
        from db_storage import init_db, get_db_connection

        init_db()
        conn = get_db_connection()
        conn.execute(
            "INSERT INTO files (file_hash, filename, upload_time) VALUES (?, ?, ?)",
            ("fromdb", "truth.xlsx", "2026-03-01"),
        )
        conn.commit()
        conn.close()

        from agents.insights_agent import insights_chat

        out = insights_chat("Give me all files stored")
        assert "truth.xlsx" in out
        assert "fromdb" in out
        assert "sample_data" not in out.lower()
    finally:
        db_storage.DB_PATH = orig


def test_insights_chat_sheets_for_workbook_from_db(tmp_path):
    import db_storage

    orig = db_storage.DB_PATH
    db_storage.DB_PATH = str(tmp_path / "shts.db")
    try:
        from db_storage import init_db, get_db_connection

        init_db()
        conn = get_db_connection()
        fn = "s2t_sbrf_pprb_305000042_dul_b_t_v049.xlsx"
        conn.execute(
            "INSERT INTO files (file_hash, filename, upload_time) VALUES (?, ?, ?)",
            ("hxl", fn, "2026-05-08"),
        )
        conn.execute(
            """INSERT INTO sheets
            (sheet_hash, file_hash, sheet_name, header_start_row, header_rows_count, nested_structure)
            VALUES (?, ?, ?, 0, 1, 0)""",
            ("shy", "hxl", "DUL_MAIN"),
        )
        conn.commit()
        conn.close()

        from agents.insights_agent import insights_chat

        out = insights_chat(f"what sheets in {fn} stored")
        assert "DUL_MAIN" in out
        assert "hxl" in out
    finally:
        db_storage.DB_PATH = orig


def test_insights_target_table_identifier_short_cut(tmp_path):
    import db_storage

    orig = db_storage.DB_PATH
    db_storage.DB_PATH = str(tmp_path / "tgt.db")
    try:
        from db_storage import init_db, get_db_connection

        init_db()
        conn = get_db_connection()
        conn.execute(
            "INSERT INTO target_tables (id, name) VALUES ('tid1', 'TName')"
        )
        conn.execute(
            """INSERT INTO column_mappings
            (id, target_table_id, target_column, source_table_id, source_column)
            VALUES ('mid', 'tid1', 'b3050000420018_ratetariff', '', '')"""
        )
        conn.commit()
        conn.close()

        from agents.insights_agent import insights_chat

        out = insights_chat("What target table uses b3050000420018_ratetariff?")
        assert "tid1" in out
        assert "TName" in out
    finally:
        db_storage.DB_PATH = orig
