import html
import json
import os
from pathlib import Path

import pytest

import storage.database as db_storage
from storage.database import get_db_connection


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LIVE_DB_PATH = Path(
    os.getenv("LIVE_AGENT_DB_PATH", "").strip()
    or PROJECT_ROOT / "excel_data.db"
).expanduser()


@pytest.mark.integration
def test_live_s2t_graph_builds_html_and_json_from_working_database(
    tmp_path,
    monkeypatch,
):
    """Build the interactive graph from the real workspace database."""
    if not LIVE_DB_PATH.is_file():
        pytest.skip("workspace excel_data.db is absent")

    import agents.tools.s2t_graph as graph_module
    from agents.tools import visualize_s2t_table_graph

    monkeypatch.setattr(db_storage, "DB_PATH", str(LIVE_DB_PATH))
    monkeypatch.setattr(graph_module, "S2T_TABLE_GRAPH_EXPORT_DIR", tmp_path)

    conn = get_db_connection()
    try:
        has_s2t_schema = conn.execute(
            """
            SELECT 1
            FROM sqlite_master
            WHERE type = 'table' AND name = 's2t_transformations'
            """
        ).fetchone()
        if has_s2t_schema is None:
            pytest.skip("live SQLite database has no s2t_transformations table")
        live_row_count = int(
            conn.execute("SELECT COUNT(*) FROM s2t_transformations").fetchone()[0]
        )
    finally:
        conn.close()
    if live_row_count == 0:
        pytest.skip("workspace s2t_transformations is empty")

    result = visualize_s2t_table_graph.invoke({})

    assert "error" not in result
    assert result["scope"] == "global"
    assert result["rows_analyzed"] == live_row_count
    assert result["node_count"] > 0
    assert result["edge_count"] > 0

    html_name = result["visualization_url"].rsplit("/", 1)[-1]
    json_name = result["data_url"].rsplit("/", 1)[-1]
    html_path = tmp_path / html_name
    json_path = tmp_path / json_name
    assert html_path.is_file()
    assert json_path.is_file()

    html_text = html_path.read_text(encoding="utf-8")
    graph_data = json.loads(json_path.read_text(encoding="utf-8"))
    assert "vis-network" in html_text
    assert len(graph_data["edges"]) == result["edge_count"]

    first_edge = graph_data["edges"][0]
    assert first_edge["source_table"]
    assert first_edge["target_table"]
    assert first_edge["mapping_count"] > 0
    assert html.escape(first_edge["source_table"]) in html_text
    assert html.escape(first_edge["target_table"]) in html_text
