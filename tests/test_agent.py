import pytest
from unittest.mock import patch, MagicMock
from langchain_core.messages import AIMessage
from agents.agent import (
    get_header_decision,
    get_model_name,
    agent_chat,
    _extract_tool_input_dict,
    _extract_final_answer,
    safe_extract_json,
)


@pytest.fixture
def mock_llm_success():
    with patch('agents.agent.call_gigachat_with_retry') as mock_call:
        mock_call.return_value = '{"header_start_row": 0, "header_rows": 1, "nested": false, "explanation": "Test decision"}'
        yield mock_call


def test_get_header_decision_single_row_header(mock_llm_success):
    preview_rows = [
        ["Name", "Age", "City"],
        ["John", 30, "New York"],
        ["Jane", 25, "London"]
    ]
    start_row, header_rows, nested = get_header_decision("Sheet1", preview_rows)
    assert start_row == 0
    assert header_rows == 1
    assert nested is False


def test_get_header_decision_multi_row_header():
    with patch('agents.agent.call_gigachat_with_retry') as mock_call:
        mock_call.return_value = '{"header_start_row": 0, "header_rows": 2, "nested": true, "explanation": "Multi-level header"}'
        preview_rows = [
            ["Name", "Name", "Age", "Age"],
            ["First", "Last", "Years", "Months"],
            ["John", "Doe", 30, 360]
        ]
        start_row, header_rows, nested = get_header_decision("Sheet2", preview_rows)
        assert start_row == 0
        assert header_rows == 2
        assert nested is True


def test_get_header_decision_fallback_on_llm_failure_default():
    """When LLM fails and no heuristic matches, return (0,1,False)."""
    # Make first row entirely long text (over 100 chars) so it does not count as "short"
    long_text = "A" * 150
    preview_rows = [
        [long_text, long_text],  # both cells long → f1 becomes False
        ["Data 1", "Data 2"]
    ]
    with patch('agents.agent.call_gigachat_with_retry', side_effect=Exception("API error")):
        start_row, header_rows, nested = get_header_decision("Sheet3", preview_rows)
        assert start_row == 0
        assert header_rows == 1
        assert nested is False


def test_get_header_decision_fallback_two_short_rows():
    """When both first rows are short, fallback returns (0,2,True)."""
    preview_rows = [
        ["Column A", "Column B"],
        ["Data 1", "Data 2"]
    ]
    with patch('agents.agent.call_gigachat_with_retry', side_effect=Exception("API error")):
        start_row, header_rows, nested = get_header_decision("Sheet4", preview_rows)
        assert start_row == 0
        assert header_rows == 2
        assert nested is True


def test_get_model_name():
    name = get_model_name()
    assert isinstance(name, str)
    assert len(name) > 0


def test_extract_tool_input_dict_nested_json():
    text = """Thought: call tool
Action: dummy_tool
Action Input: {"config": {"a": 1, "b": {"c": 2}}, "flag": true}
"""
    parsed = _extract_tool_input_dict(text)
    assert parsed == {"config": {"a": 1, "b": {"c": 2}}, "flag": True}


def test_extract_final_answer():
    text = "Thought: done\nFinal Answer:  **Hello**  \n"
    assert _extract_final_answer(text) == "**Hello**"


def test_safe_extract_json_from_fence():
    raw = 'Here:\n```json\n{"a": 1}\n```\n'
    assert safe_extract_json(raw) == '{"a": 1}'


def test_safe_extract_json_braces_fallback():
    raw = 'prefix {"x": true} trailing'
    assert safe_extract_json(raw) == '{"x": true}'


from langchain_core.messages import AIMessage


@patch("agents.agent.chat_model")
def test_agent_chat_nested_action_input_invokes_tool(mock_chat):
    mock_tool = MagicMock(return_value={"result": "ok"})
    r1 = AIMessage(
        content=(
        "Thought: test\n"
        "Action: dummy_tool\n"
        'Action Input: {"config": {"nested": {"x": 1}}}\n'
        )
    )
    r2 = AIMessage(content="Final Answer: Done.")
    mock_chat.invoke.side_effect = [r1, r2]

    with patch.dict("agents.agent.TOOL_FUNCTIONS", {"dummy_tool": mock_tool}, clear=False):
        out = agent_chat("run nested tool", max_steps=5)

    assert out == "Done."
    mock_tool.assert_called_once_with(config={"nested": {"x": 1}})


@patch("agents.agent.chat_model")
def test_agent_chat_malformed_action_input_gets_observation(mock_chat):
    r1 = AIMessage(
        content=(
        "Action: run_sql\nAction Input: {not valid json}\n"
        )
    )
    r2 = AIMessage(content="Final Answer: Recovered.")
    mock_chat.invoke.side_effect = [r1, r2]

    mock_sql = MagicMock()
    with patch.dict("agents.agent.TOOL_FUNCTIONS", {"run_sql": mock_sql}, clear=False):
        out = agent_chat("q", max_steps=5)

    assert out == "Recovered."
    mock_sql.assert_not_called()


def test_agent_chat_file_inventory_from_db(tmp_path):
    import db_storage

    orig = db_storage.DB_PATH
    db_storage.DB_PATH = str(tmp_path / "fdb.db")
    try:
        from db_storage import init_db, get_db_connection

        init_db()
        conn = get_db_connection()
        conn.execute(
            "INSERT INTO files (file_hash, filename, upload_time) VALUES (?, ?, ?)",
            ("agfh", "agent_file.xlsx", "2026-04-01"),
        )
        conn.commit()
        conn.close()

        from agents.agent import agent_chat

        out = agent_chat("What files are uploaded?")
        assert "agent_file.xlsx" in out
        assert "agfh" in out
    finally:
        db_storage.DB_PATH = orig