import pytest
from unittest.mock import patch
from agent import get_header_decision, get_model_name


@pytest.fixture
def mock_llm_success():
    with patch('agent.call_gigachat_with_retry') as mock_call:
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
    with patch('agent.call_gigachat_with_retry') as mock_call:
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
    with patch('agent.call_gigachat_with_retry', side_effect=Exception("API error")):
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
    with patch('agent.call_gigachat_with_retry', side_effect=Exception("API error")):
        start_row, header_rows, nested = get_header_decision("Sheet4", preview_rows)
        assert start_row == 0
        assert header_rows == 2
        assert nested is True


def test_get_model_name():
    name = get_model_name()
    assert isinstance(name, str)
    assert len(name) > 0