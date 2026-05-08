import pytest
from unittest.mock import patch, MagicMock
from agents.summarizer_agent import generate_summary, summarize_file

@patch('agents.summarizer_agent.get_db_connection')
@patch('agents.summarizer_agent.call_gigachat')
def test_generate_summary(mock_call_gigachat, mock_get_db_conn):
    # Mock database connection and cursor
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value = mock_cursor
    mock_get_db_conn.return_value = mock_conn

    # Mock fetchone calls:
    # 1. filename query -> returns filename
    # 2. sheet_hash query (for Sheet1) -> returns sheet_hash
    mock_cursor.fetchone.side_effect = [
        {"filename": "test.xlsx"},
        {"sheet_hash": "hash1"}
    ]

    # Mock fetchall calls in order:
    # 1. columns query -> returns column rows (with sheet_name)
    # 2. sample rows query -> returns rows with row_num and row_values
    # 3. important values query -> returns value rows
    mock_cursor.fetchall.side_effect = [
        # Columns
        [
            {"sheet_name": "Sheet1", "column_name_flat": "Name"},
            {"sheet_name": "Sheet1", "column_name_flat": "Age"}
        ],
        # Sample rows for Sheet1 (must have 'row_num' and 'row_values')
        [
            {"row_num": 1, "row_values": "Alice | 30"},
            {"row_num": 2, "row_values": "Bob | 25"}
        ],
        # Important values
        [
            {"value": "КЮЛ"},
            {"value": "S2T"}
        ]
    ]

    # Mock GigaChat responses for the four steps
    mock_call_gigachat.side_effect = [
        '{"key_entities": ["кредиты"], "business_domain": "banking"}',  # extract_schema
        "Structural summary text.",  # structural_summary
        "Domain summary text.",      # domain_summary
        "Final synthesized summary." # synthesize
    ]

    summary = generate_summary("fake_hash")
    assert isinstance(summary, str)
    assert summary == "Final synthesized summary."

@patch('agents.summarizer_agent.generate_summary')
@patch('agents.summarizer_agent.update_file_summary')
def test_summarize_file(mock_update, mock_generate):
    mock_generate.return_value = "Generated summary"
    result = summarize_file("hash", save=True)
    assert result == "Generated summary"
    mock_update.assert_called_once_with("hash", "Generated summary")