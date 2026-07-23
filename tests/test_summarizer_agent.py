import pytest
from unittest.mock import patch, MagicMock
from agents.summarizer_agent import extract_schema, generate_summary, summarize_file


def test_extract_schema_parses_prose_wrapped_json():
    state = {
        "raw_sheets": [
            {
                "sheet_name": "S1",
                "columns": ["Name", "Age"],
                "sample_rows": [],
                "description_cells": [],
            }
        ],
        "important_values": [],
        "source_description_snippets": [],
        "schema": {},
        "section_summaries": [],
        "final_summary": "",
        "validation_errors": [],
    }
    prose = (
        "Вот JSON:\n"
        '{"business_domain": "кредитование", "key_entities": ["договор"], '
        '"description_highlights": ["ставка"]}'
    )
    with patch("agents.summarizer_agent.call_gigachat", return_value=prose):
        out = extract_schema(state)
    assert out["schema"]["business_domain"] == "кредитование"
    assert out["schema"]["key_entities"] == ["договор"]
    assert out["validation_errors"] == []


@patch('agents.summarizer_agent.get_db_connection')
@patch('agents.summarizer_agent.call_gigachat')
def test_generate_summary(mock_call_gigachat, mock_get_db_conn):
    # Mock database connection and cursor
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value = mock_cursor
    mock_get_db_conn.return_value = mock_conn

    # Mock fetchone calls in order of execution in fetch_file_data:
    # 1. filename
    # 2. sheet_hash (sheet_hashes_by_name loop for Sheet1)
    # 3. sheet_hash (sample_rows loop for Sheet1)
    mock_cursor.fetchone.side_effect = [
        {"filename": "test.xlsx"},
        {"sheet_hash": "hash1"},
        {"sheet_hash": "hash1"},
    ]

    # Mock fetchall calls in order:
    # 1. JOIN sheets+columns (2 columns)
    # 2. per-sheet column list (column_hash + column_name_flat) for description scan
    # 3. sample rows for Sheet1
    # 4. important values
    mock_cursor.fetchall.side_effect = [
        [
            {"sheet_name": "Sheet1", "column_name_flat": "Name"},
            {"sheet_name": "Sheet1", "column_name_flat": "Age"},
        ],
        [
            {"column_hash": "ch1", "column_name_flat": "Name"},
            {"column_hash": "ch2", "column_name_flat": "Age"},
        ],
        [
            {"row_num": 1, "row_values": "Alice | 30"},
            {"row_num": 2, "row_values": "Bob | 25"},
        ],
        [
            {"value": "КЮЛ"},
            {"value": "S2T"},
        ],
    ]

    # Mock GigaChat responses for the four steps
    mock_call_gigachat.side_effect = [
        '{"key_entities": ["кредиты"], "business_domain": "banking", "description_highlights": []}',
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