import json

import pytest
from unittest.mock import patch, MagicMock
from agents.summarizer_agent import (
    SYSTEM_PROMPT,
    ensure_file_description,
    fetch_file_data,
    generate_description_from_summary,
    generate_description_update_from_user_query,
    generate_summary,
    summarize_file,
    update_file_description_from_user_query,
)

@patch('agents.summarizer_agent.get_db_connection')
@patch('agents.summarizer_agent.call_gigachat')
def test_generate_summary(mock_call_gigachat, mock_get_db_conn):
    # Mock database connection and cursor
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value = mock_cursor
    mock_get_db_conn.return_value = mock_conn

    mock_cursor.fetchone.return_value = {"exists": 1}

    # 1. saved headers; 2. cells from the first five rows.
    mock_cursor.fetchall.side_effect = [
        [
            {
                "sheet_id": 1,
                "sheet_name": "Sheet1",
                "headers_json": '[{"index": 0, "flat": "Name", "column_id": 1}, {"index": 1, "flat": "Age", "column_id": 2}]',
            },
        ],
        [
            {"row_num": 1, "column_id": 1, "value": "Alice"},
            {"row_num": 1, "column_id": 2, "value": "30"},
            {"row_num": 2, "column_id": 1, "value": "Bob"},
            {"row_num": 2, "column_id": 2, "value": "25"},
        ],
    ]

    mock_call_gigachat.return_value = "Final summary."

    summary = generate_summary(1)
    assert summary == "Final summary."
    mock_call_gigachat.assert_called_once()
    payload = json.loads(mock_call_gigachat.call_args.args[0])
    assert payload == {
        "sheets": [
            {
                "sheet_name": "Sheet1",
                "columns": ["Name", "Age"],
                "rows": [["Alice", "30"], ["Bob", "25"]],
            }
        ]
    }
    assert SYSTEM_PROMPT == "Сделай краткое саммари на русском языке по переданным табличным данным."


def test_fetch_file_data_returns_five_aligned_rows_for_each_sheet(
    temp_db,
    sample_excel_bytes,
):
    from storage.database import store_excel_data

    sheets = [
        {
            "sheet_name": "SheetA",
            "skip_reason": None,
            "header": {"start_row": 0, "row_count": 1, "nested": False},
            "columns": ["A", "B"],
            "data_rows": [[f"a-{index}", index] for index in range(7)],
        },
        {
            "sheet_name": "SheetB",
            "skip_reason": None,
            "header": {"start_row": 0, "row_count": 1, "nested": False},
            "columns": ["C", "D"],
            "data_rows": [[f"b-{index}", index * 10] for index in range(7)],
        },
    ]
    file_id = store_excel_data(
        sample_excel_bytes,
        "summary.xlsx",
        "model",
        sheets,
    )

    snapshot = fetch_file_data(file_id)

    assert snapshot["sheets"] == [
        {
            "sheet_name": "SheetA",
            "columns": ["A", "B"],
            "rows": [[f"a-{index}", str(index)] for index in range(5)],
        },
        {
            "sheet_name": "SheetB",
            "columns": ["C", "D"],
            "rows": [[f"b-{index}", str(index * 10)] for index in range(5)],
        },
    ]


@patch("agents.summarizer_agent.call_gigachat")
def test_description_prompts_are_russian_utf8(mock_call_gigachat):
    mock_call_gigachat.return_value = "Описание"

    generate_description_from_summary("Саммари")
    generate_description_update_from_user_query("Описание", "Саммари", "Уточнение")

    prompts = [call.args[0] for call in mock_call_gigachat.call_args_list]
    assert "Сформируй краткое описание данных" in prompts[0]
    assert "Обнови краткое описание данных" in prompts[1]
    assert all("вЂ" not in prompt and "РЎС" not in prompt for prompt in prompts)

@patch('agents.summarizer_agent.generate_summary')
@patch('agents.summarizer_agent.update_file_summary')
def test_summarize_file(mock_update, mock_generate):
    mock_generate.return_value = "Generated summary"
    result = summarize_file(1, save=True)
    assert result == "Generated summary"
    mock_update.assert_called_once_with(1, "Generated summary")


@patch("agents.summarizer_agent._file_text_fields")
def test_ensure_file_description_uses_cached_value(mock_file_fields):
    mock_file_fields.return_value = {
        "file_id": 1,
        "filename": "test.xlsx",
        "summary": "Long summary",
        "description": "Cached description",
    }

    result = ensure_file_description(1)

    assert result == "Cached description"


@patch("agents.summarizer_agent.update_file_description")
@patch("agents.summarizer_agent.generate_description_from_summary")
@patch("agents.summarizer_agent._file_text_fields")
def test_ensure_file_description_generates_and_saves(
    mock_file_fields,
    mock_generate_description,
    mock_update_description,
):
    mock_file_fields.return_value = {
        "file_id": 1,
        "filename": "test.xlsx",
        "summary": "Long summary",
        "description": None,
    }
    mock_generate_description.return_value = "Short description"

    result = ensure_file_description(1)

    assert result == "Short description"
    mock_generate_description.assert_called_once_with("Long summary")
    mock_update_description.assert_called_once_with(1, "Short description")


@patch("agents.summarizer_agent.update_file_description")
@patch("agents.summarizer_agent.generate_description_update_from_user_query")
@patch("agents.summarizer_agent._file_text_fields")
@patch("agents.summarizer_agent.ensure_file_description")
def test_update_file_description_from_user_query(
    mock_ensure_description,
    mock_file_fields,
    mock_generate_updated_description,
    mock_update_description,
):
    mock_ensure_description.return_value = "Current description"
    mock_file_fields.return_value = {
        "file_id": 1,
        "filename": "test.xlsx",
        "summary": "Long summary",
        "description": "Current description",
    }
    mock_generate_updated_description.return_value = "Updated description"

    result = update_file_description_from_user_query(
        1,
        "Добавь акцент на кредитные договоры",
    )

    assert result == "Updated description"
    mock_ensure_description.assert_called_once_with(1, refresh=False, save=True)
    mock_generate_updated_description.assert_called_once_with(
        current_description="Current description",
        summary="Long summary",
        user_query="Добавь акцент на кредитные договоры",
    )
    mock_update_description.assert_called_once_with(1, "Updated description")
