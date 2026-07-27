import json
from unittest.mock import patch

import pytest

import config.column_mapping as column_mapping_config
import storage.database as db_storage
import config.sheet_groups as sheet_groups
from sheet_skills.s2t import (
    S2T_FIELDS,
    S2TExtractionError,
    S2TRowValidationError,
    _build_sheet_llm_prompt,
    _deterministic_sheet_mapping,
    _inspect_candidate_sheets,
    run_s2t_extraction_subagent,
    verify_s2t_transformations,
    write_s2t_transformations_from_plan,
)
from storage.database import get_db_connection, init_db, store_excel_data
from storage.s2t import (
    clear_s2t_transformations,
    list_s2t_transformations,
    refresh_s2t_transformations,
)


@pytest.fixture()
def s2t_db(tmp_path, monkeypatch, mock_embeddings):
    original = db_storage.DB_PATH
    db_storage.DB_PATH = str(tmp_path / "s2t_agent.db")
    mapping_path = tmp_path / "column_mapping.json"
    mapping_path.write_text(
        json.dumps(
            {
                "s2t": {
                    "target_table": ["Target Table"],
                    "target_field": ["Target Column"],
                    "source_table": ["Source Table"],
                    "source_field": ["Source Column"],
                    "transformation_rule": ["SQL Transform"],
                    "primary_key": ["Primary Key"],
                    "target_field_data_type": ["Target Data Type"],
                },
                "source_tables": {
                    "table_name": ["Название таблицы-источника"],
                    "description": ["Описание таблицы-источника"],
                },
                "target_tables": {
                    "table_name": ["Table Name"],
                    "description": ["Table Entity Definition"],
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(column_mapping_config, "COLUMN_MAPPING_PATH", mapping_path)
    column_mapping_config.clear_column_mapping_cache()
    sheet_groups.clear_sheet_groups_cache()
    init_db()
    yield
    db_storage.DB_PATH = original
    column_mapping_config.clear_column_mapping_cache()
    sheet_groups.clear_sheet_groups_cache()


def _store_s2t(columns, rows, sheet_name="S2T"):
    nested = any(isinstance(col, list) for col in columns)
    sheets = [
        {
            "sheet_name": sheet_name,
            "skip_reason": None,
            "header": {"start_row": 0, "row_count": 2 if nested else 1, "nested": nested},
            "columns": columns,
            "data_rows": rows,
        }
    ]
    return store_excel_data(
        b"s2t-agent-bytes",
        "s2t_agent.xlsx",
        "model",
        sheets,
    )


def _store_table_catalogs(source_rows, target_rows):
    sheets = [
        {
            "sheet_name": "Source tables",
            "skip_reason": None,
            "header": {"start_row": 0, "row_count": 1, "nested": False},
            "columns": ["Название таблицы-источника", "Описание таблицы-источника"],
            "data_rows": source_rows,
        },
        {
            "sheet_name": "Target tables",
            "skip_reason": None,
            "header": {"start_row": 0, "row_count": 1, "nested": False},
            "columns": ["Table Name", "Table Entity Definition"],
            "data_rows": target_rows,
        },
    ]
    return store_excel_data(
        b"table-catalog-bytes",
        "table_catalogs.xlsx",
        "model",
        sheets,
    )


def _column_ids(file_id):
    inspection = _inspect_candidate_sheets(file_id)
    sheet = inspection["sheets"][0]
    return sheet, {column["column_name_flat"]: column["column_id"] for column in sheet["columns"]}


def _evidence(field, column_id, method="llm", matched_header_candidate=None):
    return {
        field: {
            "field": field,
            "column_id": column_id,
            "header_path": [field],
            "matched_header_candidate": matched_header_candidate,
            "matched_alias": field,
            "confidence": 0.99,
            "method": method,
            "reason": "test evidence",
        }
    }


def _column_roles_for_sheet(sheet, field_to_flat_name):
    mapping_field_by_name = {
        flat_name: field
        for field, flat_name in field_to_flat_name.items()
        if field in S2T_FIELDS
    }
    return {
        "sheet_name": sheet["sheet_name"],
        "column_roles": [
            {
                "column_name": column["column_name_flat"],
                "mapping_field": mapping_field_by_name.get(column["column_name_flat"]),
            }
            for column in sheet["columns"]
        ],
    }


def _column_roles_for_columns(file_id, field_to_flat_name):
    inspection = _inspect_candidate_sheets(file_id)
    return _column_roles_for_sheet(inspection["sheets"][0], field_to_flat_name)


def _patch_llm_roles(column_roles):
    return patch(
        "sheet_skills.s2t._invoke_llm_plain_text",
        return_value=json.dumps(column_roles, ensure_ascii=False),
    )


def test_s2t_llm_prompt_uses_only_sheet_mapping_and_column_names(s2t_db):
    file_id = _store_s2t(
        [
            ["Target", "Target Tbl"],
            ["Target", "Target Column"],
            ["Ignored", "Data Type"],
        ],
        [["t_prompt", "prompt_id", "uuid"]],
    )
    inspection = _inspect_candidate_sheets(file_id)
    sheet = inspection["sheets"][0]
    draft = _deterministic_sheet_mapping(sheet)

    prompt = _build_sheet_llm_prompt(sheet)

    assert prompt.startswith("Сопоставь полезные колонки")
    assert "Map useful columns" not in prompt
    assert "column_mapping_json" in prompt
    assert '"s2t": {"target_table": ["Target Table"]' in prompt
    assert '"primary_key"' not in prompt
    assert '"target_field_data_type"' not in prompt
    assert "column_name" in prompt
    assert "Target > Target Tbl" in prompt
    assert "sample_values" in prompt
    assert "mapping_field" in prompt
    assert "column_id" not in prompt
    assert "sheet_id" not in prompt
    assert "file_id" not in prompt
    assert "header_path" not in prompt
    assert "column_index" not in prompt
    assert "initial_role" not in prompt
    assert "initial_match" not in prompt
    assert "valid_roles" not in prompt
    assert "critical_roles" not in prompt
    assert "nullable_roles" not in prompt
    assert "role_to_column_mapping_field" not in prompt


def test_s2t_subagent_exact_multilevel_headers_write_minimal_rows(s2t_db):
    file_id = _store_s2t(
        [
            ["Target", "Target Table"],
            ["Target", "Target Column"],
            ["Source", "Source Table"],
            ["Source", "Source Column"],
            ["Transform", "SQL Transform"],
            ["Ignored", "Data Type"],
        ],
        [["t_customer", "customer_id", "src_customer", "id", "cast(id as uuid)", "uuid"]],
    )

    column_roles = _column_roles_for_columns(
        file_id,
        {
            "target_table": "Target > Target Table",
            "target_field": "Target > Target Column",
            "source_table": "Source > Source Table",
            "source_field": "Source > Source Column",
            "transformation_rule": "Transform > SQL Transform",
        },
    )

    with _patch_llm_roles(column_roles) as mock_llm:
        report = run_s2t_extraction_subagent(file_id)

    assert report["status"] == "ok"
    assert report["subagent"] == "usefull_col_extraction"
    assert report["target"] == "s2t_transformations"
    assert report["verification"]["count"] == 1
    assert report["attempts"] == 0
    assert mock_llm.call_count == 0
    assert report["sheets"] == [
        {"sheet_name": "S2T", "method": "deterministic", "attempts": 0}
    ]
    mapping = _deterministic_sheet_mapping(_inspect_candidate_sheets(file_id)["sheets"][0])
    assert set(mapping["field_column_ids"]) == set(S2T_FIELDS)
    assert not any(key.endswith("_column_id") for key in mapping)

    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT row_num, target_table, target_field, source_table, source_field,
               transformation_rule
        FROM s2t_transformations
        """
    )
    row = dict(cur.fetchone())
    cur.execute("SELECT COUNT(*) AS n FROM s2t_transformations WHERE row_num = -1")
    header_rows = int(cur.fetchone()["n"])
    conn.close()

    assert header_rows == 0
    assert row == {
        "row_num": 0,
        "target_table": "t_customer",
        "target_field": "customer_id",
        "source_table": "src_customer",
        "source_field": "id",
        "transformation_rule": "cast(id as uuid)",
    }

    listed = list_s2t_transformations(file_id)
    assert listed["total"] == 1
    assert listed["rows"][0] == {
        "row_num": 0,
        "target_table": "t_customer",
        "target_field": "customer_id",
        "source_table": "src_customer",
        "source_field": "id",
        "transformation_rule": "cast(id as uuid)",
    }


def test_s2t_subagent_uses_deterministic_fuzzy_header_mapping(s2t_db):
    file_id = _store_s2t(
        [
            "Target Tbl",
            "Target Col",
            "Source Tbl",
            "Source Col",
            "SQL Transfrm",
        ],
        [["t_fuzzy", "fuzzy_id", "src_fuzzy", "id", "trim(id)"]],
    )

    column_roles = _column_roles_for_columns(
        file_id,
        {
            "target_table": "Target Tbl",
            "target_field": "Target Col",
            "source_table": "Source Tbl",
            "source_field": "Source Col",
            "transformation_rule": "SQL Transfrm",
        },
    )

    with _patch_llm_roles(column_roles) as mock_llm:
        report = run_s2t_extraction_subagent(file_id)

    assert report["status"] == "ok"
    assert report["verification"]["count"] == 1
    assert report["attempts"] == 0
    assert mock_llm.call_count == 0
    assert report["sheets"][0]["method"] == "deterministic"
    assert report["aliases_added"] >= 2
    assert "Target Tbl" in column_mapping_config.get_field_aliases("s2t", "target_table")
    assert "Target Col" in column_mapping_config.get_field_aliases("s2t", "target_field")
    assert verify_s2t_transformations(file_id)["rows"][0] == {
        "row_num": 0,
        "target_table": "t_fuzzy",
        "target_field": "fuzzy_id",
        "source_table": "src_fuzzy",
        "source_field": "id",
        "transformation_rule": "trim(id)",
    }


def test_s2t_subagent_uses_sheet_group_subagent_to_find_s2t_sheet(s2t_db):
    file_id = _store_s2t(
        ["Target Table", "Target Column", "Source Table", "Source Column", "SQL Transform"],
        [["t_sheet_fuzzy", "sheet_id", "src_sheet", "id", "copy"]],
        sheet_name="SourceToTargt",
    )

    with (
        patch("agents.sheet_group_classifier.invoke_llm_plain_text") as mock_sheet_llm,
        patch("agents.sheet_group_classifier.add_sheet_group_alias", return_value=["SourceToTargt"]) as mock_add_alias,
    ):
        report = run_s2t_extraction_subagent(file_id)

    assert mock_sheet_llm.call_count == 0
    assert report["status"] == "ok"
    assert report["sheets"] == [
        {"sheet_name": "SourceToTargt", "method": "deterministic", "attempts": 0}
    ]
    mock_add_alias.assert_called_once_with("s2t", "SourceToTargt")
    assert report["verification"]["rows"][0]["target_table"] == "t_sheet_fuzzy"


def test_s2t_subagent_uses_llm_mapping_for_unmatched_multilevel_headers(s2t_db):
    file_id = _store_s2t(
        [
            ["Receiver", "Physical destination"],
            ["Receiver", "Destination attribute"],
            ["Provider", "Physical source"],
            ["Provider", "Source attribute"],
            ["Rule", "Expression"],
        ],
        [["t_order", "order_id", "src_order", "id", "direct"]],
    )
    column_roles = _column_roles_for_columns(
        file_id,
        {
            "target_table": "Receiver > Physical destination",
            "target_field": "Receiver > Destination attribute",
            "source_table": "Provider > Physical source",
            "source_field": "Provider > Source attribute",
            "transformation_rule": "Rule > Expression",
        },
    )

    with _patch_llm_roles(column_roles) as mock_llm:
        report = run_s2t_extraction_subagent(file_id)

    assert mock_llm.call_count == 1
    assert report["status"] == "ok"
    assert report["verification"]["count"] == 1
    assert verify_s2t_transformations(file_id)["rows"][0]["target_table"] == "t_order"


def test_s2t_subagent_returns_rejection_reason_to_llm_and_accepts_correction(s2t_db):
    file_id = _store_s2t(
        [
            ["Receiver", "Physical destination"],
            ["Receiver", "Destination attribute"],
            ["Provider", "Physical source"],
            ["Provider", "Source attribute"],
            ["Rule", "Expression"],
        ],
        [["t_retry", "retry_id", "src_retry", "id", "direct"]],
    )
    inspection = _inspect_candidate_sheets(file_id)
    sheet = inspection["sheets"][0]
    good_roles = _column_roles_for_sheet(
        sheet,
        {
            "target_table": "Receiver > Physical destination",
            "target_field": "Receiver > Destination attribute",
            "source_table": "Provider > Physical source",
            "source_field": "Provider > Source attribute",
            "transformation_rule": "Rule > Expression",
        },
    )
    bad_roles = json.loads(json.dumps(good_roles, ensure_ascii=False))
    bad_roles["column_roles"][0]["mapping_field"] = "transformation_rule"

    with patch(
        "sheet_skills.s2t._invoke_llm_plain_text",
        side_effect=[
            json.dumps(bad_roles, ensure_ascii=False),
            json.dumps(good_roles, ensure_ascii=False),
        ],
    ) as mock_llm:
        report = run_s2t_extraction_subagent(file_id)

    assert mock_llm.call_count == 2
    correction_prompt = mock_llm.call_args_list[1].args[0]
    assert "Предыдущий ответ отклонён валидатором" in correction_prompt
    assert "mapping_field transformation_rule is assigned to multiple columns" in correction_prompt
    assert str(sheet["sheet_id"]) not in correction_prompt
    assert str(file_id) not in correction_prompt
    assert report["attempts"] == 2
    assert report["sheets"][0] == {
        "sheet_name": "S2T",
        "method": "llm",
        "attempts": 2,
    }
    assert report["verification"]["count"] == 1


def test_s2t_subagent_bad_json_makes_correction_request_and_does_not_write(s2t_db):
    file_id = _store_s2t(
        [["Receiver", "Physical destination"], ["Receiver", "Destination attribute"]],
        [["t_retry", "retry_id"]],
    )
    with patch(
        "sheet_skills.s2t._invoke_llm_plain_text",
        return_value="not-json",
    ) as mock_llm:
        with pytest.raises(S2TExtractionError) as exc:
            run_s2t_extraction_subagent(file_id)

    assert mock_llm.call_count == 2
    assert exc.value.report["attempts"] == 2
    assert verify_s2t_transformations(file_id)["count"] == 0


def test_s2t_subagent_incomplete_llm_response_makes_correction_request_and_does_not_write(s2t_db):
    file_id = _store_s2t(
        ["Target Table", "Target Column", "Ignored"],
        [["t_retry_columns", "retry_id", "not_s2t"]],
    )
    good_roles = _column_roles_for_columns(
        file_id,
        {
            "target_table": "Target Table",
            "target_field": "Target Column",
        },
    )
    bad_roles = {
        **good_roles,
        "column_roles": good_roles["column_roles"][:-1],
    }

    with patch(
        "sheet_skills.s2t._invoke_llm_plain_text",
        return_value=json.dumps(bad_roles, ensure_ascii=False),
    ) as mock_llm:
        with pytest.raises(S2TExtractionError) as exc:
            run_s2t_extraction_subagent(file_id)

    assert mock_llm.call_count == 2
    assert exc.value.report["attempts"] == 2
    assert verify_s2t_transformations(file_id)["count"] == 0


def test_s2t_subagent_calls_llm_once_per_s2t_sheet(s2t_db):
    sheets = [
        {
            "sheet_name": "S2T",
            "skip_reason": None,
            "header": {"start_row": 0, "row_count": 1, "nested": False},
            "columns": ["Target Table", "Target Column"],
            "data_rows": [["t_first", "first_id"]],
        },
        {
            "sheet_name": "SourceToTarget",
            "skip_reason": None,
            "header": {"start_row": 0, "row_count": 1, "nested": False},
            "columns": ["Target Table", "Target Column"],
            "data_rows": [["t_second", "second_id"]],
        },
    ]
    file_id = store_excel_data(
        b"s2t-two-sheets",
        "two_s2t_sheets.xlsx",
        "model",
        sheets,
    )
    inspection = _inspect_candidate_sheets(file_id)
    responses = [
        json.dumps(
            _column_roles_for_sheet(
                sheet,
                {
                    "target_table": "Target Table",
                    "target_field": "Target Column",
                },
            ),
            ensure_ascii=False,
        )
        for sheet in inspection["sheets"]
    ]

    with patch("sheet_skills.s2t._invoke_llm_plain_text", side_effect=responses) as mock_llm:
        report = run_s2t_extraction_subagent(file_id)

    assert mock_llm.call_count == 2
    assert report["attempts"] == 2
    assert report["verification"]["count"] == 2


def test_s2t_subagent_calls_llm_only_for_incomplete_s2t_sheet(s2t_db):
    sheets = [
        {
            "sheet_name": "S2T",
            "skip_reason": None,
            "header": {"start_row": 0, "row_count": 1, "nested": False},
            "columns": ["Target Table", "Target Column", "Source Table", "Source Column", "SQL Transform"],
            "data_rows": [["t_first", "first_id", "src_first", "id", "copy"]],
        },
        {
            "sheet_name": "SourceToTarget",
            "skip_reason": None,
            "header": {"start_row": 0, "row_count": 1, "nested": False},
            "columns": ["Receiver Table", "Receiver Field", "Provider Table", "Provider Field", "Rule Text"],
            "data_rows": [["t_second", "second_id", "src_second", "id", "trim(id)"]],
        },
    ]
    file_id = store_excel_data(
        b"s2t-mixed-complete-incomplete",
        "mixed_s2t_sheets.xlsx",
        "model",
        sheets,
    )
    inspection = _inspect_candidate_sheets(file_id)
    incomplete_sheet = next(sheet for sheet in inspection["sheets"] if sheet["sheet_name"] == "SourceToTarget")
    response = json.dumps(
        _column_roles_for_sheet(
            incomplete_sheet,
            {
                "target_table": "Receiver Table",
                "target_field": "Receiver Field",
                "source_table": "Provider Table",
                "source_field": "Provider Field",
                "transformation_rule": "Rule Text",
            },
        ),
        ensure_ascii=False,
    )

    with patch("sheet_skills.s2t._invoke_llm_plain_text", return_value=response) as mock_llm:
        report = run_s2t_extraction_subagent(file_id)

    assert mock_llm.call_count == 1
    assert report["attempts"] == 1
    assert report["verification"]["count"] == 2
    assert [sheet["method"] for sheet in report["sheets"]] == [
        "deterministic",
        "llm",
    ]


def test_s2t_subagent_bad_response_returns_error_without_fallback_write(s2t_db):
    file_id = _store_s2t(
        [["Receiver", "Physical destination"], ["Receiver", "Destination attribute"]],
        [["t_fail", "fail_id"]],
    )

    with patch("sheet_skills.s2t._invoke_llm_plain_text", return_value="not-json"):
        with pytest.raises(S2TExtractionError) as exc:
            run_s2t_extraction_subagent(file_id)

    assert exc.value.report["status"] == "error"
    assert exc.value.report["attempts"] == 2
    assert verify_s2t_transformations(file_id)["count"] == 0


def test_write_tool_rejects_duplicate_column_field_and_keeps_existing_rows(s2t_db):
    file_id = _store_s2t(
        [["Target", "Target Table"], ["Target", "Target Column"]],
        [["t1", "c1"]],
    )
    column_roles = _column_roles_for_columns(
        file_id,
        {
            "target_table": "Target > Target Table",
            "target_field": "Target > Target Column",
        },
    )
    with _patch_llm_roles(column_roles):
        assert refresh_s2t_transformations(file_id) == 1
    sheet, column_ids = _column_ids(file_id)
    duplicate_column_id = column_ids["Target > Target Table"]
    bad_mapping = {
        "sheet_id": sheet["sheet_id"],
        "field_column_ids": {
            "target_table": duplicate_column_id,
            "target_field": duplicate_column_id,
        },
        "evidence": {
            **_evidence("target_table", duplicate_column_id),
            **_evidence("target_field", duplicate_column_id),
        },
    }

    with pytest.raises(ValueError):
        write_s2t_transformations_from_plan(file_id, [bad_mapping])

    assert verify_s2t_transformations(file_id)["count"] == 1


def test_write_tool_reports_missing_target_table_and_keeps_existing_rows(s2t_db):
    file_id = _store_s2t(
        ["Target Table", "Target Column"],
        [[None, "c1"]],
    )
    sheet, column_ids = _column_ids(file_id)
    target_table_id = column_ids["Target Table"]
    target_field_id = column_ids["Target Column"]
    mapping = {
        "sheet_id": sheet["sheet_id"],
        "field_column_ids": {
            "target_table": target_table_id,
            "target_field": target_field_id,
        },
        "evidence": {
            **_evidence("target_table", target_table_id),
            **_evidence("target_field", target_field_id),
        },
    }
    conn = get_db_connection()
    conn.execute(
        """
        INSERT INTO s2t_transformations
        (file_id, sheet_id, sheet_name, row_num, target_table, target_field)
        VALUES (777, 777, 'old', 7, 'old_target', 'old_column')
        """
    )
    conn.commit()
    conn.close()

    with pytest.raises(S2TRowValidationError) as exc:
        write_s2t_transformations_from_plan(file_id, [mapping])

    assert exc.value.report["stage"] == "validate_rows"
    assert exc.value.report["validation_errors"] == [
        {
            "file_id": file_id,
            "sheet_id": sheet["sheet_id"],
            "sheet_name": "S2T",
            "row_num": 0,
            "field": "target_table",
            "error": "В строке S2T не заполнена целевая таблица",
        }
    ]
    conn = get_db_connection()
    old_row = conn.execute(
        "SELECT target_table, target_field FROM s2t_transformations WHERE file_id = 777"
    ).fetchone()
    conn.close()
    assert dict(old_row) == {"target_table": "old_target", "target_field": "old_column"}


def test_refresh_s2t_transformations_rebuilds_global_table_and_preserves_duplicates(s2t_db):
    file_id = _store_s2t(
        ["Target Table", "Target Column", "Source Table", "Source Column", "SQL Transform"],
        [
            ["t1", "c1", "src", "src_c", "copy"],
            ["t1", "c1", "src", "src_c", "copy"],
            ["t1", "c2", "src", "src_c2", "copy"],
        ],
    )
    conn = get_db_connection()
    conn.execute(
        """
        INSERT INTO s2t_transformations
        (id, file_id, sheet_id, sheet_name, row_num, target_table, target_field)
        VALUES (999, ?, 999, 'S2T', 99, 'old', 'old')
        """,
        (file_id,),
    )
    conn.execute(
        """
        INSERT INTO s2t_transformations
        (id, file_id, sheet_id, sheet_name, row_num, target_table, target_field)
        VALUES (1000, 777, 777, 'S2T', 7, 'foreign', 'foreign')
        """
    )
    conn.commit()
    conn.close()

    column_roles = _column_roles_for_columns(
        file_id,
        {
            "target_table": "Target Table",
            "target_field": "Target Column",
            "source_table": "Source Table",
            "source_field": "Source Column",
            "transformation_rule": "SQL Transform",
        },
    )
    with _patch_llm_roles(column_roles):
        assert refresh_s2t_transformations(file_id) == 3

    rows = verify_s2t_transformations(file_id, limit=10)["rows"]
    assert rows == [
        {
            "row_num": 0,
            "target_table": "t1",
            "target_field": "c1",
            "source_table": "src",
            "source_field": "src_c",
            "transformation_rule": "copy",
        },
        {
            "row_num": 1,
            "target_table": "t1",
            "target_field": "c1",
            "source_table": "src",
            "source_field": "src_c",
            "transformation_rule": "copy",
        },
        {
            "row_num": 2,
            "target_table": "t1",
            "target_field": "c2",
            "source_table": "src",
            "source_field": "src_c2",
            "transformation_rule": "copy",
        },
    ]

    conn = get_db_connection()
    foreign_rows = conn.execute(
        "SELECT COUNT(*) AS n FROM s2t_transformations WHERE file_id = 777"
    ).fetchone()["n"]
    conn.close()
    assert foreign_rows == 0


def test_table_catalog_extraction_writes_name_description_and_preserves_duplicates(s2t_db):
    file_id = _store_table_catalogs(
        [
            ["src_same", "Одинаковое описание"],
            ["src_same", "Одинаковое описание"],
        ],
        [
            ["t_same", "Same description"],
            ["t_same", "Same description"],
        ],
    )

    report = run_s2t_extraction_subagent(file_id)

    assert report["status"] == "ok"
    assert report["verification"]["count"] == 0
    assert report["table_catalogs"]["targets"]["source_tables"]["count"] == 2
    assert report["table_catalogs"]["targets"]["target_tables"]["count"] == 2

    conn = get_db_connection()
    source_rows = [
        dict(row)
        for row in conn.execute(
            """
            SELECT row_num, table_name, description, description_embedding
            FROM source_tables
            WHERE file_id = ?
            ORDER BY row_num
            """,
            (file_id,),
        ).fetchall()
    ]
    target_rows = [
        dict(row)
        for row in conn.execute(
            """
            SELECT row_num, table_name, description, description_embedding
            FROM target_tables
            WHERE file_id = ?
            ORDER BY row_num
            """,
            (file_id,),
        ).fetchall()
    ]
    conn.close()

    assert source_rows == [
        {
            "row_num": 0,
            "table_name": "src_same",
            "description": "Одинаковое описание",
            "description_embedding": "embedding:Одинаковое описание".encode("utf-8"),
        },
        {
            "row_num": 1,
            "table_name": "src_same",
            "description": "Одинаковое описание",
            "description_embedding": "embedding:Одинаковое описание".encode("utf-8"),
        },
    ]
    assert target_rows == [
        {
            "row_num": 0,
            "table_name": "t_same",
            "description": "Same description",
            "description_embedding": b"embedding:Same description",
        },
        {
            "row_num": 1,
            "table_name": "t_same",
            "description": "Same description",
            "description_embedding": b"embedding:Same description",
        },
    ]


def test_clear_s2t_transformations_deletes_only_current_file(s2t_db):
    conn = get_db_connection()
    conn.executemany(
        """
        INSERT INTO s2t_transformations
        (id, file_id, sheet_id, sheet_name, row_num, target_table)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        [
            (1, 10, 100, "S2T", 1, "t1"),
            (2, 10, 100, "S2T", 2, "t2"),
            (3, 20, 200, "S2T", 1, "t3"),
        ],
    )
    conn.commit()
    conn.close()

    assert clear_s2t_transformations(10) == 2

    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT id, file_id FROM s2t_transformations ORDER BY id")
    rows = [dict(row) for row in cur.fetchall()]
    conn.close()

    assert rows == [{"id": 3, "file_id": 20}]


def test_list_s2t_transformations_without_file_id_reads_global_table(s2t_db):
    conn = get_db_connection()
    conn.executemany(
        """
        INSERT INTO s2t_transformations
        (id, file_id, sheet_id, sheet_name, row_num, target_table)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        [
            (11, 10, 100, "S2T", 1, "t_first"),
            (12, 20, 200, "S2T", 2, "t_second"),
        ],
    )
    conn.commit()
    conn.close()

    result = list_s2t_transformations(file_id=None, limit=10)

    assert result["scope"] == "global"
    assert result["total"] == 2
    assert [row["target_table"] for row in result["rows"]] == [
        "t_first",
        "t_second",
    ]
    assert "file_id" not in result
