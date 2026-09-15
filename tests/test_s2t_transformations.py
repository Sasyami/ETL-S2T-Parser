import json
from unittest.mock import patch

import pytest

import config.column_mapping as column_mapping_config
import storage.database as db_storage
import config.sheet_groups as sheet_groups
from config.table_layers import resolve_sheet_layers
from sheet_skills.s2t import (
    S2T_FIELDS,
    S2TExtractionError,
    S2TRowValidationError,
    _build_sheet_llm_prompt,
    _deterministic_sheet_mapping,
    _inspect_candidate_sheets,
    _sheet_mapping_from_column_roles,
    run_s2t_extraction_subagent,
    verify_s2t_transformations,
    write_s2t_transformations_from_plan,
)
from agents.sheet_group_classifier import classify_file_sheet_groups
from sheet_skills.additional_objects import (
    _parse_object,
    extract_additional_object_transformations,
)
from sheet_skills.structured_metadata import extract_structured_metadata
from sheet_skills.table_catalog import extract_table_catalogs
from sheet_skills.column_catalog import (
    ColumnCatalogExtractionError,
    _is_null_question,
    _merge_catalog,
    _normalise_flag,
    backfill_column_description_embeddings,
    extract_column_catalogs,
)
from storage.database import get_db_connection, init_db, store_excel_data
from storage.s2t import (
    clear_s2t_transformations,
    insert_s2t_transformations,
    list_s2t_transformations,
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
                    "source_primary_key": ["Primary Key"],
                    "target_primary_key": ["Primary Key"],
                    "source_not_null": ["not null"],
                    "target_not_null": ["not null"],
                    "source_description": ["Description"],
                    "target_description": ["Description"],
                    "source_field_data_type": ["Data Type"],
                    "target_field_data_type": ["Target Data Type", "Data Type"],
                },
                "source_tables": {
                    "table_name": ["Название таблицы-источника"],
                    "description": ["Описание таблицы-источника"],
                },
                "target_tables": {
                    "table_name": ["Table Name"],
                    "description": ["Table Entity Definition"],
                },
                "source_columns": {
                    "table_name": ["table", "Название таблицы"],
                    "column_name": ["column", "Название поля"],
                    "data_type": ["datatype", "Тип поля"],
                    "primary_key": ["pk", "PK"],
                    "not_null": ["notnull", "not null", "NULL?"],
                    "description": ["comment", "Описание поля"],
                },
                "target_columns": {
                    "table_name": ["table", "Column Table Name"],
                    "column_name": ["column", "Column Name"],
                    "data_type": ["datatype", "Column Datatype"],
                    "primary_key": ["pk", "Column IsPK"],
                    "not_null": ["notnull", "Column Null Option"],
                    "description": ["comment", "Column Attribute Definition"],
                },
                "additional_objects": {
                    "name": ["name", "Название объекта", "Имя объекта для S2T"],
                    "sql": ["sql", "SQL"],
                },
                "pxf_to_a": {
                    "external_a_table": ["Название внешней A-таблицы"],
                    "materialized_storage": ["Название таблицы в Hadoop"],
                    "replica_table": ["Название таблицы реплики"],
                    "sod": ["СОД"],
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
        "table_catalogs.xlsx",
        "model",
        sheets,
    )


def _store_structured_metadata():
    return store_excel_data(
        "structured_metadata.xlsx",
        "model",
        [
            {
                "sheet_name": "Additional objects",
                "skip_reason": None,
                "header": {"start_row": 0, "row_count": 1, "nested": False},
                "columns": ["Имя объекта для S2T", "SQL"],
                "data_rows": [
                    ["view_orders", "SELECT * FROM raw.orders"],
                    ["view_orders", "SELECT * FROM raw.orders"],
                ],
            },
            {
                "sheet_name": "pxf_to_a",
                "skip_reason": None,
                "header": {"start_row": 0, "row_count": 1, "nested": False},
                "columns": [
                    "Название внешней A-таблицы",
                    "Название таблицы в Hadoop",
                    "Название таблицы реплики",
                    "СОД",
                ],
                "data_rows": [
                    ["ext_orders", "mat_orders", "replica_orders", "SOD-1"],
                ],
            },
        ],
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


def test_table_layer_rules_use_sheet_group_instead_of_table_names():
    expected = {"source_layer": "B", "target_layer": "T"}
    assert resolve_sheet_layers("S2T") == expected
    assert resolve_sheet_layers("SourceToTarget") == expected
    assert resolve_sheet_layers("Additional objects") == {
        "source_layer": None,
        "target_layer": "B",
    }
    assert resolve_sheet_layers("Unconfigured sheet") == {
        "source_layer": None,
        "target_layer": None,
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

    prompt = _build_sheet_llm_prompt(sheet, draft)

    assert prompt.startswith("Сопоставь полезные колонки")
    assert "Map useful columns" not in prompt
    assert "column_mapping_json" in prompt
    assert '"s2t": {"target_table": ["Target Table"]' in prompt
    assert '"primary_key"' not in prompt
    assert '"target_field_data_type"' not in prompt
    assert "column_name" in prompt
    assert "Target > Target Tbl" in prompt
    assert "sample_values" in prompt
    assert "occurrence" in prompt
    assert "side_hint" in prompt
    assert "previous_column" in prompt
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


def test_s2t_llm_mapping_distinguishes_equal_headers_by_occurrence(s2t_db):
    file_id = _store_s2t(
        ["Target Table", "Same Name", "Source Table", "Same Name"],
        [["mart.t", "target_id", "raw.t", "source_id"]],
    )
    sheet = _inspect_candidate_sheets(file_id)["sheets"][0]

    mapping = _sheet_mapping_from_column_roles(
        {
            "sheet_name": "S2T",
            "column_roles": [
                {"column_name": "Target Table", "occurrence": 1, "mapping_field": "target_table"},
                {"column_name": "Same Name", "occurrence": 1, "mapping_field": "target_field"},
                {"column_name": "Source Table", "occurrence": 1, "mapping_field": "source_table"},
                {"column_name": "Same Name", "occurrence": 2, "mapping_field": "source_field"},
            ],
        },
        sheet,
    )

    assert mapping["field_column_ids"]["target_field"] == 2
    assert mapping["field_column_ids"]["source_field"] == 4


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
        {
            "sheet_name": "S2T",
            "method": "deterministic",
            "attempts": 0,
            "rows_written": 1,
            "empty_target_columns_count": 0,
        }
    ]
    assert report["empty_target_columns_count"] == 0
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
        "source_layer": "B",
        "target_layer": "T",
    }


def test_s2t_upload_pipeline_assigns_source_and_target_layers(s2t_db):
    file_id = _store_s2t(
        [
            "Target Table",
            "Target Column",
            "Source Table",
            "Source Column",
            "SQL Transform",
        ],
        [
            [
                "target_without_layer_prefix",
                "customer_id",
                "source_without_layer_prefix",
                "id",
                "copy",
            ],
            [
                "another_target",
                "agreement_id",
                "another_source, second_source",
                "id",
                "copy",
            ],
        ],
    )

    report = run_s2t_extraction_subagent(file_id)

    assert report["written"] == 2
    rows = list_s2t_transformations(file_id)["rows"]
    assert [(row["source_layer"], row["target_layer"]) for row in rows] == [
        ("B", "T"),
        ("B", "T"),
    ]


def test_s2t_upload_pipeline_reports_empty_target_columns_per_sheet(s2t_db):
    file_id = _store_s2t(
        [
            "Target Table",
            "Target Column",
            "Source Table",
            "Source Column",
            "SQL Transform",
        ],
        [
            ["t_customer", "customer_id", "src_customer", "id", "copy"],
            ["t_customer", None, "src_customer", "name", "trim(name)"],
        ],
    )

    report = run_s2t_extraction_subagent(file_id)

    assert report["written"] == 2
    assert report["verification"]["count"] == 2
    assert report["empty_target_columns_count"] == 1
    assert report["sheets"] == [
        {
            "sheet_name": "S2T",
            "method": "deterministic",
            "attempts": 0,
            "rows_written": 2,
            "empty_target_columns_count": 1,
        }
    ]
    assert verify_s2t_transformations(file_id, limit=10)["rows"][1][
        "target_field"
    ] is None


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
        "source_layer": "B",
        "target_layer": "T",
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
        {
            "sheet_name": "SourceToTargt",
            "method": "deterministic",
            "attempts": 0,
            "rows_written": 1,
            "empty_target_columns_count": 0,
        }
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


def test_s2t_subagent_bad_json_fails_after_one_request_and_does_not_write(s2t_db):
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

    assert mock_llm.call_count == 1
    assert exc.value.report["attempts"] == 1
    assert verify_s2t_transformations(file_id)["count"] == 0


def test_s2t_subagent_incomplete_llm_response_fails_after_one_request(s2t_db):
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

    assert mock_llm.call_count == 1
    assert exc.value.report["attempts"] == 1
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
    assert exc.value.report["attempts"] == 1
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
        assert run_s2t_extraction_subagent(file_id)["verification"]["count"] == 1
    sheet, column_ids = _column_ids(file_id)
    duplicate_column_id = column_ids["Target > Target Table"]
    bad_mapping = {
        "sheet_name": sheet["sheet_name"],
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
        "sheet_name": sheet["sheet_name"],
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
        (file_id, sheet_name, row_num, target_table, target_field)
        VALUES (777, 'old', 7, 'old_target', 'old_column')
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


def test_write_tool_ignores_rows_without_any_mapped_s2t_values(s2t_db):
    file_id = _store_s2t(
        ["Target Table", "Target Column", "Дата"],
        [
            ["t1", "c1", None],
            [None, None, "2024-03-19"],
        ],
    )
    sheet, column_ids = _column_ids(file_id)
    mapping = {
        "sheet_name": sheet["sheet_name"],
        "field_column_ids": {
            "target_table": column_ids["Target Table"],
            "target_field": column_ids["Target Column"],
        },
        "evidence": {
            **_evidence("target_table", column_ids["Target Table"]),
            **_evidence("target_field", column_ids["Target Column"]),
        },
    }

    result = write_s2t_transformations_from_plan(file_id, [mapping])

    assert result["count"] == 1
    assert verify_s2t_transformations(file_id)["count"] == 1


def test_write_tool_keeps_all_rows_with_empty_target_columns_and_reports_count(
    s2t_db,
):
    file_id = _store_s2t(
        [
            "Target Table",
            "Target Column",
            "Source Table",
            "Source Column",
            "SQL Transform",
        ],
        [
            ["t1", "c1", "src", "src_c1", "copy"],
            ["t1", None, "src", "src_c2", "coalesce(src_c2, 0)"],
            ["t1", None, "src", "src_c2", "coalesce(src_c2, 0)"],
            [None, None, None, None, None],
        ],
    )
    sheet, column_ids = _column_ids(file_id)
    mapping = {
        "sheet_name": sheet["sheet_name"],
        "field_column_ids": {
            "target_table": column_ids["Target Table"],
            "target_field": column_ids["Target Column"],
            "source_table": column_ids["Source Table"],
            "source_field": column_ids["Source Column"],
            "transformation_rule": column_ids["SQL Transform"],
        },
        "evidence": {
            **_evidence("target_table", column_ids["Target Table"]),
            **_evidence("target_field", column_ids["Target Column"]),
            **_evidence("source_table", column_ids["Source Table"]),
            **_evidence("source_field", column_ids["Source Column"]),
            **_evidence("transformation_rule", column_ids["SQL Transform"]),
        },
    }

    result = write_s2t_transformations_from_plan(file_id, [mapping])

    assert result["count"] == 3
    assert result["empty_target_columns_count"] == 2
    assert result["sheet_results"] == [
        {
            "sheet_name": "S2T",
            "rows_written": 3,
            "empty_target_columns_count": 2,
        }
    ]
    rows = verify_s2t_transformations(file_id, limit=10)["rows"]
    assert [row["row_num"] for row in rows] == [0, 1, 2]
    assert [row["target_field"] for row in rows] == ["c1", None, None]
    assert rows[1] == rows[2] | {"row_num": 1}


def test_s2t_extraction_appends_without_deleting_stored_rows(s2t_db):
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
        (id, file_id, sheet_name, row_num, target_table, target_field)
        VALUES (999, ?, 'S2T', 99, 'old', 'old')
        """,
        (file_id,),
    )
    conn.execute(
        """
        INSERT INTO s2t_transformations
        (id, file_id, sheet_name, row_num, target_table, target_field)
        VALUES (1000, 777, 'S2T', 7, 'foreign', 'foreign')
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
        report = run_s2t_extraction_subagent(file_id)

    assert report["written"] == 3
    assert report["verification"]["count"] == 4
    rows = verify_s2t_transformations(file_id, limit=10)["rows"]
    assert rows == [
        {
            "row_num": 0,
            "target_table": "t1",
            "target_field": "c1",
            "source_table": "src",
            "source_field": "src_c",
            "transformation_rule": "copy",
            "source_layer": "B",
            "target_layer": "T",
        },
        {
            "row_num": 1,
            "target_table": "t1",
            "target_field": "c1",
            "source_table": "src",
            "source_field": "src_c",
            "transformation_rule": "copy",
            "source_layer": "B",
            "target_layer": "T",
        },
        {
            "row_num": 2,
            "target_table": "t1",
            "target_field": "c2",
            "source_table": "src",
            "source_field": "src_c2",
            "transformation_rule": "copy",
            "source_layer": "B",
            "target_layer": "T",
        },
        {
            "row_num": 99,
            "target_table": "old",
            "target_field": "old",
            "source_table": None,
            "source_field": None,
            "transformation_rule": None,
            "source_layer": None,
            "target_layer": None,
        },
    ]

    conn = get_db_connection()
    foreign_rows = conn.execute(
        "SELECT COUNT(*) AS n FROM s2t_transformations WHERE file_id = 777"
    ).fetchone()["n"]
    conn.close()
    assert foreign_rows == 1


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
    conn = get_db_connection()
    conn.execute(
        """
        INSERT INTO s2t_transformations
        (file_id, sheet_name, row_num, target_table, target_field)
        VALUES (777, 'existing', 7, 'existing_target', 'existing_field')
        """
    )
    conn.commit()
    conn.close()

    analysis = classify_file_sheet_groups(file_id, use_llm=False)
    report = extract_table_catalogs(file_id, analysis)

    assert report["status"] == "ok"
    assert report["targets"]["source_tables"]["count"] == 2
    assert report["targets"]["target_tables"]["count"] == 2

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
    existing_s2t_rows = conn.execute(
        "SELECT COUNT(*) AS n FROM s2t_transformations WHERE file_id = 777"
    ).fetchone()["n"]
    conn.close()

    assert existing_s2t_rows == 1
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


def test_column_catalog_uses_dedicated_values_and_raw_s2t_fallback(s2t_db):
    file_id = store_excel_data(
        "column_catalogs.xlsx",
        "model",
        [
            {
                "sheet_name": "S2T",
                "skip_reason": None,
                "header": {"start_row": 0, "row_count": 1, "nested": False},
                "columns": [
                    "Target Table",
                    "Target Column",
                    "Data Type",
                    "not null",
                    "Primary Key",
                    "Description",
                    "Source Table",
                    "Source Column",
                    "Data Type",
                    "not null",
                    "Primary Key",
                    "Description",
                    "SQL Transform",
                ],
                "data_rows": [
                    [
                        "mart.customer",
                        "customer_id",
                        "bigint",
                        "yes",
                        "yes",
                        "Target identifier",
                        "raw.customer",
                        "id",
                        "bigint",
                        "yes",
                        "yes",
                        "Raw identifier",
                        "copy",
                    ],
                    [
                        "mart.customer",
                        "customer_name",
                        "text",
                        "no",
                        "no",
                        "Target name",
                        "raw.customer",
                        "name",
                        "varchar(100)",
                        "no",
                        "no",
                        "Raw name",
                        "trim(name)",
                    ],
                ],
            },
            {
                "sheet_name": "Source columns",
                "skip_reason": None,
                "header": {"start_row": 0, "row_count": 1, "nested": False},
                "columns": ["table", "column", "datatype", "notnull", "pk", "comment"],
                "data_rows": [
                    [
                        "raw.customer",
                        "id",
                        "numeric(20)",
                        "yes",
                        "yes",
                        "Dedicated identifier",
                    ]
                ],
            },
        ],
    )
    analysis = classify_file_sheet_groups(file_id, use_llm=False)
    s2t_report = run_s2t_extraction_subagent(
        file_id, sheet_group_analysis=analysis
    )

    report = extract_column_catalogs(
        file_id,
        analysis,
        s2t_sheet_mappings=s2t_report["sheet_mappings"],
    )

    assert report["status"] == "ok"
    assert report["count"] == 4
    assert report["embedded_description_count"] == 4
    assert report["targets"]["target_columns"]["sheet_count"] == 0
    assert report["targets"]["target_columns"]["source_counts"] == {
        "columns_sheet": 0,
        "s2t": 2,
        "mixed": 0,
    }
    assert report["targets"]["source_columns"]["source_counts"] == {
        "columns_sheet": 0,
        "s2t": 1,
        "mixed": 1,
    }
    assert any(
        conflict["field"] == "data_type"
        for conflict in report["targets"]["source_columns"]["conflicts"]
    )

    conn = get_db_connection()
    try:
        source_rows = [
            dict(row)
            for row in conn.execute(
                """
                SELECT table_name, column_name, data_type, primary_key,
                       not_null, description, description_embedding
                FROM source_columns
                WHERE file_id = ?
                ORDER BY column_name
                """,
                (file_id,),
            ).fetchall()
        ]
        target_rows = [
            dict(row)
            for row in conn.execute(
                """
                SELECT table_name, column_name, data_type, primary_key,
                       not_null, description, description_embedding
                FROM target_columns
                WHERE file_id = ?
                ORDER BY column_name
                """,
                (file_id,),
            ).fetchall()
        ]
    finally:
        conn.close()

    assert source_rows == [
        {
            "table_name": "raw.customer",
            "column_name": "id",
            "data_type": "numeric(20)",
            "primary_key": 1,
            "not_null": 1,
            "description": "Dedicated identifier",
            "description_embedding": (
                "embedding:Название колонки: id\nОписание: Dedicated identifier".encode(
                    "utf-8"
                )
            ),
        },
        {
            "table_name": "raw.customer",
            "column_name": "name",
            "data_type": "varchar(100)",
            "primary_key": 0,
            "not_null": 0,
            "description": "Raw name",
            "description_embedding": (
                "embedding:Название колонки: name\nОписание: Raw name".encode(
                    "utf-8"
                )
            ),
        },
    ]
    assert target_rows == [
        {
            "table_name": "mart.customer",
            "column_name": "customer_id",
            "data_type": "bigint",
            "primary_key": 1,
            "not_null": 1,
            "description": "Target identifier",
            "description_embedding": (
                "embedding:Название колонки: customer_id\n"
                "Описание: Target identifier"
            ).encode(
                "utf-8"
            ),
        },
        {
            "table_name": "mart.customer",
            "column_name": "customer_name",
            "data_type": "text",
            "primary_key": 0,
            "not_null": 0,
            "description": "Target name",
            "description_embedding": (
                "embedding:Название колонки: customer_name\nОписание: Target name"
            ).encode(
                "utf-8"
            ),
        },
    ]


def test_column_catalog_recovers_headers_from_first_stored_data_row(s2t_db):
    file_id = store_excel_data(
        "recovered_column_headers.xlsx",
        "model",
        [
            {
                "sheet_name": "S2T",
                "skip_reason": None,
                "header": {"start_row": 0, "row_count": 1, "nested": False},
                "columns": [
                    "Target Table",
                    "Target Column",
                    "Source Table",
                    "Source Column",
                    "SQL Transform",
                ],
                "data_rows": [
                    ["mart.orders", "order_id", "raw.orders", "id", "copy"]
                ],
            },
            {
                "sheet_name": "Source columns",
                "skip_reason": None,
                "header": {"start_row": 0, "row_count": 1, "nested": False},
                "columns": ["Column_1", "Column_2", "Column_3", "4", "5", "6"],
                "data_rows": [
                    ["Название таблицы", "Название поля", "Тип поля", "NULL?", "PK", "Описание поля"],
                    ["raw.orders", "id", "uuid", "нет", "да", "Идентификатор заказа"],
                    ["raw.orders", "payload", "jsonb", None, None, "Тело заказа"],
                ],
            },
        ],
    )
    analysis = classify_file_sheet_groups(file_id, use_llm=False)
    s2t_report = run_s2t_extraction_subagent(
        file_id, sheet_group_analysis=analysis
    )

    report = extract_column_catalogs(
        file_id,
        analysis,
        s2t_sheet_mappings=s2t_report["sheet_mappings"],
    )

    source_report = report["targets"]["source_columns"]
    assert source_report["sheet_mappings"][0]["header_recovered"] is True
    assert source_report["sheet_mappings"][0]["header_row_num"] == 0
    conn = get_db_connection()
    try:
        row = dict(
            conn.execute(
                """
                SELECT table_name, column_name, data_type, primary_key,
                       not_null, description
                FROM source_columns
                WHERE file_id = ? AND column_name = 'id'
                """,
                (file_id,),
            ).fetchone()
        )
        blank_flags = tuple(
            conn.execute(
                """
                SELECT primary_key, not_null
                FROM source_columns
                WHERE file_id = ? AND column_name = 'payload'
                """,
                (file_id,),
            ).fetchone()
        )
    finally:
        conn.close()
    assert row == {
        "table_name": "raw.orders",
        "column_name": "id",
        "data_type": "uuid",
        "primary_key": 1,
        "not_null": 1,
        "description": "Идентификатор заказа",
    }
    assert blank_flags == (0, None)


def test_column_catalog_uses_one_llm_check_for_unmapped_header_and_saves_alias(
    s2t_db,
):
    file_id = store_excel_data(
        "column_catalog_llm.xlsx",
        "model",
        [
            {
                "sheet_name": "S2T",
                "skip_reason": None,
                "header": {"start_row": 0, "row_count": 1, "nested": False},
                "columns": [
                    "Target Table",
                    "Target Column",
                    "Source Table",
                    "Source Column",
                    "SQL Transform",
                ],
                "data_rows": [
                    ["mart.orders", "order_id", "raw.orders", "id", "copy"]
                ],
            },
            {
                "sheet_name": "Source columns",
                "skip_reason": None,
                "header": {"start_row": 0, "row_count": 1, "nested": False},
                "columns": [
                    "table",
                    "column",
                    "datatype",
                    "notnull",
                    "pk",
                    "Business label",
                ],
                "data_rows": [
                    [
                        "raw.orders",
                        "id",
                        "uuid",
                        "yes",
                        "yes",
                        "Идентификатор заказа",
                    ]
                ],
            },
        ],
    )
    analysis = classify_file_sheet_groups(file_id, use_llm=False)
    s2t_report = run_s2t_extraction_subagent(
        file_id, sheet_group_analysis=analysis
    )
    response = {
        "sheet_name": "Source columns",
        "column_roles": [
            {"column_name": "table", "mapping_field": "table_name"},
            {"column_name": "column", "mapping_field": "column_name"},
            {"column_name": "datatype", "mapping_field": "data_type"},
            {"column_name": "notnull", "mapping_field": "not_null"},
            {"column_name": "pk", "mapping_field": "primary_key"},
            {"column_name": "Business label", "mapping_field": "description"},
        ],
    }

    with patch(
        "sheet_skills.s2t._invoke_llm_plain_text",
        return_value=json.dumps(response, ensure_ascii=False),
    ) as mock_llm:
        report = extract_column_catalogs(
            file_id,
            analysis,
            s2t_sheet_mappings=s2t_report["sheet_mappings"],
        )

    mock_llm.assert_called_once()
    prompt = mock_llm.call_args.args[0]
    assert '"source_columns"' in prompt
    assert "Business label" in prompt
    assert "Идентификатор заказа" in prompt
    assert "side_hint" not in prompt
    assert "previous_column" not in prompt
    assert "column_id" not in prompt
    source_report = report["targets"]["source_columns"]
    assert source_report["attempts"] == 1
    assert source_report["aliases_added"] == 1
    assert source_report["sheet_mappings"][0]["method"] == "llm"
    assert source_report["sheet_mappings"][0]["attempts"] == 1
    assert "Business label" in column_mapping_config.get_field_aliases(
        "source_columns", "description"
    )

    conn = get_db_connection()
    try:
        description = conn.execute(
            """
            SELECT description
            FROM source_columns
            WHERE file_id = ? AND table_name = 'raw.orders' AND column_name = 'id'
            """,
            (file_id,),
        ).fetchone()[0]
    finally:
        conn.close()
    assert description == "Идентификатор заказа"


def test_column_catalog_rejects_invalid_llm_mapping_without_partial_write(s2t_db):
    file_id = store_excel_data(
        "column_catalog_bad_llm.xlsx",
        "model",
        [
            {
                "sheet_name": "S2T",
                "skip_reason": None,
                "header": {"start_row": 0, "row_count": 1, "nested": False},
                "columns": [
                    "Target Table",
                    "Target Column",
                    "Source Table",
                    "Source Column",
                    "SQL Transform",
                ],
                "data_rows": [
                    ["mart.orders", "order_id", "raw.orders", "id", "copy"]
                ],
            },
            {
                "sheet_name": "Source columns",
                "skip_reason": None,
                "header": {"start_row": 0, "row_count": 1, "nested": False},
                "columns": [
                    "table",
                    "column",
                    "datatype",
                    "notnull",
                    "pk",
                    "Unknown description header",
                ],
                "data_rows": [
                    ["raw.orders", "id", "uuid", "yes", "yes", "Identifier"]
                ],
            },
        ],
    )
    analysis = classify_file_sheet_groups(file_id, use_llm=False)
    s2t_report = run_s2t_extraction_subagent(
        file_id, sheet_group_analysis=analysis
    )

    with patch(
        "sheet_skills.s2t._invoke_llm_plain_text", return_value="not-json"
    ) as mock_llm:
        with pytest.raises(ColumnCatalogExtractionError, match="Source columns"):
            extract_column_catalogs(
                file_id,
                analysis,
                s2t_sheet_mappings=s2t_report["sheet_mappings"],
            )

    mock_llm.assert_called_once()
    conn = get_db_connection()
    try:
        count = conn.execute(
            "SELECT COUNT(*) FROM source_columns WHERE file_id = ?",
            (file_id,),
        ).fetchone()[0]
    finally:
        conn.close()
    assert count == 0


def test_column_catalog_resolves_missing_sheet_table_from_unique_s2t_pair(s2t_db):
    file_id = store_excel_data(
        "missing_column_table.xlsx",
        "model",
        [
            {
                "sheet_name": "S2T",
                "skip_reason": None,
                "header": {"start_row": 0, "row_count": 1, "nested": False},
                "columns": [
                    "Target Table",
                    "Target Column",
                    "Source Table",
                    "Source Column",
                    "SQL Transform",
                ],
                "data_rows": [
                    ["mart.orders", "order_id", "raw.orders", "id", "copy"],
                    ["mart.orders", "payload_id", "raw.payload", "payload", "copy"],
                ],
            },
            {
                "sheet_name": "Source columns",
                "skip_reason": None,
                "header": {"start_row": 0, "row_count": 1, "nested": False},
                "columns": [
                    "table",
                    "column",
                    "datatype",
                    "notnull",
                    "pk",
                    "comment",
                ],
                "data_rows": [
                    ["raw.orders", "id", "uuid", "yes", "yes", "Ключ"],
                    [None, "payload", "jsonb", "no", "no", "Тело"],
                ],
            },
        ],
    )
    analysis = classify_file_sheet_groups(file_id, use_llm=False)
    s2t_report = run_s2t_extraction_subagent(
        file_id, sheet_group_analysis=analysis
    )

    report = extract_column_catalogs(
        file_id,
        analysis,
        s2t_sheet_mappings=s2t_report["sheet_mappings"],
    )

    source_report = report["targets"]["source_columns"]
    assert source_report["missing_table_rows"] == [
        {
            "sheet_name": "Source columns",
            "row_num": 1,
            "column_name": "payload",
        }
    ]
    assert source_report["resolved_missing_tables"] == [
        {
            "sheet_name": "Source columns",
            "row_num": 1,
            "column_name": "payload",
            "table_name": "raw.payload",
        }
    ]
    conn = get_db_connection()
    try:
        rows = [
            tuple(row)
            for row in conn.execute(
                """
                SELECT table_name, column_name, description
                FROM source_columns WHERE file_id = ?
                ORDER BY table_name, column_name
                """,
                (file_id,),
            ).fetchall()
        ]
    finally:
        conn.close()
    assert ("raw.payload", "payload", "Тело") in rows
    assert not any(
        table_name == "raw.orders" and column_name == "payload"
        for table_name, column_name, _ in rows
    )


@pytest.mark.parametrize(
    ("value", "header", "expected"),
    [
        (None, "Not Null", (None, False)),
        ("", "NULL?", (None, False)),
        ("Yes", "NULL?", (0, False)),
        ("No", "NULL?", (1, False)),
        ("Да", "Допускается пустое значение?", (0, False)),
        ("Нет", "Допускается пустое значение?", (1, False)),
        ("True", "Обязательное поле", (1, False)),
        ("False", "Обязательное поле", (0, False)),
        ("Истина", "Not Null", (1, False)),
        ("Ложь", "Not Null", (0, False)),
        ("Истина", "NULL?", (0, False)),
        ("Ложь", "NULL?", (1, False)),
        ("mandatory", "NULL?", (1, False)),
        ("optional", "Not Null", (0, False)),
        ("обязательно", "NULL?", (1, False)),
        ("необязательно", "Not Null", (0, False)),
        ("неизвестно", "Not Null", (None, True)),
    ],
)
def test_column_catalog_normalises_nullable_and_not_null_values(
    value, header, expected
):
    assert _normalise_flag(value, header) == expected


@pytest.mark.parametrize(
    ("header", "expected"),
    [
        ("NULL?", True),
        ("Column Null Option", True),
        ("Nullable", True),
        ("Допускается NULL", True),
        ("Допускается пустое значение?", True),
        ("Not Null", False),
        ("notnullsource", False),
        ("Required", False),
        ("Обязательное поле", False),
        ("NULL не допускается", False),
    ],
)
def test_column_catalog_detects_header_polarity(header, expected):
    assert _is_null_question(header) is expected


def test_column_catalog_unknown_not_null_does_not_override_known_value():
    rows, report = _merge_catalog(
        "target_columns",
        [],
        [
            {
                "file_id": 1,
                "sheet_name": "Target columns",
                "row_num": 770,
                "table_name": "T_PROD",
                "column_name": "START_DT",
                "data_type": None,
                "primary_key": None,
                "not_null": None,
                "description": "Дата начала",
            },
            {
                "file_id": 1,
                "sheet_name": "Target columns",
                "row_num": 1136,
                "table_name": "T_PROD",
                "column_name": "START_DT",
                "data_type": None,
                "primary_key": None,
                "not_null": 1,
                "description": None,
            },
        ],
    )

    assert len(rows) == 1
    assert rows[0]["not_null"] == 1
    assert not [
        conflict
        for conflict in report["conflicts"]
        if conflict["field"] == "not_null"
    ]


def test_column_catalog_reports_conflicting_repeated_descriptions():
    common = {
        "file_id": 1,
        "sheet_name": "Source columns",
        "table_name": "b3080000460006_payments_2_0_posting_dto",
        "column_name": "codecommission_uid",
        "data_type": "uuid",
        "primary_key": 0,
        "not_null": 0,
    }
    rows, report = _merge_catalog(
        "source_columns",
        [],
        [
            {
                **common,
                "row_num": 132,
                "description": "Код типа комиссии",
            },
            {
                **common,
                "row_num": 455,
                "description": "SK(codecommission)",
            },
        ],
    )

    assert len(rows) == 1
    assert rows[0]["description"] == "Код типа комиссии"
    assert report["conflicts"] == [
        {
            "table_name": "b3080000460006_payments_2_0_posting_dto",
            "column_name": "codecommission_uid",
            "field": "description",
            "selected_source": "columns_sheet",
            "selected_value": "Код типа комиссии",
            "other_value": "SK(codecommission)",
        }
    ]


def test_column_catalog_uses_s2t_not_null_when_columns_sheet_value_is_empty():
    rows, report = _merge_catalog(
        "target_columns",
        [
            {
                "file_id": 1,
                "sheet_name": "S2T",
                "row_num": 10,
                "table_name": "mart.orders",
                "column_name": "order_id",
                "data_type": None,
                "primary_key": None,
                "not_null": 1,
                "description": None,
            }
        ],
        [
            {
                "file_id": 1,
                "sheet_name": "Target columns",
                "row_num": 20,
                "table_name": "mart.orders",
                "column_name": "order_id",
                "data_type": "uuid",
                "primary_key": None,
                "not_null": None,
                "description": "Идентификатор заказа",
            }
        ],
    )

    assert rows[0]["not_null"] == 1
    assert report["source_counts"]["mixed"] == 1
    assert report["conflicts"] == []


def test_column_catalog_does_not_guess_missing_table_with_ambiguous_s2t_pairs():
    s2t_rows = [
        {
            "file_id": 1,
            "sheet_name": "S2T",
            "row_num": row_num,
            "table_name": table_name,
            "column_name": "id",
            "description": None,
        }
        for row_num, table_name in ((1, "raw.orders"), (2, "raw.clients"))
    ]
    sheet_row = {
        "file_id": 1,
        "sheet_name": "Source columns",
        "row_num": 10,
        "table_name": None,
        "column_name": "id",
        "description": "Идентификатор",
    }

    rows, report = _merge_catalog("source_columns", s2t_rows, [sheet_row])

    assert len(rows) == 3
    assert any(row["table_name"] is None for row in rows)
    assert report["resolved_missing_tables"] == []
    assert report["ambiguous_missing_tables"] == [
        {
            "sheet_name": "Source columns",
            "row_num": 10,
            "column_name": "id",
            "candidate_tables": ["raw.clients", "raw.orders"],
        }
    ]


def test_column_catalog_reports_explicit_not_null_conflict():
    rows, report = _merge_catalog(
        "source_columns",
        [],
        [
            {
                "file_id": 1,
                "sheet_name": "Source columns",
                "row_num": 1,
                "table_name": "raw.orders",
                "column_name": "id",
                "not_null": 0,
            },
            {
                "file_id": 1,
                "sheet_name": "Source columns",
                "row_num": 2,
                "table_name": "raw.orders",
                "column_name": "id",
                "not_null": 1,
            },
        ],
    )

    assert rows[0]["not_null"] == 0
    assert report["conflicts"] == [
        {
            "table_name": "raw.orders",
            "column_name": "id",
            "field": "not_null",
            "selected_source": "columns_sheet",
            "selected_value": 0,
            "other_value": 1,
        }
    ]


def test_column_catalog_preserves_sheet_column_without_s2t_link(s2t_db):
    rows, report = _merge_catalog(
        "source_columns",
        [],
        [
            {
                "file_id": 1,
                "sheet_name": "Source columns",
                "row_num": 7,
                "table_name": "raw.unused",
                "column_name": "payload",
                "data_type": "jsonb",
                "primary_key": 0,
                "not_null": 0,
                "description": "Дополнительная колонка",
            }
        ],
    )

    assert len(rows) == 1
    assert report["source_counts"]["columns_sheet"] == 1
    assert report["unlinked_rows"] == [
        {
            "sheet_name": "Source columns",
            "row_num": 7,
            "table_name": "raw.unused",
            "column_name": "payload",
        }
    ]


def test_backfill_column_description_embeddings_uses_names_and_descriptions(s2t_db):
    conn = get_db_connection()
    conn.executemany(
        """
        INSERT INTO source_columns
        (file_id, sheet_name, row_num, table_name, column_name,
         description)
        VALUES (?, 'Source columns', ?, 'raw.orders', ?, ?)
        """,
        [
            (10, 0, "id", "Идентификатор заказа"),
            (10, 1, "payload", None),
            (20, 0, "other_id", "Другой идентификатор"),
        ],
    )
    conn.execute(
        """
        INSERT INTO target_columns
        (file_id, sheet_name, row_num, table_name, column_name,
         description)
        VALUES (10, 'Target columns', 0, 'mart.orders', 'order_id',
                'Ключ витрины')
        """
    )
    conn.commit()
    conn.close()

    assert backfill_column_description_embeddings(10) == {
        "file_id": 10,
        "candidates": 3,
        "updated": 3,
    }

    conn = get_db_connection()
    rows = [
        tuple(row)
        for row in conn.execute(
            """
            SELECT file_id, column_name, description_embedding
            FROM source_columns
            UNION ALL
            SELECT file_id, column_name, description_embedding
            FROM target_columns
            ORDER BY file_id, column_name
            """
        ).fetchall()
    ]
    conn.close()
    assert rows == [
        (
            10,
            "id",
            "embedding:Название колонки: id\nОписание: Идентификатор заказа".encode(
                "utf-8"
            ),
        ),
        (
            10,
            "order_id",
            "embedding:Название колонки: order_id\nОписание: Ключ витрины".encode(
                "utf-8"
            ),
        ),
        (10, "payload", "embedding:Название колонки: payload".encode("utf-8")),
        (20, "other_id", None),
    ]


def test_structured_metadata_extraction_writes_configured_fields_and_duplicates(
    s2t_db,
):
    file_id = _store_structured_metadata()

    analysis = classify_file_sheet_groups(file_id, use_llm=False)
    report = extract_structured_metadata(file_id, analysis)

    metadata_report = report["targets"]
    assert metadata_report["additional_objects"]["count"] == 2
    assert metadata_report["pxf_to_a"]["count"] == 1
    etl_report = metadata_report["additional_objects"]["etl_transformations"]
    assert etl_report["dialect"] == "greenplum"
    assert etl_report["object_count"] == 2
    assert etl_report["written"] == 2

    conn = get_db_connection()
    try:
        additional_rows = [
            tuple(row)
            for row in conn.execute(
                """
                SELECT row_num, name, sql
                FROM additional_objects
                WHERE file_id = ?
                ORDER BY row_num, id
                """,
                (file_id,),
            ).fetchall()
        ]
        pxf_rows = [
            tuple(row)
            for row in conn.execute(
                """
                SELECT row_num, external_a_table, materialized_storage,
                       replica_table, sod
                FROM pxf_to_a
                WHERE file_id = ?
                ORDER BY row_num, id
                """,
                (file_id,),
            ).fetchall()
        ]
        etl_rows = [
            tuple(row)
            for row in conn.execute(
                """
                SELECT target_table, target_field, source_table, source_field,
                       transformation_rule, source_layer, target_layer
                FROM s2t_transformations
                WHERE file_id = ?
                ORDER BY id
                """,
                (file_id,),
            ).fetchall()
        ]
    finally:
        conn.close()

    assert additional_rows == [
        (0, "view_orders", "SELECT * FROM raw.orders"),
        (1, "view_orders", "SELECT * FROM raw.orders"),
    ]
    assert pxf_rows == [
        (0, "ext_orders", "mat_orders", "replica_orders", "SOD-1")
    ]
    assert etl_rows == [
        ("view_orders", "*", "raw.orders", "*", "*", None, "B"),
        ("view_orders", "*", "raw.orders", "*", "*", None, "B"),
    ]


def test_additional_objects_parse_greenplum_comments_and_keep_valid_objects(s2t_db):
    file_id = store_excel_data(
        "additional_lineage.xlsx",
        "model",
        [
            {
                "sheet_name": "Additional objects",
                "skip_reason": None,
                "header": {"start_row": 0, "row_count": 1, "nested": False},
                "columns": ["name", "SQL"],
                "data_rows": [
                    [
                        "b_orders",
                        "DROP VIEW IF EXISTS mart.b_orders; "
                        "CREATE VIEW mart.b_orders AS "
                        "SELECT o.id, -- id comment\n"
                        "UPPER(o.name) AS normalized_name, "
                        "1::bigint AS constant FROM raw.orders o",
                    ],
                    ["broken_object", "SELECT ("],
                ],
            }
        ],
    )

    analysis = classify_file_sheet_groups(file_id, use_llm=False)
    with patch(
        "sheet_skills.additional_objects._repair_sql_with_llm",
        return_value="SELECT (",
    ) as repair_sql:
        report = extract_structured_metadata(file_id, analysis)
    etl_report = report["targets"]["additional_objects"]["etl_transformations"]

    assert etl_report["status"] == "partial"
    assert etl_report["object_count"] == 2
    assert etl_report["parsed_object_count"] == 1
    assert etl_report["error_count"] == 1
    assert etl_report["repair_attempt_count"] == 1
    assert etl_report["repaired_object_count"] == 0
    assert etl_report["repair_error_count"] == 1
    assert etl_report["written"] == 3
    assert etl_report["objects"][0]["select_count"] == 1
    assert etl_report["objects"][0]["statement_count"] == 2
    broken_report = etl_report["objects"][1]
    assert broken_report["llm_repair_attempted"] is True
    assert broken_report["llm_repair_status"] == "retry_parse_error"
    assert broken_report["initial_parse_error"]
    assert broken_report["retry_parse_error"]
    repair_sql.assert_called_once()

    conn = get_db_connection()
    try:
        rows = [
            dict(row)
            for row in conn.execute(
                """
                SELECT target_table, target_field, source_table, source_field,
                       transformation_rule, source_layer, target_layer
                FROM s2t_transformations
                WHERE file_id = ?
                ORDER BY id
                """,
                (file_id,),
            ).fetchall()
        ]
    finally:
        conn.close()

    assert rows == [
        {
            "target_table": "mart.b_orders",
            "target_field": "id",
            "source_table": "raw.orders",
            "source_field": "id",
            "transformation_rule": "o.id /* id comment */",
            "source_layer": None,
            "target_layer": "B",
        },
        {
            "target_table": "mart.b_orders",
            "target_field": "normalized_name",
            "source_table": "raw.orders",
            "source_field": "name",
            "transformation_rule": "UPPER(o.name)",
            "source_layer": None,
            "target_layer": "B",
        },
        {
            "target_table": "mart.b_orders",
            "target_field": "constant",
            "source_table": None,
            "source_field": None,
            "transformation_rule": "CAST(1 AS BIGINT)",
            "source_layer": None,
            "target_layer": "B",
        },
    ]


def test_additional_objects_repair_invalid_sql_and_retry_sqlglot(s2t_db):
    original_sql = "SELECT ( -- незавершённое выражение"
    repaired_sql = "SELECT o.id FROM raw.orders AS o -- исходные заказы"
    file_id = store_excel_data(
        "additional_repair.xlsx",
        "model",
        [
            {
                "sheet_name": "Additional objects",
                "skip_reason": None,
                "header": {"start_row": 0, "row_count": 1, "nested": False},
                "columns": ["name", "SQL"],
                "data_rows": [["mart.repaired", original_sql]],
            }
        ],
    )

    analysis = classify_file_sheet_groups(file_id, use_llm=False)
    with patch(
        "sheet_skills.additional_objects._repair_sql_with_llm",
        return_value=repaired_sql,
    ) as repair_sql:
        report = extract_structured_metadata(file_id, analysis)

    etl_report = report["targets"]["additional_objects"]["etl_transformations"]
    object_report = etl_report["objects"][0]
    assert etl_report["status"] == "ok"
    assert etl_report["repair_attempt_count"] == 1
    assert etl_report["repaired_object_count"] == 1
    assert etl_report["repair_error_count"] == 0
    assert etl_report["written"] == 1
    assert object_report["status"] == "ok"
    assert object_report["llm_repair_attempted"] is True
    assert object_report["llm_repair_status"] == "success"
    assert object_report["initial_parse_error"]
    assert object_report["retry_parse_error"] is None
    repair_sql.assert_called_once()
    assert repair_sql.call_args.args[0] == original_sql
    assert repair_sql.call_args.args[1] == object_report["initial_parse_error"]

    conn = get_db_connection()
    try:
        stored_sql = conn.execute(
            "SELECT sql FROM additional_objects WHERE file_id = ?",
            (file_id,),
        ).fetchone()["sql"]
        raw_sql = conn.execute(
            """
            SELECT value
            FROM data
            WHERE file_id = ? AND table_name = ? AND value = ?
            """,
            (file_id, "Additional objects", original_sql),
        ).fetchone()
        lineage_row = dict(
            conn.execute(
                """
                SELECT target_table, target_field, source_table, source_field,
                       transformation_rule, source_layer, target_layer
                FROM s2t_transformations
                WHERE file_id = ?
                """,
                (file_id,),
            ).fetchone()
        )
    finally:
        conn.close()

    assert stored_sql == original_sql
    assert raw_sql["value"] == original_sql
    assert lineage_row == {
        "target_table": "mart.repaired",
        "target_field": "id",
        "source_table": "raw.orders",
        "source_field": "id",
        "transformation_rule": "o.id",
        "source_layer": None,
        "target_layer": "B",
    }


def test_additional_objects_preserve_cte_scope_lineage(s2t_db):
    rows, report = _parse_object(
        {
            "id": 1,
            "file_id": 1,
            "sheet_name": "Additional objects",
            "row_num": 7,
            "name": "mart.result",
            "sql": """
                WITH src AS (
                    SELECT o.id, o.name FROM raw.orders o
                ), norm AS (
                    SELECT id, UPPER(name) AS name FROM src
                )
                SELECT id, name FROM norm
            """,
        }
    )

    assert report["select_count"] == 3
    assert report["scope_count"] == 3
    assert report["intermediate_scope_count"] == 2
    assert report["written"] == 6
    assert {
        (
            row["source_table"],
            row["source_field"],
            row["target_table"],
            row["target_field"],
        )
        for row in rows
    } == {
        ("raw.orders", "id", "mart.result::cte::src", "id"),
        ("raw.orders", "name", "mart.result::cte::src", "name"),
        (
            "mart.result::cte::src",
            "id",
            "mart.result::cte::norm",
            "id",
        ),
        (
            "mart.result::cte::src",
            "name",
            "mart.result::cte::norm",
            "name",
        ),
        ("mart.result::cte::norm", "id", "mart.result", "id"),
        ("mart.result::cte::norm", "name", "mart.result", "name"),
    }
    norm_name = next(
        row
        for row in rows
        if row["target_table"] == "mart.result::cte::norm"
        and row["target_field"] == "name"
    )
    assert norm_name["transformation_rule"] == "UPPER(src.name)"
    assert all(row["source_layer"] is None for row in rows)
    assert all(
        row["target_layer"]
        == ("B" if row["target_table"] == "mart.result" else None)
        for row in rows
    )


def test_additional_objects_keep_wildcard_between_scopes(s2t_db):
    rows, report = _parse_object(
        {
            "id": 1,
            "file_id": 1,
            "sheet_name": "Additional objects",
            "row_num": 8,
            "name": "mart.wild",
            "sql": """
                WITH src AS (SELECT * FROM raw.orders),
                     pass AS (SELECT * FROM src)
                SELECT * FROM pass
            """,
        }
    )

    assert report["written"] == 3
    assert [
        (row["source_table"], row["target_table"])
        for row in rows
    ] == [
        ("raw.orders", "mart.wild::cte::src"),
        ("mart.wild::cte::src", "mart.wild::cte::pass"),
        ("mart.wild::cte::pass", "mart.wild"),
    ]
    assert all(row["source_field"] == "*" for row in rows)
    assert all(row["target_field"] == "*" for row in rows)


def test_additional_objects_name_union_scopes_and_insert_columns(s2t_db):
    union_rows, union_report = _parse_object(
        {
            "id": 1,
            "file_id": 1,
            "sheet_name": "Additional objects",
            "row_num": 9,
            "name": "mart.union_result",
            "sql": "SELECT id FROM raw.a UNION ALL SELECT id FROM raw.b",
        }
    )
    insert_rows, _insert_report = _parse_object(
        {
            "id": 2,
            "file_id": 1,
            "sheet_name": "Additional objects",
            "row_num": 10,
            "name": "fallback_name",
            "sql": (
                "INSERT INTO mart.target (target_id) "
                "SELECT id FROM raw.a"
            ),
        }
    )

    assert union_report["select_count"] == 2
    assert union_report["scope_count"] == 3
    assert {
        (row["source_table"], row["target_table"])
        for row in union_rows
    } == {
        ("raw.a", "mart.union_result::branch::1"),
        ("raw.b", "mart.union_result::branch::2"),
        ("mart.union_result::branch::1", "mart.union_result"),
        ("mart.union_result::branch::2", "mart.union_result"),
    }
    assert insert_rows[0]["target_table"] == "mart.target"
    assert insert_rows[0]["target_field"] == "target_id"


def test_additional_objects_preserve_sqlglot_set_operation_tree(s2t_db):
    rows, report = _parse_object(
        {
            "id": 1,
            "file_id": 1,
            "sheet_name": "Additional objects",
            "row_num": 11,
            "name": "mart.mixed_union",
            "sql": (
                "SELECT id FROM raw.a "
                "UNION ALL SELECT id FROM raw.b "
                "UNION SELECT id FROM raw.c "
                "UNION ALL SELECT id FROM raw.d"
            ),
        }
    )

    assert report["sqlglot_scope_count"] == 7
    assert report["scope_count"] == report["sqlglot_scope_count"]
    set_rows = [row for row in rows if row["transformation_rule"].startswith("UNION")]
    assert len(set_rows) == 6
    assert sum(row["transformation_rule"] == "UNION ALL" for row in set_rows) == 4
    assert sum(row["transformation_rule"] == "UNION" for row in set_rows) == 2
    assert {
        (row["source_table"], row["target_table"], row["transformation_rule"])
        for row in set_rows
        if row["source_table"] in {
            "mart.mixed_union::union::1",
            "mart.mixed_union::union::2",
        }
    } == {
        (
            "mart.mixed_union::union::1",
            "mart.mixed_union::union::2",
            "UNION",
        ),
        (
            "mart.mixed_union::union::2",
            "mart.mixed_union",
            "UNION ALL",
        ),
    }


def test_additional_objects_preserve_sqlglot_except_tree(s2t_db):
    rows, report = _parse_object(
        {
            "id": 1,
            "file_id": 1,
            "sheet_name": "Additional objects",
            "row_num": 12,
            "name": "mart.except_result",
            "sql": (
                "SELECT id FROM raw.a "
                "EXCEPT SELECT id FROM raw.b "
                "EXCEPT SELECT id FROM raw.c"
            ),
        }
    )

    assert report["sqlglot_scope_count"] == 5
    assert report["scope_count"] == report["sqlglot_scope_count"]
    set_rows = [row for row in rows if row["transformation_rule"] == "EXCEPT"]
    assert len(set_rows) == 4
    assert {
        row["target_table"] for row in set_rows
    } == {"mart.except_result::except::1", "mart.except_result"}


def test_additional_objects_consume_union_scope_once(s2t_db):
    rows, report = _parse_object(
        {
            "id": 1,
            "file_id": 1,
            "sheet_name": "Additional objects",
            "row_num": 13,
            "name": "mart.cte_union",
            "sql": (
                "WITH u AS ("
                "SELECT a_id AS id FROM raw.a "
                "UNION ALL SELECT b_id AS other FROM raw.b"
                ") SELECT id FROM u"
            ),
        }
    )

    assert report["scope_count"] == report["sqlglot_scope_count"] == 4
    final_rows = [row for row in rows if row["target_table"] == "mart.cte_union"]
    assert [
        (row["source_table"], row["source_field"], row["target_field"])
        for row in final_rows
    ] == [("mart.cte_union::cte::u", "id", "id")]


def test_additional_objects_do_not_treat_cte_as_union_operand(s2t_db):
    rows, report = _parse_object(
        {
            "id": 1,
            "file_id": 1,
            "sheet_name": "Additional objects",
            "row_num": 14,
            "name": "mart.union_with_cte",
            "sql": (
                "WITH src AS (SELECT id FROM raw.a) "
                "SELECT id FROM src "
                "UNION ALL SELECT id FROM raw.b"
            ),
        }
    )

    assert report["scope_count"] == report["sqlglot_scope_count"] == 4
    assert any(
        row["source_table"] == "mart.union_with_cte::cte::src"
        and row["target_table"] == "mart.union_with_cte::branch::1"
        for row in rows
    )
    assert not any(
        row["source_table"] == "mart.union_with_cte"
        and row["target_table"] != "mart.union_with_cte"
        for row in rows
    )


def test_additional_object_refresh_keeps_previous_generated_rows(s2t_db):
    file_id = _store_structured_metadata()
    analysis = classify_file_sheet_groups(file_id, use_llm=False)

    first = extract_structured_metadata(file_id, analysis)
    conn = get_db_connection()
    try:
        first_ids = [
            row[0]
            for row in conn.execute(
                "SELECT id FROM s2t_transformations WHERE file_id = ? ORDER BY id",
                (file_id,),
            ).fetchall()
        ]
    finally:
        conn.close()
    second_etl = extract_additional_object_transformations(file_id)

    first_etl = first["targets"]["additional_objects"]["etl_transformations"]
    assert first_etl["written"] == 2
    assert first_etl["existing_object_count"] == 0
    assert second_etl["written"] == 0
    assert second_etl["existing_object_count"] == 2
    conn = get_db_connection()
    try:
        second_ids = [
            row[0]
            for row in conn.execute(
                "SELECT id FROM s2t_transformations WHERE file_id = ? ORDER BY id",
                (file_id,),
            ).fetchall()
        ]
    finally:
        conn.close()
    assert second_ids == first_ids


def test_missing_structured_metadata_sheet_does_not_change_its_target(s2t_db):
    file_id = store_excel_data(
        "additional_only.xlsx",
        "model",
        [
            {
                "sheet_name": "Additional objects",
                "skip_reason": None,
                "header": {"start_row": 0, "row_count": 1, "nested": False},
                "columns": ["name", "sql"],
                "data_rows": [["view_new", "SELECT 2"]],
            }
        ],
    )
    conn = get_db_connection()
    conn.execute(
        """
        INSERT INTO pxf_to_a
        (file_id, sheet_name, row_num, external_a_table)
        VALUES (?, 'existing', 99, 'ext_existing')
        """,
        (file_id,),
    )
    conn.commit()
    conn.close()

    analysis = classify_file_sheet_groups(file_id, use_llm=False)
    report = extract_structured_metadata(file_id, analysis)

    assert report["targets"]["additional_objects"]["count"] == 1
    assert report["targets"]["pxf_to_a"]["count"] == 0
    conn = get_db_connection()
    try:
        assert conn.execute(
            "SELECT COUNT(*) FROM pxf_to_a WHERE file_id = ?",
            (file_id,),
        ).fetchone()[0] == 1
    finally:
        conn.close()


@pytest.mark.parametrize(
    (
        "sheet_name",
        "columns",
        "data_row",
        "present_target",
        "missing_target",
    ),
    [
        (
            "Source tables",
            ["Название таблицы-источника", "Описание таблицы-источника"],
            ["src_only", "Source description"],
            "source_tables",
            "target_tables",
        ),
        (
            "Target tables",
            ["Table Name", "Table Entity Definition"],
            ["t_only", "Target description"],
            "target_tables",
            "source_tables",
        ),
    ],
)
def test_missing_catalog_sheet_does_not_change_its_target(
    s2t_db,
    sheet_name,
    columns,
    data_row,
    present_target,
    missing_target,
):
    file_id = store_excel_data(
        "one_catalog_sheet.xlsx",
        "model",
        [
            {
                "sheet_name": sheet_name,
                "skip_reason": None,
                "header": {"start_row": 0, "row_count": 1, "nested": False},
                "columns": columns,
                "data_rows": [data_row],
            }
        ],
    )
    conn = get_db_connection()
    conn.execute(
        f"""
        INSERT INTO {missing_target}
        (file_id, sheet_name, row_num, table_name, description)
        VALUES (?, 'existing', 99, 'existing_table', 'existing description')
        """,
        (file_id,),
    )
    conn.commit()
    conn.close()

    analysis = classify_file_sheet_groups(file_id, use_llm=False)
    report = extract_table_catalogs(file_id, analysis)

    assert report["targets"][present_target]["count"] == 1
    assert report["targets"][missing_target]["sheet_count"] == 0
    assert report["targets"][missing_target]["count"] == 0
    conn = get_db_connection()
    try:
        assert conn.execute(
            f"SELECT COUNT(*) AS n FROM {missing_target} WHERE file_id = ?",
            (file_id,),
        ).fetchone()["n"] == 1
    finally:
        conn.close()


def test_clear_s2t_transformations_deletes_only_current_file(s2t_db):
    conn = get_db_connection()
    conn.executemany(
        """
        INSERT INTO s2t_transformations
        (id, file_id, sheet_name, row_num, target_table)
        VALUES (?, ?, ?, ?, ?)
        """,
        [
            (1, 10, "S2T", 1, "t1"),
            (2, 10, "S2T", 2, "t2"),
            (3, 20, "S2T", 1, "t3"),
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
    conn = get_db_connection()
    try:
        state = conn.execute(
            """
            SELECT desired_revision, applied_revision
            FROM graph_sync_outbox
            WHERE file_id = 10
            """
        ).fetchone()
        assert tuple(state) == (1, 0)
    finally:
        conn.close()


def test_insert_s2t_transformations_enqueues_graph_revision(s2t_db):
    result = insert_s2t_transformations(
        10,
        [
            {
                "file_id": 10,
                "sheet_name": "S2T",
                "row_num": 1,
                "target_table": "target_a",
                "source_table": "source_a",
            }
        ],
    )

    assert result == {"file_id": 10, "count": 1}
    conn = get_db_connection()
    try:
        state = conn.execute(
            """
            SELECT desired_revision, applied_revision
            FROM graph_sync_outbox
            WHERE file_id = 10
            """
        ).fetchone()
        assert tuple(state) == (1, 0)
    finally:
        conn.close()


def test_insert_s2t_rolls_back_when_outbox_write_fails(s2t_db):
    with patch(
        "storage.s2t.enqueue_graph_sync",
        side_effect=RuntimeError("outbox unavailable"),
    ), pytest.raises(RuntimeError, match="outbox unavailable"):
        insert_s2t_transformations(
            10,
            [
                {
                    "file_id": 10,
                    "sheet_name": "S2T",
                    "row_num": 1,
                    "target_table": "target_a",
                }
            ],
        )

    conn = get_db_connection()
    try:
        assert conn.execute(
            "SELECT COUNT(*) FROM s2t_transformations"
        ).fetchone()[0] == 0
    finally:
        conn.close()


def test_clear_s2t_rolls_back_when_outbox_write_fails(s2t_db):
    conn = get_db_connection()
    try:
        conn.execute(
            """
            INSERT INTO s2t_transformations
            (id, file_id, sheet_name, row_num, target_table)
            VALUES (1, 10, 'S2T', 1, 'target_a')
            """
        )
        conn.commit()
    finally:
        conn.close()

    with patch(
        "storage.s2t.enqueue_graph_sync",
        side_effect=RuntimeError("outbox unavailable"),
    ), pytest.raises(RuntimeError, match="outbox unavailable"):
        clear_s2t_transformations(10)

    conn = get_db_connection()
    try:
        assert conn.execute(
            "SELECT COUNT(*) FROM s2t_transformations WHERE file_id = 10"
        ).fetchone()[0] == 1
    finally:
        conn.close()


def test_list_s2t_transformations_without_file_id_reads_global_table(s2t_db):
    conn = get_db_connection()
    conn.executemany(
        """
        INSERT INTO s2t_transformations
        (id, file_id, sheet_name, row_num, target_table)
        VALUES (?, ?, ?, ?, ?)
        """,
        [
            (11, 10, "S2T", 1, "t_first"),
            (12, 20, "S2T", 2, "t_second"),
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


def test_list_s2t_transformations_selects_columns(s2t_db):
    conn = get_db_connection()
    conn.execute(
        """
        INSERT INTO s2t_transformations
        (id, file_id, sheet_name, row_num, target_table, transformation_rule)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (13, 10, "S2T", 3, "t_target", "source.value"),
    )
    conn.commit()
    conn.close()

    result = list_s2t_transformations(
        file_id=None,
        limit=10,
        columns=["transformation_rule"],
    )

    assert result["columns"] == ["transformation_rule"]
    assert result["rows"] == [{"transformation_rule": "source.value"}]


def test_list_s2t_transformations_filters_exact_mapping_fields(s2t_db):
    conn = get_db_connection()
    conn.executemany(
        """
        INSERT INTO s2t_transformations
        (id, file_id, sheet_name, row_num, source_table, source_field,
         target_table, target_field)
        VALUES (?, 10, 'S2T', ?, ?, ?, ?, ?)
        """,
        [
            (21, 1, "s_exact", "source_id", "t_exact", "target_id"),
            (22, 2, "s_exact", "other_id", "t_exact", "target_id"),
            (23, 3, "s_exact", "source_id", "t_exact", "other_id"),
        ],
    )
    conn.commit()
    conn.close()

    result = list_s2t_transformations(
        file_id=None,
        source_table="s_exact",
        source_field="source_id",
        target_table="t_exact",
        target_field="target_id",
    )

    assert result["total"] == 1
    assert result["filters"] == {
        "target_table": "t_exact",
        "source_table": "s_exact",
        "target_field": "target_id",
        "source_field": "source_id",
    }
    assert result["rows"][0]["row_num"] == 1
