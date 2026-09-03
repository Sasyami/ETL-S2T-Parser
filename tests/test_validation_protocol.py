from unittest.mock import patch

from agents.tools.common import pack_tabular_rows
from agents.validation_protocol import (
    S2TAnalysisItem,
    S2TAnalysisOutput,
    ValidationProtocolContract,
    _sql_facts,
    build_deterministic_analysis_items,
    build_s2t_analysis_display_payloads,
    build_s2t_analysis_payload,
    merge_s2t_analysis_output,
    read_validation_protocol_inputs,
    render_s2t_analysis_answer,
    validate_s2t_analysis_output,
)


def _packed_s2t(rows):
    columns = [
        "file_id",
        "sheet_name",
        "row_num",
        "source_table",
        "source_field",
        "target_table",
        "target_field",
        "transformation_rule",
        "source_layer",
        "target_layer",
    ]
    return {
        **pack_tabular_rows(
            rows,
            columns=columns,
            dictionary_columns=(
                "sheet_name",
                "source_table",
                "target_table",
                "transformation_rule",
                "source_layer",
                "target_layer",
            ),
        ),
        "truncated": False,
    }


def _contract():
    return ValidationProtocolContract(
        file_id=41,
        source_tables=["source_entity"],
        target_tables=["target_entity"],
        requested_analyses=[
            "row_loss_risk",
            "duplicate_risk",
            "unmapped_required_fields",
            "transformation_consistency",
        ],
    )


def test_row_loss_contract_does_not_require_internal_file_id():
    contract = ValidationProtocolContract(
        source_tables=["source_entity"],
        target_tables=["target_entity"],
        requested_analyses=["row_loss_risk"],
    )
    empty_s2t = _packed_s2t([])

    with (
        patch("agents.validation_protocol.read_s2t_by_target_table") as reader,
        patch("agents.validation_protocol.list_target_column_catalog") as catalog,
        patch("agents.validation_protocol._persist_full_result"),
    ):
        reader.invoke.return_value = empty_s2t
        results = read_validation_protocol_inputs(contract)

    assert contract.file_id is None
    assert [result["kind"] for result in results] == ["s2t_target"]
    catalog.invoke.assert_not_called()


def test_catalog_scope_can_be_resolved_from_exact_filename():
    contract = ValidationProtocolContract(
        filename="Mapping.xlsx",
        source_tables=["source_entity"],
        target_tables=["target_entity"],
        requested_analyses=["unmapped_required_fields"],
    )
    empty_s2t = _packed_s2t([])
    empty_catalog = {"columns": [], "rows": [], "truncated": False}

    with (
        patch("agents.validation_protocol.resolve_file") as resolver,
        patch("agents.validation_protocol.read_s2t_by_target_table") as reader,
        patch("agents.validation_protocol.list_target_column_catalog") as catalog,
        patch("agents.validation_protocol._persist_full_result"),
    ):
        resolver.invoke.return_value = {
            "file_id": 41,
            "filename": "Mapping.xlsx",
        }
        reader.invoke.return_value = empty_s2t
        catalog.invoke.return_value = empty_catalog
        results = read_validation_protocol_inputs(contract)

    resolver.invoke.assert_called_once_with({"filename": "Mapping.xlsx"})
    catalog.invoke.assert_called_once_with(
        {"file_id": 41, "table_name": "target_entity"}
    )
    assert [result["kind"] for result in results] == [
        "s2t_target",
        "target_column_catalog",
    ]


def test_s2t_analysis_payload_exposes_full_typed_reader_results():
    rule = (
        "SELECT s.* FROM $$305stage.source_entity AS s "
        "LEFT JOIN $$305stage.dictionary AS d ON s.dictionary_id = d.id "
        "WHERE s.deleted_at IS NULL"
    )
    pair_rows = [
        {
            "file_id": 41,
            "sheet_name": "Mapping",
            "row_num": 10,
            "source_table": "source_entity",
            "source_field": "entity_id",
            "target_table": "target_entity",
            "target_field": "entity_id",
            "transformation_rule": rule,
            "source_layer": "stage",
            "target_layer": "core",
        }
    ]
    pair_rows.append({**pair_rows[0], "row_num": 11})
    catalog_rows = [
        {
            "file_id": 41,
            "table_name": "target_entity",
            "column_name": "ENTITY_ID",
            "primary_key": True,
            "not_null": True,
        }
    ]

    payload = build_s2t_analysis_payload(
        _contract(),
        reader_results=[
            {
                "kind": "s2t_target",
                "args": {"target_table": "target_entity"},
                "payload": _packed_s2t(pair_rows),
            },
            {
                "kind": "target_column_catalog",
                "args": {"file_id": 41, "table_name": "target_entity"},
                "payload": {
                    "columns": list(catalog_rows[0]),
                    "rows": catalog_rows,
                    "truncated": False,
                },
            },
        ],
    )

    assert payload["contract"]["requested_analyses"] == [
        "row_loss_risk",
        "duplicate_risk",
        "unmapped_required_fields",
        "transformation_consistency",
    ]
    assert [result["kind"] for result in payload["reader_results"]] == [
        "s2t_target",
        "target_column_catalog",
    ]
    s2t_result = payload["reader_results"][0]
    assert s2t_result["raw_row_count"] == 2
    assert len(s2t_result["mappings"]) == 2
    assert {row["rule_id"] for row in s2t_result["mappings"]} == {"rule_1"}
    assert len(s2t_result["rules"]) == 1
    sql_facts = s2t_result["rules"][0]["sql_facts"]
    assert sql_facts["where_predicates"] == ["s.deleted_at IS NULL"]
    assert sql_facts["effective_where_predicates"] == [
        "s.deleted_at IS NULL"
    ]
    assert sql_facts["parse_status"] == "ok"
    assert sql_facts["quoted_dollar_schemas_for_parse"] is True
    assert sql_facts["joins"][0]["preserves_left_rows"] is True
    assert (
        sql_facts["joins"][0]["on_predicate_filters_preserved_left_rows"]
        is False
    )
    assert payload["reader_results"][1]["rows"] == catalog_rows
    coverage = payload["derived_facts"]["mapping_coverage"][0]
    assert coverage["required_fields"] == ["ENTITY_ID"]
    assert coverage["unmapped_required_fields"] == []
    assert coverage["catalog_fields"] == ["ENTITY_ID"]
    assert coverage["unmapped_catalog_fields"] == []
    assert coverage["mapped_fields_not_in_catalog"] == []
    assert coverage["catalog_fields_count"] == 1
    assert coverage["mapped_target_fields_count"] == 1
    assert coverage["exact_rule_count"] == 1
    assert coverage["physical_source_tables"] == [
        "$$305stage.source_entity",
        "$$305stage.dictionary",
    ]
    assert payload["llm_requested_analyses"] == [
        "transformation_consistency",
    ]


def test_s2t_analysis_display_exposes_reader_evidence_without_runtime_ref():
    displays = build_s2t_analysis_display_payloads(
        [
            {
                "kind": "s2t_target",
                "args": {"target_table": "target_entity"},
                "tool_name": "read_s2t_by_target_table",
                "payload": {
                    "columns": ["target_table", "transformation_rule"],
                    "rows": [
                        {
                            "target_table": "target_entity",
                            "transformation_rule": "SELECT * FROM source_entity",
                        }
                    ],
                    "truncated": False,
                    "saved_result": {"result_ref": "internal-only"},
                },
            }
        ]
    )

    assert [item["name"] for item in displays] == [
        "read_s2t_by_target_table"
    ]
    assert '"target_table":"target_entity"' in displays[0]["content"]
    assert "SELECT * FROM source_entity" in displays[0]["content"]
    assert "internal-only" not in displays[0]["content"]


def test_sql_facts_exclude_tautology_and_right_join_subquery_filter():
    facts = _sql_facts(
        "WITH src AS ("
        "SELECT * FROM $$305stage.source_entity "
        "WHERE TRUE AND entity_id IS NOT NULL"
        ") "
        "SELECT src.* FROM src "
        "LEFT JOIN (SELECT * FROM $$305stage.dictionary WHERE active = 1) d "
        "ON src.dictionary_id = d.id WHERE 1 = 1"
    )

    assert facts["parse_status"] == "ok"
    assert facts["effective_where_predicates"] == [
        "TRUE AND entity_id IS NOT NULL"
    ]
    assert any("active = 1" in value for value in facts["where_predicates"])
    assert all(
        "active = 1" not in value
        for value in facts["effective_where_predicates"]
    )


def test_unmapped_required_fields_are_computed_without_llm_case_insensitively():
    contract = _contract()
    reader_results = [
        {
            "kind": "s2t_target",
            "args": {"target_table": "target_entity"},
            "payload": _packed_s2t(
                [
                    {
                        "file_id": 41,
                        "sheet_name": "Mapping",
                        "row_num": 10,
                        "source_table": "source_entity",
                        "source_field": "entity_id",
                        "target_table": "target_entity",
                        "target_field": "entity_id",
                        "transformation_rule": "s.entity_id",
                        "source_layer": "stage",
                        "target_layer": "core",
                    }
                ]
            ),
        },
        {
            "kind": "target_column_catalog",
            "args": {"file_id": 41, "table_name": "target_entity"},
            "payload": {
                "columns": ["column_name", "not_null"],
                "rows": [
                    {"column_name": "ENTITY_ID", "not_null": True},
                    {"column_name": "OPTIONAL_VALUE", "not_null": False},
                ],
                "truncated": False,
            },
        },
    ]

    items = build_deterministic_analysis_items(
        contract,
        reader_results=reader_results,
    )

    unmapped = next(
        item for item in items if item.kind == "unmapped_required_fields"
    )
    assert unmapped.target_table == "target_entity"
    assert unmapped.conclusion == (
        "Обязательных target-полей без S2T-маппинга нет."
    )
    assert "unmapped_required_fields=[]" in unmapped.evidence

    row_loss = next(item for item in items if item.kind == "row_loss_risk")
    assert "не найден" in row_loss.conclusion

    duplicates = next(item for item in items if item.kind == "duplicate_risk")
    assert "JOIN-индуцированный риск" in duplicates.conclusion


def test_deterministic_items_merge_with_semantic_output():
    semantic = S2TAnalysisOutput(
        analyses=[
            S2TAnalysisItem(
                target_table="target_entity",
                kind="row_loss_risk",
                conclusion="Риск подтверждён условием WHERE.",
            )
        ]
    )
    deterministic = [
        S2TAnalysisItem(
            target_table="target_entity",
            kind="unmapped_required_fields",
            conclusion="Обязательных полей без маппинга нет.",
        )
    ]

    merged = merge_s2t_analysis_output(semantic, deterministic)

    assert [item.kind for item in merged.analyses] == [
        "row_loss_risk",
        "unmapped_required_fields",
    ]


def test_s2t_analysis_output_must_cover_exact_requested_set():
    contract = ValidationProtocolContract(
        file_id=41,
        source_tables=["source_entity"],
        target_tables=["target_entity"],
        requested_analyses=["row_loss_risk", "duplicate_risk"],
    )
    incomplete = S2TAnalysisOutput(
        analyses=[
            S2TAnalysisItem(
                target_table="target_entity",
                kind="row_loss_risk",
                conclusion="Фильтр может исключать строки.",
            )
        ]
    )

    try:
        validate_s2t_analysis_output(contract, incomplete)
    except ValueError as exc:
        assert "по одному выводу" in str(exc)
    else:
        raise AssertionError("Неполный анализ должен быть отклонён")


def test_s2t_analysis_renderer_does_not_claim_physical_execution_or_emit_sql():
    output = S2TAnalysisOutput(
        analyses=[
            S2TAnalysisItem(
                target_table="target_entity",
                kind="row_loss_risk",
                conclusion="WHERE deleted_at IS NULL может исключить строки.",
                evidence=["transformation_rule содержит WHERE"],
                limitations=["Фактическое число строк неизвестно"],
            ),
            S2TAnalysisItem(
                target_table="target_entity",
                kind="duplicate_risk",
                conclusion="LEFT JOIN требует проверки кардинальности.",
            ),
            S2TAnalysisItem(
                target_table="target_entity",
                kind="unmapped_required_fields",
                conclusion="Обязательных полей без маппинга нет.",
            ),
            S2TAnalysisItem(
                target_table="target_entity",
                kind="transformation_consistency",
                conclusion="Пара и правило присутствуют в S2T.",
            ),
        ]
    )

    answer = render_s2t_analysis_answer(_contract(), output)

    assert "Физические данные логических ETL-таблиц не запрашивались" in answer
    assert "WHERE deleted_at IS NULL" in answer
    assert "Фактическое число строк неизвестно" in answer
    assert "SELECT COUNT" not in answer
    assert "FROM \"target_entity\"" not in answer
