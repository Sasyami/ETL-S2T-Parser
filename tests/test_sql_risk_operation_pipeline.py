"""Tests for exact reads → neutral SQLGlot → bounded LLM assessment."""

from __future__ import annotations

from typing import Any

import pytest

from agents.sql_risk_assessment import (
    SqlRiskAssessmentContext,
    validate_sql_risk_assessment,
)
from agents.sql_risk_operation_pipeline import (
    operation_spec_from_contract,
    run_sql_risk_operation_pipeline,
)
from agents.sql_risk_scope_contract import (
    GetSourceTargetColumnPairRequirement,
    ListS2TFieldMappingRequirement,
    SqlRiskExactScope,
    ReadS2TSourceToTargetRequirement,
    SqlRiskScopeContract,
)
from agents.tools.saved_results import SavedResultStore


JOIN_RULE = (
    "SELECT s.id, d.label FROM src_np s LEFT JOIN dim_np d "
    "ON d.id = s.dim_id AND d.version = s.version WHERE s.active = TRUE"
)
MAPPING_ARGS = {"source_table": "src_np", "target_table": "tgt_np"}
METADATA_ARGS = {
    "file_id": 9101,
    "source_table": "src_np",
    "source_column": "id",
    "target_table": "tgt_np",
    "target_column": "id",
}


def _contract(mode: str) -> SqlRiskScopeContract:
    field_mode = mode in {"nullable_constraint", "value_changes"}
    source = "src_np.id" if field_mode else "src_np"
    target = "tgt_np.id" if field_mode else "tgt_np"
    aspect = {
        "row_filtering": "row_filtering",
        "conditional_cardinality": "cardinality",
        "nullable_constraint": "constraint_rejection",
        "value_changes": "value_changes",
        "write_semantics": "write_semantics",
    }[mode]
    requirements: tuple[Any, ...] = (
        (
            ListS2TFieldMappingRequirement(
                source_table="src_np",
                source_field="id",
                target_table="tgt_np",
                target_field="id",
            )
            if field_mode
            else ReadS2TSourceToTargetRequirement(**MAPPING_ARGS)
        ),
    )
    if mode == "nullable_constraint":
        requirements += (GetSourceTargetColumnPairRequirement(**METADATA_ARGS),)
    return SqlRiskScopeContract(
        scope=SqlRiskExactScope(
            source=source,
            target=target,
            source_table="src_np",
            target_table="tgt_np",
            source_field="id" if field_mode else None,
            target_field="id" if field_mode else None,
            file_id=9101 if mode == "nullable_constraint" else None,
        ),
        aspects=(aspect,),  # type: ignore[arg-type]
        requirements=requirements,
        execution_mode=mode,  # type: ignore[arg-type]
    )


def _mapping_rows(rule: str = JOIN_RULE) -> list[dict[str, Any]]:
    return [
        {
            "source_table": "src_np",
            "source_field": "id",
            "target_table": "tgt_np",
            "target_field": "id",
            "transformation_rule": rule,
        },
        {
            "source_table": "src_np",
            "source_field": "value",
            "target_table": "tgt_np",
            "target_field": "value",
            "transformation_rule": rule,
        },
    ]


def _payload(rows: list[dict[str, Any]], *, truncated: bool = False) -> dict:
    total = len(rows) + (1 if truncated else 0)
    return {
        "rows": rows,
        "total_matches": total,
        "returned_rows": len(rows),
        "truncated": truncated,
    }


def _metadata_rows() -> list[dict[str, Any]]:
    return [
        {
            "column_role": "source",
            "file_id": 9101,
            "table_name": "src_np",
            "column_name": "id",
            "data_type": "uuid",
            "primary_key": 0,
            "not_null": 0,
        },
        {
            "column_role": "target",
            "file_id": 9101,
            "table_name": "tgt_np",
            "column_name": "id",
            "data_type": "uuid",
            "primary_key": 1,
            "not_null": 1,
        },
    ]


def _valid_assessment(
    captured: list[SqlRiskAssessmentContext] | None = None,
    *,
    display: bool = True,
    answer: str = "Сохранённый JOIN создаёт условный риск fan-out.",
):
    def runner(context: SqlRiskAssessmentContext):
        if captured is not None:
            captured.append(context)
        return validate_sql_risk_assessment(
            {
                "status": "complete",
                "outcome": "risk_present",
                "answer": answer,
                "used_evidence_ids": list(context.required_evidence_ids),
                "reviewed_rule_ids": list(context.required_rule_ids),
                "display_evidence_ids": (
                    list(context.displayable_evidence_ids) if display else []
                ),
                "limitations": ["Фактические данные строк не выполнялись."],
            },
            context=context,
        )

    return runner


@pytest.fixture
def store():
    value = SavedResultStore()
    try:
        yield value
    finally:
        value.close()


@pytest.mark.parametrize(
    "mode",
    [
        "row_filtering",
        "conditional_cardinality",
        "nullable_constraint",
        "value_changes",
        "write_semantics",
    ],
)
def test_specs_are_exact_and_contain_no_answer_strategy(mode):
    spec = operation_spec_from_contract(_contract(mode))
    assert spec is not None
    assert spec.execution_mode == mode
    assert "answer_source" not in spec.model_fields
    assert [read.tool_name for read in spec.reads] == (
        ["list_s2t_field_mapping", "get_source_target_column_pair"]
        if mode == "nullable_constraint"
        else [
            "list_s2t_field_mapping"
            if mode == "value_changes"
            else "read_s2t_source_to_target"
        ]
    )
    assert all(read.display_on_complete for read in spec.reads)


def test_pipeline_passes_exact_reader_and_neutral_sqlglot_to_llm(store):
    captured: list[SqlRiskAssessmentContext] = []
    calls: list[dict[str, Any]] = []

    def reader(**arguments):
        calls.append(arguments)
        return _payload(_mapping_rows())

    result = run_sql_risk_operation_pipeline(
        _contract("conditional_cardinality"),
        original_task="Оцени src_np → tgt_np.",
        stable_context="Термин риск означает только риск размножения строк.",
        assessment_runner=_valid_assessment(captured),
        store=store,
        readers={"read_s2t_source_to_target": reader},
        evidence_namespace="neutral",
    )

    assert result.status == "complete"
    assert result.answer_source == "sql_risk_scope_llm"
    assert result.answer == "Сохранённый JOIN создаёт условный риск fan-out."
    assert calls == [MAPPING_ARGS]
    assert result.used_evidence_ids == [result.evidence[0].evidence_id]
    assert result.display_evidence_ids == result.used_evidence_ids
    assert len(result.display_items) == 1
    context = captured[0]
    assert context.stable_context == (
        "Термин риск означает только риск размножения строк."
    )
    assert context.scope.execution_mode == "conditional_cardinality"
    mapping_content = context.evidence[0].content["mapping_rows"]
    assert [item["values"] for item in mapping_content] == _mapping_rows()
    assert len(context.structural_rules) == 1
    structure = context.structural_rules[0].structure
    join = structure["statements"][0]["joins"][0]
    assert join["predicate"] == "d.id = s.dim_id AND d.version = s.version"
    assert [item["sql"] for item in join["equalities"]] == [
        "d.id = s.dim_id",
        "d.version = s.version",
    ]
    assert "outcome" not in structure
    assert "risk" not in structure


def test_nullable_context_contains_both_exact_catalog_roles(store):
    captured: list[SqlRiskAssessmentContext] = []
    readers = {
        "list_s2t_field_mapping": lambda **_: _payload(_mapping_rows()[:1]),
        "get_source_target_column_pair": lambda **_: _payload(_metadata_rows()),
    }

    result = run_sql_risk_operation_pipeline(
        _contract("nullable_constraint"),
        original_task="file_id=9101: src_np.id → tgt_np.id",
        assessment_runner=_valid_assessment(captured, display=False),
        store=store,
        readers=readers,
    )

    assert result.status == "complete"
    assert len(result.evidence) == 2
    metadata_evidence = captured[0].evidence[1]
    assert [row["column_role"] for row in metadata_evidence.content["metadata_rows"]] == [
        "source",
        "target",
    ]
    assert [row["not_null"] for row in metadata_evidence.content["metadata_rows"]] == [
        0,
        1,
    ]


def test_unrelated_write_target_is_neutral_input_not_code_conclusion(store):
    captured: list[SqlRiskAssessmentContext] = []
    rule = "INSERT INTO unrelated_audit SELECT * FROM src_np"
    result = run_sql_risk_operation_pipeline(
        _contract("write_semantics"),
        original_task="Проверь write semantics src_np → tgt_np.",
        assessment_runner=_valid_assessment(captured, display=False),
        store=store,
        readers={
            "read_s2t_source_to_target": lambda **_: _payload(
                _mapping_rows(rule)
            )
        },
    )
    assert result.status == "complete"
    statement = captured[0].structural_rules[0].structure["statements"][0]
    assert statement["statement_kind"] == "INSERT"
    assert statement["write_targets"] == ["unrelated_audit"]
    assert "conclusion" not in statement


def test_invalid_assessment_remains_unavailable_without_fallback(store):
    def invalid_runner(context):
        return validate_sql_risk_assessment(
            {
                "status": "complete",
                "outcome": "risk_present",
                "answer": "Unsupported answer",
                "used_evidence_ids": ["invented"],
                "reviewed_rule_ids": [],
                "display_evidence_ids": [],
                "limitations": [],
            },
            context=context,
        )

    result = run_sql_risk_operation_pipeline(
        _contract("row_filtering"),
        original_task="Проверь src_np → tgt_np.",
        assessment_runner=invalid_runner,
        store=store,
        readers={
            "read_s2t_source_to_target": lambda **_: _payload(_mapping_rows())
        },
    )
    assert result.status == "unavailable"
    assert result.answer_source == "sql_risk_scope_unavailable"
    assert [issue.code for issue in result.issues] == ["analysis_unavailable"]


def test_missing_assessment_runner_never_uses_deterministic_answer(store):
    result = run_sql_risk_operation_pipeline(
        _contract("row_filtering"),
        original_task="Проверь src_np → tgt_np.",
        assessment_runner=None,
        store=store,
        readers={
            "read_s2t_source_to_target": lambda **_: _payload(_mapping_rows())
        },
    )
    assert result.status == "unavailable"
    assert result.answer_source == "sql_risk_scope_unavailable"
    assert not result.facts


def test_empty_mapping_is_structurally_unavailable_before_assessment(store):
    called = False

    def runner(context):
        nonlocal called
        called = True
        return _valid_assessment()(context)

    result = run_sql_risk_operation_pipeline(
        _contract("conditional_cardinality"),
        original_task="Проверь src_np → tgt_np.",
        assessment_runner=runner,
        store=store,
        readers={"read_s2t_source_to_target": lambda **_: _payload([])},
    )
    assert result.status == "unavailable"
    assert called is False
    assert result.issues[0].code == "unpersistable_payload"


def test_sqlglot_parse_error_is_unavailable_before_assessment(store):
    called = False

    def runner(context):
        nonlocal called
        called = True
        return _valid_assessment()(context)

    result = run_sql_risk_operation_pipeline(
        _contract("row_filtering"),
        original_task="Проверь src_np → tgt_np.",
        assessment_runner=runner,
        store=store,
        readers={
            "read_s2t_source_to_target": lambda **_: _payload(
                _mapping_rows("SELECT (")
            )
        },
    )

    assert result.status == "unavailable"
    assert called is False
    assert [issue.code for issue in result.issues] == ["structure_error"]
    assert "sql_parse_error" in result.issues[0].message


def test_foreign_or_truncated_reader_payload_fails_before_assessment(store):
    foreign = _mapping_rows()
    foreign[0] = {**foreign[0], "target_table": "other_target"}
    result = run_sql_risk_operation_pipeline(
        _contract("row_filtering"),
        original_task="Проверь src_np → tgt_np.",
        assessment_runner=_valid_assessment(),
        store=store,
        readers={"read_s2t_source_to_target": lambda **_: _payload(foreign)},
    )
    assert result.status == "unavailable"
    assert result.issues[0].code == "incomplete_evidence"

    second_store = SavedResultStore()
    try:
        truncated = run_sql_risk_operation_pipeline(
            _contract("row_filtering"),
            original_task="Проверь src_np → tgt_np.",
            assessment_runner=_valid_assessment(),
            store=second_store,
            readers={
                "read_s2t_source_to_target": lambda **_: _payload(
                    _mapping_rows()[:1],
                    truncated=True,
                )
            },
        )
    finally:
        second_store.close()
    assert truncated.status == "unavailable"
    assert truncated.issues[0].code == "incomplete_evidence"


def test_invalid_contract_and_missing_store_do_not_execute_reader():
    calls: list[dict[str, Any]] = []
    invalid = _contract("conditional_cardinality")
    invalid = SqlRiskScopeContract(
        scope=invalid.scope,
        aspects=("row_filtering",),
        requirements=invalid.requirements,
        execution_mode=invalid.execution_mode,
    )

    result = run_sql_risk_operation_pipeline(
        invalid,
        original_task="Проверь src_np → tgt_np.",
        assessment_runner=_valid_assessment(),
        readers={
            "read_s2t_source_to_target": lambda **kwargs: calls.append(kwargs)
        },
    )
    assert result.status == "unavailable"
    assert result.issues[0].code == "invalid_contract"
    assert calls == []

    result = run_sql_risk_operation_pipeline(
        _contract("row_filtering"),
        original_task="Проверь src_np → tgt_np.",
        assessment_runner=_valid_assessment(),
        readers={
            "read_s2t_source_to_target": lambda **kwargs: calls.append(kwargs)
        },
    )
    assert result.status == "unavailable"
    assert result.issues[0].code == "store_unavailable"
    assert calls == []


def test_reader_exception_is_structured_unavailable(store):
    def broken(**_):
        raise RuntimeError("database offline")

    result = run_sql_risk_operation_pipeline(
        _contract("row_filtering"),
        original_task="Проверь src_np → tgt_np.",
        assessment_runner=_valid_assessment(),
        store=store,
        readers={"read_s2t_source_to_target": broken},
    )
    assert result.status == "unavailable"
    assert result.issues[0].code == "reader_error"
    assert "database offline" in result.issues[0].message
