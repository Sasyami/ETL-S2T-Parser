"""Reader-contract tests for the model-extracted SQL-risk scope."""

from __future__ import annotations

import pytest

from agents.contracts import EvidenceArtifact
from agents.sql_risk_scope_contract import (
    GetSourceTargetColumnPairRequirement,
    ListS2TFieldMappingRequirement,
    ReadS2TSourceToTargetRequirement,
    build_sql_risk_scope_contract,
    missing_sql_risk_requirements,
    sql_risk_scope_evidence_enabled,
)
from agents.sql_risk_scope_extraction import SqlRiskScopeExtraction


def _extraction(mode: str) -> SqlRiskScopeExtraction:
    field_mode = mode in {"nullable_constraint", "value_changes"}
    return SqlRiskScopeExtraction.model_validate(
        {
            "execution_mode": mode,
            "source": {
                "table_name": "src_alpha",
                "field_name": "source_code" if field_mode else None,
            },
            "target": {
                "table_name": "tgt_beta",
                "field_name": "target_code" if field_mode else None,
            },
            "file_id": 17 if mode == "nullable_constraint" else None,
            "origin": {
                "source": {
                    "table_name": "src_alpha",
                    **(
                        {"field_name": "source_code"}
                        if field_mode
                        else {}
                    ),
                },
                "target": {
                    "table_name": "tgt_beta",
                    **(
                        {"field_name": "target_code"}
                        if field_mode
                        else {}
                    ),
                },
                "file_id": "17" if mode == "nullable_constraint" else None,
            },
        }
    )


def test_operation_scope_flag_is_binary_and_disabled_by_default(monkeypatch):
    monkeypatch.delenv(
        "OPERATION_SQL_RISK_SCOPE_EVIDENCE_EXPERIMENT",
        raising=False,
    )
    assert sql_risk_scope_evidence_enabled() is False
    assert sql_risk_scope_evidence_enabled("0") is False
    assert sql_risk_scope_evidence_enabled("1") is True
    with pytest.raises(ValueError, match="must be 0 or 1"):
        sql_risk_scope_evidence_enabled("true")


@pytest.mark.parametrize(
    ("mode", "aspect"),
    [
        ("row_filtering", "row_filtering"),
        ("conditional_cardinality", "cardinality"),
        ("value_changes", "value_changes"),
        ("write_semantics", "write_semantics"),
    ],
)
def test_model_extraction_builds_one_exact_mapping_requirement(mode, aspect):
    contract = build_sql_risk_scope_contract(_extraction(mode))

    assert contract.execution_mode == mode
    assert contract.aspects == (aspect,)
    assert contract.scope.source_table == "src_alpha"
    assert contract.scope.target_table == "tgt_beta"
    assert contract.scope.source_field == (
        "source_code" if mode == "value_changes" else None
    )
    expected = (
        ListS2TFieldMappingRequirement(
            source_table="src_alpha",
            source_field="source_code",
            target_table="tgt_beta",
            target_field="target_code",
        )
        if mode == "value_changes"
        else ReadS2TSourceToTargetRequirement(
            source_table="src_alpha",
            target_table="tgt_beta",
        )
    )
    assert contract.requirements == (expected,)


def test_nullable_extraction_builds_mapping_and_role_metadata_reads():
    contract = build_sql_risk_scope_contract(_extraction("nullable_constraint"))

    assert contract.aspects == ("constraint_rejection",)
    assert contract.scope.label == (
        "src_alpha.source_code → tgt_beta.target_code"
    )
    assert contract.requirements == (
        ListS2TFieldMappingRequirement(
            source_table="src_alpha",
            source_field="source_code",
            target_table="tgt_beta",
            target_field="target_code",
        ),
        GetSourceTargetColumnPairRequirement(
            file_id=17,
            source_table="src_alpha",
            source_column="source_code",
            target_table="tgt_beta",
            target_column="target_code",
        ),
    )


def test_contract_builder_has_no_task_text_or_literal_parser_boundary():
    with pytest.raises((AttributeError, TypeError)):
        build_sql_risk_scope_contract(  # type: ignore[arg-type]
            "Проверь src_alpha → tgt_beta"
        )


def test_requirement_matcher_checks_exact_name_args_and_truncation():
    contract = build_sql_risk_scope_contract(_extraction("nullable_constraint"))
    mapping, metadata = contract.requirements
    complete = [
        {
            "tool_name": mapping.tool_name,
            "args": mapping.arguments,
            "truncated": False,
        },
        EvidenceArtifact(
            evidence_id="metadata",
            tool_name=metadata.tool_name,
            compact_args=metadata.arguments,
            truncated=False,
        ),
    ]
    assert missing_sql_risk_requirements(contract, complete) == ()

    wrong = [
        complete[0],
        {
            "tool_name": metadata.tool_name,
            "args": {**metadata.arguments, "file_id": 18},
            "truncated": False,
        },
    ]
    assert missing_sql_risk_requirements(contract, wrong) == (metadata,)

    truncated = [complete[0], complete[1].model_copy(update={"truncated": True})]
    assert missing_sql_risk_requirements(contract, truncated) == (metadata,)
