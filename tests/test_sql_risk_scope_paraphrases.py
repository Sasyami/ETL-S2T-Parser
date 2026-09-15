"""Wording-independent validation of model-owned SQL-risk extractions."""

from __future__ import annotations

import pytest

from agents.sql_risk_operation_pipeline import operation_spec_from_contract
from agents.sql_risk_scope_contract import build_sql_risk_scope_contract
from agents.sql_risk_scope_extraction import (
    validate_sql_risk_scope_extraction,
)


@pytest.mark.parametrize(
    ("task", "source", "target"),
    [
        (
            "По направлению stage_clickstream → dm_session_rollup нужен "
            "условный анализ размножения строк.",
            "stage_clickstream",
            "dm_session_rollup",
        ),
        (
            "Using stored S2T from ods_payments into mart_payment_daily, "
            "assess fan-out and do not claim actual duplicates.",
            "ods_payments",
            "mart_payment_daily",
        ),
        (
            "Реальные дубли не утверждай: источник landing_sales, "
            "приёмник core_sales; нужен только cardinality.",
            "landing_sales",
            "core_sales",
        ),
    ],
)
def test_different_wording_accepts_the_model_selected_exact_scope(
    task,
    source,
    target,
):
    validation = validate_sql_risk_scope_extraction(
        {
            "execution_mode": "conditional_cardinality",
            "source": {"table_name": source, "field_name": None},
            "target": {"table_name": target, "field_name": None},
            "file_id": None,
            "origin": {
                "source": {"table_name": source},
                "target": {"table_name": target},
                "file_id": None,
            },
        },
        original_task=task,
    )
    assert validation.status == "valid"
    assert validation.contract is not None

    contract = build_sql_risk_scope_contract(validation.contract)
    spec = operation_spec_from_contract(contract)
    assert spec is not None
    assert spec.execution_mode == "conditional_cardinality"
    assert spec.reads[0].arguments == {
        "source_table": source,
        "target_table": target,
    }


@pytest.mark.parametrize(
    ("task", "file_id", "source", "source_field", "target", "target_field"),
    [
        (
            "В file_id=631 проверь nullable для ingest_customer.client_id → "
            "hub_customer.client_id.",
            631,
            "ingest_customer",
            "client_id",
            "hub_customer",
            "client_id",
        ),
        (
            "Only nullability, raw_profile.profile_id → dim_profile.profile_id, "
            "file_id: 742.",
            742,
            "raw_profile",
            "profile_id",
            "dim_profile",
            "profile_id",
        ),
    ],
)
def test_nullable_wording_is_owned_by_llm_while_origin_stays_exact(
    task,
    file_id,
    source,
    source_field,
    target,
    target_field,
):
    validation = validate_sql_risk_scope_extraction(
        {
            "execution_mode": "nullable_constraint",
            "source": {"table_name": source, "field_name": source_field},
            "target": {"table_name": target, "field_name": target_field},
            "file_id": file_id,
            "origin": {
                "source": {
                    "table_name": source,
                    "field_name": source_field,
                },
                "target": {
                    "table_name": target,
                    "field_name": target_field,
                },
                "file_id": str(file_id),
            },
        },
        original_task=task,
    )
    assert validation.status == "valid"
    assert validation.contract is not None
    spec = operation_spec_from_contract(
        build_sql_risk_scope_contract(validation.contract)
    )
    assert spec is not None
    assert [read.tool_name for read in spec.reads] == [
        "list_s2t_field_mapping",
        "get_source_target_column_pair",
    ]
