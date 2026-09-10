"""Generalization tests for compositional typed SQL-risk intent detection.

The examples deliberately use identifiers and sentence shapes that do not
occur in the revealed live fixtures.  Eligibility must depend on the requested
operation and its typed literal scope, not on one memorized prompt template.
"""

from __future__ import annotations

import pytest

from agents.sql_risk_scope_contract import build_sql_risk_scope_contract
from agents.sql_risk_typed_plan import build_typed_sql_risk_worker_plan


CARDINALITY_PARAPHRASES = [
    pytest.param(
        (
            "По направлению stage_clickstream → dm_session_rollup нужен "
            "только условный анализ размножения строк. Укажи JOIN из "
            "сохранённого S2T и отдельно поясни, что известно о механизме, "
            "уникальности ключей и наличии реальных дубликатов."
        ),
        "stage_clickstream",
        "dm_session_rollup",
        id="ru-scope-first-row-multiplication",
    ),
    pytest.param(
        (
            "Какой фактический JOIN способен дать fan-out при переносе "
            "load_events → fact_event_day? Оцени только эту условную "
            "возможность по сохранённому mapping; уникальность ключей и "
            "фактические дубли не считай установленными."
        ),
        "load_events",
        "fact_event_day",
        id="ru-question-first-fan-out",
    ),
    pytest.param(
        (
            "Using the stored S2T for ods_payments → mart_payment_daily, "
            "assess only conditional row multiplication. State the concrete "
            "JOIN, then distinguish the established mechanism from unknown "
            "join-key uniqueness and unproven actual duplicates."
        ),
        "ods_payments",
        "mart_payment_daily",
        id="en-mapping-first-conditional",
    ),
    pytest.param(
        (
            "Реальные дубликаты заранее не утверждай. Для "
            "landing_sales → core_sales оцени по сохранённому S2T только "
            "условный cardinality risk: сначала состояние уникальности "
            "ключей, затем подтверждённый JOIN-механизм."
        ),
        "landing_sales",
        "core_sales",
        id="ru-reordered-negative-first",
    ),
]


NULLABLE_PARAPHRASES = [
    pytest.param(
        (
            "В file_id=631 для ingest_customer.client_id → "
            "hub_customer.client_id оцени только constraint rejection, "
            "связанный с nullable-контрактом. Сообщи оба признака not_null "
            "и заключение; прочие SQL-риски не рассматривай."
        ),
        631,
        "ingest_customer",
        "client_id",
        "hub_customer",
        "client_id",
        id="ru-scope-first-nullable-contract",
    ),
    pytest.param(
        (
            "Другие constraints игнорируй. Только риск отказа записи из-за "
            "nullable оцени для bronze_invoice.invoice_key → "
            "silver_invoice.invoice_key в file_id: 908; выдай "
            "source_not_null, target_not_null и итог."
        ),
        908,
        "bronze_invoice",
        "invoice_key",
        "silver_invoice",
        "invoice_key",
        id="ru-exclusion-first-colon-file-id",
    ),
    pytest.param(
        (
            "Ignore every other SQL risk. For raw_profile.profile_id → "
            "dim_profile.profile_id in file_id=742, determine only whether "
            "the source/target nullability contract can cause a NOT NULL "
            "constraint rejection. State both not-null flags and the "
            "conclusion."
        ),
        742,
        "raw_profile",
        "profile_id",
        "dim_profile",
        "profile_id",
        id="en-exclusion-first-not-null",
    ),
    pytest.param(
        (
            "Нужны source_not_null и target_not_null плюс вывод. Анализ "
            "ограничь только nullable constraint rejection для пары "
            "src_claim.claim_no → tgt_claim.claim_no, file_id=519."
        ),
        519,
        "src_claim",
        "claim_no",
        "tgt_claim",
        "claim_no",
        id="ru-output-first-reordered",
    ),
]


@pytest.mark.parametrize(
    ("task", "source_table", "target_table"),
    CARDINALITY_PARAPHRASES,
)
def test_cardinality_paraphrases_build_the_same_typed_evidence_plan(
    task,
    source_table,
    target_table,
):
    contract = build_sql_risk_scope_contract(
        task,
        ["cardinality"],
        enabled=True,
        execution_mode="conditional_cardinality",
    )

    typed = build_typed_sql_risk_worker_plan(
        task,
        contract,
        sql_risk_execution_mode="conditional_cardinality",
    )

    assert contract is not None
    assert contract.scope.source_table == source_table
    assert contract.scope.target_table == target_table
    assert contract.execution_mode == "conditional_cardinality"
    assert typed is not None
    assert typed.aspect == "cardinality"
    assert typed.plan_source == "deterministic_sql_risk_scope_v2"
    assert len(typed.plan.steps) == 1
    assert typed.plan.steps[0].coverage == "all_matches"
    assert [requirement.tool_name for requirement in contract.requirements] == [
        "read_s2t_source_to_target"
    ]
    assert [dict(requirement.arguments) for requirement in contract.requirements] == [
        {
            "source_table": source_table,
            "target_table": target_table,
        }
    ]


@pytest.mark.parametrize(
    (
        "task",
        "file_id",
        "source_table",
        "source_field",
        "target_table",
        "target_field",
    ),
    NULLABLE_PARAPHRASES,
)
def test_nullable_paraphrases_build_the_same_typed_evidence_plan(
    task,
    file_id,
    source_table,
    source_field,
    target_table,
    target_field,
):
    contract = build_sql_risk_scope_contract(
        task,
        ["constraint_rejection"],
        enabled=True,
        execution_mode="nullable_constraint",
    )

    typed = build_typed_sql_risk_worker_plan(
        task,
        contract,
        sql_risk_execution_mode="nullable_constraint",
    )

    assert contract is not None
    assert contract.scope.file_id == file_id
    assert contract.scope.source_table == source_table
    assert contract.scope.source_field == source_field
    assert contract.scope.target_table == target_table
    assert contract.scope.target_field == target_field
    assert contract.execution_mode == "nullable_constraint"
    assert typed is not None
    assert typed.aspect == "constraint_rejection"
    assert typed.plan_source == "deterministic_sql_risk_scope_v2"
    assert len(typed.plan.steps) == 1
    assert typed.plan.steps[0].coverage == "all_matches"
    assert [requirement.tool_name for requirement in contract.requirements] == [
        "read_s2t_source_to_target",
        "get_source_target_column_pair",
    ]
    assert dict(contract.requirements[0].arguments) == {
        "source_table": source_table,
        "target_table": target_table,
    }
    assert dict(contract.requirements[1].arguments) == {
        "file_id": file_id,
        "source_table": source_table,
        "source_column": source_field,
        "target_table": target_table,
        "target_column": target_field,
    }
