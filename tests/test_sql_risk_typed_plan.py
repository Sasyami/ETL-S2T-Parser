"""Pure tests for deterministic exact SQL-risk worker-plan synthesis."""

from __future__ import annotations

from dataclasses import replace

import pytest

from agents.contracts import WorkerPlan
from agents.plan_origin import validate_worker_plan_origin
from agents.plan_requirements import validate_sql_risk_plan_requirements
from agents.sql_risk_scope_contract import (
    GetSourceTargetColumnPairRequirement,
    ReadS2TSourceToTargetRequirement,
    SqlRiskScopeContract,
    build_sql_risk_scope_contract,
)
from agents.sql_risk_typed_plan import (
    TypedSqlRiskWorkerPlan,
    build_typed_sql_risk_worker_plan,
    is_exclusive_constraint_rejection_request,
)


CARDINALITY_TASK = (
    "Оцени риск появления дубликатов при сохранённой S2T-трансформации "
    "raw_order_events_v2 → mart_order_daily_v3. Назови фактический JOIN и "
    "явно отдели подтверждённый механизм от условия по уникальности."
)
CONSTRAINT_TASK = (
    "Для file_id=417 оцени только SQL-риск constraint rejection из-за "
    "nullable-ограничений raw_order_events_v2.customer_key → "
    "mart_order_daily_v3.customer_key. Верни source_not_null=<0|1>, "
    "target_not_null=<0|1> и вывод."
)


def _contract(task: str, aspect: str) -> SqlRiskScopeContract:
    contract = build_sql_risk_scope_contract(
        task,
        [aspect],
        enabled=True,
    )
    assert contract is not None
    return contract


def test_builds_one_closed_cardinality_mapping_step():
    contract = _contract(CARDINALITY_TASK, "cardinality")

    typed = build_typed_sql_risk_worker_plan(CARDINALITY_TASK, contract)

    assert typed is not None
    assert typed.plan_source == "deterministic_sql_risk_scope_v2"
    assert typed.aspect == "cardinality"
    assert typed.evidence_step_index == 0
    assert len(typed.plan.steps) == 1
    step = typed.plan.steps[0]
    assert step.coverage == "all_matches"
    assert step.entity is None
    assert step.scope is None
    assert "raw_order_events_v2" in step.task
    assert "mart_order_daily_v3" in step.task
    assert "metadata" not in step.task.casefold()
    assert contract.requirements == (
        ReadS2TSourceToTargetRequirement(
            source_table="raw_order_events_v2",
            target_table="mart_order_daily_v3",
        ),
    )
    validate_worker_plan_origin(typed.plan, CARDINALITY_TASK)
    validate_sql_risk_plan_requirements(
        typed.plan,
        CARDINALITY_TASK,
        sql_risk_aspects=["cardinality"],
    )


def test_builds_one_joint_constraint_mapping_and_metadata_step():
    contract = _contract(CONSTRAINT_TASK, "constraint_rejection")

    typed = build_typed_sql_risk_worker_plan(CONSTRAINT_TASK, contract)

    assert typed is not None
    assert typed.aspect == "constraint_rejection"
    assert typed.evidence_step_index == 0
    assert len(typed.plan.steps) == 1
    step = typed.plan.steps[0]
    assert step.coverage == "all_matches"
    assert step.entity is None
    assert step.scope is not None
    assert step.scope.file_id == 417
    assert "raw_order_events_v2.customer_key" in step.task
    assert "mart_order_daily_v3.customer_key" in step.task
    assert "mapping" in step.task.casefold()
    assert "metadata" in step.task.casefold()
    assert contract.requirements == (
        ReadS2TSourceToTargetRequirement(
            source_table="raw_order_events_v2",
            target_table="mart_order_daily_v3",
        ),
        GetSourceTargetColumnPairRequirement(
            file_id=417,
            source_table="raw_order_events_v2",
            source_column="customer_key",
            target_table="mart_order_daily_v3",
            target_column="customer_key",
        ),
    )
    validate_worker_plan_origin(typed.plan, CONSTRAINT_TASK)
    validate_sql_risk_plan_requirements(
        typed.plan,
        CONSTRAINT_TASK,
        sql_risk_aspects=["constraint_rejection"],
    )


@pytest.mark.parametrize(
    "task",
    [
        CONSTRAINT_TASK,
        (
            "For file_id=417 assess only constraint rejection from nullable "
            "constraints for raw_order_events_v2.customer_key → "
            "mart_order_daily_v3.customer_key. Return "
            "source_not_null=<0|1>, target_not_null=<0|1>, and a conclusion."
        ),
    ],
)
def test_recognizes_explicitly_exclusive_constraint_intent(task):
    assert is_exclusive_constraint_rejection_request(task) is True


@pytest.mark.parametrize(
    "task",
    [
        (
            "Для file_id=417 оцени constraint rejection для "
            "raw_order_events_v2.customer_key → "
            "mart_order_daily_v3.customer_key."
        ),
        (
            "Оцени только row filtering. Потом constraint rejection для "
            "raw_order_events_v2.customer_key → "
            "mart_order_daily_v3.customer_key."
        ),
        (
            "Constraint rejection из-за nullable для "
            "raw_order_events_v2.customer_key → "
            "mart_order_daily_v3.customer_key: оцени только этот риск."
        ),
        (
            "Для file_id=417 оцени только constraint rejection для "
            "raw_order_events_v2.customer_key → "
            "mart_order_daily_v3.customer_key и также покажи весь каталог "
            "колонок."
        ),
        (
            "For file_id=417 assess only constraint rejection caused by "
            "incompatible data types for source field "
            "raw_order_events_v2.customer_key → "
            "mart_order_daily_v3.customer_key."
        ),
        "Только оцени SQL-риск без указанного аспекта.",
    ],
)
def test_constraint_exclusivity_fails_closed(task):
    assert is_exclusive_constraint_rejection_request(task) is False


def test_nonexclusive_constraint_request_stays_on_downstream_path():
    task = CONSTRAINT_TASK.replace("только SQL-риск ", "SQL-риск ")
    contract = _contract(task, "constraint_rejection")

    assert build_typed_sql_risk_worker_plan(task, contract) is None


def test_constraint_request_for_full_sql_stays_on_downstream_path():
    task = (
        "Для file_id=417 оцени только nullable constraint rejection для "
        "raw_order_events_v2.customer_key → "
        "mart_order_daily_v3.customer_key и верни весь SQL."
    )
    contract = _contract(task, "constraint_rejection")

    assert build_typed_sql_risk_worker_plan(task, contract) is None


def test_factual_cardinality_request_stays_on_downstream_path():
    task = (
        "Посчитай фактическое количество дубликатов для "
        "raw_order_events_v2 → mart_order_daily_v3."
    )
    contract = _contract(task, "cardinality")

    assert build_typed_sql_risk_worker_plan(task, contract) is None


@pytest.mark.parametrize(
    "secondary_request",
    [
        "Сравни типы колонок.",
        "Составь DDL исправления.",
        "Дай тестовый SQL проверки.",
        "Дай рекомендации по исправлению.",
        "Оцени качество данных.",
    ],
)
def test_cardinality_with_secondary_request_stays_on_downstream_path(
    secondary_request,
):
    task = f"{CARDINALITY_TASK} {secondary_request}"
    contract = _contract(task, "cardinality")

    assert build_typed_sql_risk_worker_plan(task, contract) is None


@pytest.mark.parametrize(
    ("aspect", "task"),
    [
        (
            "row_filtering",
            "Оцени row filtering для raw_order_events_v2 → "
            "mart_order_daily_v3.",
        ),
        (
            "value_changes",
            "Оцени value changes для "
            "stg.raw_order_events_v2.customer_key → "
            "dwh.mart_order_daily_v3.customer_key.",
        ),
        (
            "write_semantics",
            "Оцени write semantics для raw_order_events_v2 → "
            "mart_order_daily_v3.",
        ),
    ],
)
def test_other_aspects_remain_on_downstream_path(aspect, task):
    contract = _contract(task, aspect)

    assert build_typed_sql_risk_worker_plan(task, contract) is None


def test_multiple_aspects_remain_on_downstream_path():
    contract = _contract(CONSTRAINT_TASK, "constraint_rejection")
    multi_aspect = replace(
        contract,
        aspects=("constraint_rejection", "value_changes"),
    )

    assert (
        build_typed_sql_risk_worker_plan(CONSTRAINT_TASK, multi_aspect)
        is None
    )


def test_stale_or_forged_contract_is_not_materialized_into_a_plan():
    contract = _contract(CARDINALITY_TASK, "cardinality")
    forged_scope = replace(
        contract.scope,
        target="mart_other_v9",
        target_table="mart_other_v9",
    )
    forged = replace(
        contract,
        scope=forged_scope,
        requirements=(
            ReadS2TSourceToTargetRequirement(
                source_table="raw_order_events_v2",
                target_table="mart_other_v9",
            ),
        ),
    )

    assert build_typed_sql_risk_worker_plan(CARDINALITY_TASK, forged) is None


def test_constraint_plan_requires_both_typed_evidence_requirements():
    contract = _contract(CONSTRAINT_TASK, "constraint_rejection")
    missing_metadata = replace(
        contract,
        requirements=(contract.requirements[0],),
    )

    assert (
        build_typed_sql_risk_worker_plan(CONSTRAINT_TASK, missing_metadata)
        is None
    )


def test_typed_plan_rejects_requirement_arguments_not_owned_by_scope():
    contract = _contract(CARDINALITY_TASK, "cardinality")
    mismatched_requirement = replace(
        contract,
        requirements=(
            ReadS2TSourceToTargetRequirement(
                source_table=contract.scope.source_table,
                target_table="mart_other_v9",
            ),
        ),
    )

    assert (
        build_typed_sql_risk_worker_plan(
            CARDINALITY_TASK,
            mismatched_requirement,
        )
        is None
    )


def test_typed_plan_envelope_rejects_out_of_range_evidence_step():
    contract = _contract(CARDINALITY_TASK, "cardinality")
    plan = WorkerPlan(
        steps=[
            {
                "task": (
                    "Прочитать exact mapping "
                    "raw_order_events_v2 → mart_order_daily_v3."
                )
            }
        ]
    )

    with pytest.raises(ValueError, match="evidence_step_index"):
        TypedSqlRiskWorkerPlan(
            plan=plan,
            contract=contract,
            evidence_step_index=1,
        )
