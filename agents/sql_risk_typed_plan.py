"""Deterministic worker plans for narrow exact SQL-risk evidence scopes.

This module owns no data access and does not execute tools.  It translates an
already conservative :class:`SqlRiskScopeContract` into one read-only worker
task only when that task is a closed evidence plan.  The coordinator may then
skip the fallible downstream-plan LLM while retaining the normal agentic
worker, observer and upstream stages.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Literal

from .contracts import (
    PlanScope,
    PlanStep,
    SqlRiskAspect,
    SqlRiskExecutionMode,
    WorkerPlan,
)
from .sql_risk_scope_contract import (
    GetSourceTargetColumnPairRequirement,
    ReadS2TSourceToTargetRequirement,
    SqlRiskScopeContract,
    extract_literal_sql_risk_scope,
)


TypedSqlRiskPlanSource = Literal["deterministic_sql_risk_scope_v2"]
_MAX_IDENTIFIER_CHARS = 200


@dataclass(frozen=True)
class TypedSqlRiskWorkerPlan:
    """One coordinator-owned plan with an explicit evidence-bearing step."""

    plan: WorkerPlan
    contract: SqlRiskScopeContract
    evidence_step_index: int = 0

    plan_source: ClassVar[TypedSqlRiskPlanSource] = (
        "deterministic_sql_risk_scope_v2"
    )

    def __post_init__(self) -> None:
        if not self.plan.steps:
            raise ValueError("typed SQL-risk worker plan must not be empty")
        if not 0 <= self.evidence_step_index < len(self.plan.steps):
            raise ValueError(
                "typed SQL-risk evidence_step_index must reference a plan step"
            )

    @property
    def aspect(self) -> SqlRiskAspect:
        """Return the only aspect supported by this closed typed plan."""

        if len(self.contract.aspects) != 1:
            raise ValueError("typed SQL-risk worker plan requires one aspect")
        return self.contract.aspects[0]


def _contract_matches_original_task(
    original_task: str,
    contract: SqlRiskScopeContract,
    *,
    endpoint_kind: Literal["table", "field"],
) -> bool:
    """Reject manually forged or stale contracts at the pure plan boundary."""

    literal = extract_literal_sql_risk_scope(
        original_task,
        endpoint_kind=endpoint_kind,
    )
    return literal == contract.scope


def _has_mapping_only_requirements(contract: SqlRiskScopeContract) -> bool:
    if len(contract.requirements) != 1 or not isinstance(
        contract.requirements[0],
        ReadS2TSourceToTargetRequirement,
    ):
        return False
    requirement = contract.requirements[0]
    return (
        requirement.source_table == contract.scope.source_table
        and requirement.target_table == contract.scope.target_table
    )


def _has_constraint_requirements(contract: SqlRiskScopeContract) -> bool:
    if (
        len(contract.requirements) != 2
        or not isinstance(
            contract.requirements[0],
            ReadS2TSourceToTargetRequirement,
        )
        or not isinstance(
            contract.requirements[1],
            GetSourceTargetColumnPairRequirement,
        )
    ):
        return False
    scope = contract.scope
    mapping = contract.requirements[0]
    metadata = contract.requirements[1]
    return bool(
        scope.source_field is not None
        and scope.target_field is not None
        and scope.file_id is not None
        and mapping.source_table == scope.source_table
        and mapping.target_table == scope.target_table
        and metadata.file_id == scope.file_id
        and metadata.source_table == scope.source_table
        and metadata.source_column == scope.source_field
        and metadata.target_table == scope.target_table
        and metadata.target_column == scope.target_field
    )


def _scope_identifiers_are_bounded(contract: SqlRiskScopeContract) -> bool:
    scope = contract.scope
    return all(
        len(value) <= _MAX_IDENTIFIER_CHARS
        for value in (
            scope.source_table,
            scope.target_table,
            *(value for value in (scope.source_field, scope.target_field) if value),
        )
    )


def _cardinality_step(contract: SqlRiskScopeContract) -> PlanStep:
    scope = contract.scope
    return PlanStep(
        task=(
            "Прочитать полный exact directed S2T mapping для точного scope "
            f"`{scope.source}` → `{scope.target}` без сужения по полям, "
            "ключам или предполагаемому механизму риска."
        ),
        coverage="all_matches",
    )


def _constraint_step(contract: SqlRiskScopeContract) -> PlanStep:
    scope = contract.scope
    assert scope.file_id is not None
    return PlanStep(
        task=(
            "Прочитать совместно полный exact directed S2T mapping между "
            f"source_table `{scope.source_table}` и target_table "
            f"`{scope.target_table}`, а также ролевые column metadata для "
            f"точной endpoint-пары `{scope.source}` → `{scope.target}` при "
            f"file_id={scope.file_id} без фильтра по ожидаемому not_null."
        ),
        scope=PlanScope(file_id=scope.file_id),
        coverage="all_matches",
    )


def build_typed_sql_risk_worker_plan(
    original_task: str,
    contract: SqlRiskScopeContract | None,
    *,
    sql_risk_execution_mode: SqlRiskExecutionMode,
) -> TypedSqlRiskWorkerPlan | None:
    """Build one safe closed worker plan, otherwise return ``None``.

    The initial architecture intentionally covers only the two cases whose
    required evidence is known to be complete before planning:

    * conditional cardinality from one full directed table mapping;
    * nullable constraint compatibility from that mapping plus the exact
      role-preserving source/target column pair.

    Other aspects and broader requests remain on the unchanged downstream
    planner path.  No identifier is resolved, normalized or inferred here.
    """

    if (
        contract is None
        or len(contract.aspects) != 1
        or contract.execution_mode != sql_risk_execution_mode
        or not _scope_identifiers_are_bounded(contract)
    ):
        return None

    aspect = contract.aspects[0]
    step: PlanStep
    if sql_risk_execution_mode == "conditional_cardinality":
        if not (
            aspect == "cardinality"
            and _has_mapping_only_requirements(contract)
            and _contract_matches_original_task(
                original_task,
                contract,
                endpoint_kind="table",
            )
        ):
            return None
        step = _cardinality_step(contract)
    elif sql_risk_execution_mode == "nullable_constraint":
        if not (
            aspect == "constraint_rejection"
            and contract.scope.is_field_pair
            and contract.scope.file_id is not None
            and _has_constraint_requirements(contract)
            and _contract_matches_original_task(
                original_task,
                contract,
                endpoint_kind="field",
            )
        ):
            return None
        step = _constraint_step(contract)
    else:
        return None

    return TypedSqlRiskWorkerPlan(
        plan=WorkerPlan(steps=[step]),
        contract=contract,
    )


__all__ = [
    "TypedSqlRiskPlanSource",
    "TypedSqlRiskWorkerPlan",
    "build_typed_sql_risk_worker_plan",
]
