"""Fail-closed evidence sufficiency for typed cardinality analysis.

Semantic intent is owned by the structured operation contract.  This module
only verifies that an accepted exact directed reader materialized a complete
mapping for the contract's literal scope.  It never classifies natural
language or claims factual counts or uniqueness.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

from .contracts import EvidenceArtifact
from .sql_risk_scope_contract import (
    ReadS2TSourceToTargetRequirement,
    SqlRiskScopeContract,
)
from .tools.saved_results import SavedResultStore


_EXACT_DIRECTED_TOOLS = frozenset(
    {"read_s2t_mapping", "read_s2t_source_to_target"}
)
_REQUIRED_COLUMNS = frozenset(
    {"source_table", "target_table", "transformation_rule"}
)


@dataclass(frozen=True)
class ExactTablePair:
    """One exact directed table pair from a structured contract."""

    source_table: str
    target_table: str


def cardinality_pair_from_contract(
    contract: SqlRiskScopeContract,
) -> ExactTablePair | None:
    """Return the exact table scope of a conditional-cardinality contract."""

    if (
        contract.execution_mode != "conditional_cardinality"
        or contract.aspects != ("cardinality",)
        or contract.scope.is_field_pair
        or contract.scope.source_field is not None
        or contract.scope.target_field is not None
        or contract.scope.source != contract.scope.source_table
        or contract.scope.target != contract.scope.target_table
        or len(contract.requirements) != 1
        or not isinstance(
            contract.requirements[0],
            ReadS2TSourceToTargetRequirement,
        )
    ):
        return None
    requirement = contract.requirements[0]
    if (
        requirement.source_table != contract.scope.source_table
        or requirement.target_table != contract.scope.target_table
    ):
        return None
    return ExactTablePair(
        source_table=contract.scope.source_table,
        target_table=contract.scope.target_table,
    )


def _sql_literal(value: str) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def complete_cardinality_mapping_evidence_ids(
    contract: SqlRiskScopeContract,
    artifacts: Sequence[EvidenceArtifact],
    store: SavedResultStore,
) -> List[str]:
    """Return evidence IDs that prove one exact mapping is fully materialized.

    The descriptor must expose a trustworthy source total equal to its saved
    row count, be untruncated, contain the required columns, and every saved
    row must match the literal directed pair and carry a non-empty rule.
    """

    pair = cardinality_pair_from_contract(contract)
    if pair is None:
        return []
    evidence_ids: List[str] = []
    for artifact in artifacts:
        if artifact.tool_name not in _EXACT_DIRECTED_TOOLS:
            continue
        args = artifact.compact_args
        if (
            str(args.get("source_table") or "").casefold()
            != pair.source_table.casefold()
            or str(args.get("target_table") or "").casefold()
            != pair.target_table.casefold()
            or artifact.dataset_ref is None
        ):
            continue
        descriptor = store.descriptor(artifact.dataset_ref)
        if (
            descriptor is None
            or descriptor.source_tool != artifact.tool_name
            or descriptor.truncated
            or descriptor.row_count <= 0
            or descriptor.source_total is None
            or descriptor.source_total != descriptor.row_count
            or not _REQUIRED_COLUMNS.issubset(
                {column.name for column in descriptor.columns}
            )
        ):
            continue

        source = _sql_literal(pair.source_table)
        target = _sql_literal(pair.target_table)
        result = store.query(
            result_ref=descriptor.result_ref,
            query=(
                "SELECT COUNT(*) AS saved_rows, "
                "SUM(CASE WHEN LOWER(TRIM(source_table)) = "
                f"LOWER(TRIM({source})) AND LOWER(TRIM(target_table)) = "
                f"LOWER(TRIM({target})) THEN 1 ELSE 0 END) AS exact_rows, "
                "SUM(CASE WHEN NULLIF(TRIM(transformation_rule), '') "
                "IS NULL OR TRIM(transformation_rule) = '-' "
                "THEN 1 ELSE 0 END) AS missing_rule_rows FROM result"
            ),
            preview_limit=1,
        )
        rows = result.get("rows") or []
        if (
            result.get("error")
            or result.get("input_truncated")
            or result.get("truncated")
            or len(rows) != 1
            or not isinstance(rows[0], dict)
            or int(rows[0].get("saved_rows") or 0) != descriptor.row_count
            or int(rows[0].get("exact_rows") or 0) != descriptor.row_count
            or int(rows[0].get("missing_rule_rows") or 0) != 0
        ):
            continue
        if artifact.evidence_id not in evidence_ids:
            evidence_ids.append(artifact.evidence_id)
    return evidence_ids


__all__ = [
    "ExactTablePair",
    "cardinality_pair_from_contract",
    "complete_cardinality_mapping_evidence_ids",
]
