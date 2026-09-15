"""Exact reader contracts for the separate SQL-risk scope pipeline.

Natural-language interpretation and mode selection happen in the pipeline's
typed LLM extraction stage.  This module only translates that already
validated extraction into immutable exact-reader requirements; it never reads
or classifies the user's text.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, Iterable, Literal, Mapping, TypedDict

from .contracts import SqlRiskAspect
from .experiment_flags import (
    OPERATION_SQL_RISK_SCOPE_EVIDENCE_EXPERIMENT_ENV,
    experiment_flag_enabled,
    parse_binary_experiment_flag,
)
from .sql_risk_scope_extraction import (
    SqlRiskScopeExecutionMode,
    SqlRiskScopeExtraction,
)

SqlRiskToolName = Literal[
    "read_s2t_source_to_target",
    "list_s2t_field_mapping",
    "get_source_target_column_pair",
]

class ReadS2TSourceToTargetArguments(TypedDict):
    """Exact arguments of ``read_s2t_source_to_target``."""

    source_table: str
    target_table: str


class ListS2TFieldMappingArguments(TypedDict):
    """Exact arguments of ``list_s2t_field_mapping``."""

    source_table: str
    source_field: str
    target_table: str
    target_field: str


class GetSourceTargetColumnPairArguments(TypedDict):
    """Exact arguments of ``get_source_target_column_pair``."""

    file_id: int
    source_table: str
    source_column: str
    target_table: str
    target_column: str


@dataclass(frozen=True)
class SqlRiskExactScope:
    """One exact pair selected by the internal scope-extraction LLM."""

    source: str
    target: str
    source_table: str
    target_table: str
    source_field: str | None = None
    target_field: str | None = None
    file_id: int | None = None

    @property
    def is_field_pair(self) -> bool:
        """Return whether both endpoints are explicit ``table.field`` names."""

        return self.source_field is not None and self.target_field is not None

    @property
    def label(self) -> str:
        """Return the exact compact label suitable for a public answer."""

        return f"{self.source} → {self.target}"


@dataclass(frozen=True)
class ReadS2TSourceToTargetRequirement:
    """Required full directed S2T read for the literal table pair."""

    source_table: str
    target_table: str

    tool_name: ClassVar[Literal["read_s2t_source_to_target"]] = (
        "read_s2t_source_to_target"
    )

    @property
    def arguments(self) -> ReadS2TSourceToTargetArguments:
        return {
            "source_table": self.source_table,
            "target_table": self.target_table,
        }


@dataclass(frozen=True)
class ListS2TFieldMappingRequirement:
    """Required exact role-preserving S2T read for one field pair."""

    source_table: str
    source_field: str
    target_table: str
    target_field: str

    tool_name: ClassVar[Literal["list_s2t_field_mapping"]] = (
        "list_s2t_field_mapping"
    )

    @property
    def arguments(self) -> ListS2TFieldMappingArguments:
        return {
            "source_table": self.source_table,
            "source_field": self.source_field,
            "target_table": self.target_table,
            "target_field": self.target_field,
        }


@dataclass(frozen=True)
class GetSourceTargetColumnPairRequirement:
    """Required exact role-preserving endpoint metadata read."""

    file_id: int
    source_table: str
    source_column: str
    target_table: str
    target_column: str

    tool_name: ClassVar[Literal["get_source_target_column_pair"]] = (
        "get_source_target_column_pair"
    )

    @property
    def arguments(self) -> GetSourceTargetColumnPairArguments:
        return {
            "file_id": self.file_id,
            "source_table": self.source_table,
            "source_column": self.source_column,
            "target_table": self.target_table,
            "target_column": self.target_column,
        }


SqlRiskEvidenceRequirement = (
    ReadS2TSourceToTargetRequirement
    | ListS2TFieldMappingRequirement
    | GetSourceTargetColumnPairRequirement
)


@dataclass(frozen=True)
class SqlRiskScopeContract:
    """Typed, immutable runtime requirements for one literal scope."""

    scope: SqlRiskExactScope
    aspects: tuple[SqlRiskAspect, ...]
    requirements: tuple[SqlRiskEvidenceRequirement, ...]
    execution_mode: SqlRiskScopeExecutionMode

    @property
    def tool_names(self) -> tuple[SqlRiskToolName, ...]:
        return tuple(requirement.tool_name for requirement in self.requirements)


def sql_risk_scope_evidence_enabled(value: str | None = None) -> bool:
    """Return whether the isolated operation-scope pipeline is enabled.

    The environment is used when ``value`` is omitted. Only literal ``0`` and
    ``1`` are valid; an unset variable uses the disabled default.
    """

    if value is None:
        return experiment_flag_enabled(
            OPERATION_SQL_RISK_SCOPE_EVIDENCE_EXPERIMENT_ENV,
        )
    return parse_binary_experiment_flag(
        OPERATION_SQL_RISK_SCOPE_EVIDENCE_EXPERIMENT_ENV,
        value,
    )


def build_sql_risk_scope_contract(
    extraction: SqlRiskScopeExtraction,
) -> SqlRiskScopeContract:
    """Translate a validated model-owned extraction into exact reads.

    This function deliberately accepts no task text and has no off/ambiguity
    fallback. Exact origin validation is performed before this boundary by
    :func:`validate_sql_risk_scope_extraction`.
    """

    mode_to_aspect: dict[SqlRiskScopeExecutionMode, SqlRiskAspect] = {
        "row_filtering": "row_filtering",
        "conditional_cardinality": "cardinality",
        "nullable_constraint": "constraint_rejection",
        "value_changes": "value_changes",
        "write_semantics": "write_semantics",
    }
    execution_mode: SqlRiskScopeExecutionMode = extraction.execution_mode
    aspect = mode_to_aspect[execution_mode]
    source_field = extraction.source.field_name
    target_field = extraction.target.field_name
    scope = SqlRiskExactScope(
        source=extraction.source.literal,
        target=extraction.target.literal,
        source_table=extraction.source.table_name,
        target_table=extraction.target.table_name,
        source_field=source_field,
        target_field=target_field,
        file_id=extraction.file_id,
    )

    if scope.is_field_pair:
        assert scope.source_field is not None
        assert scope.target_field is not None
        requirements: list[SqlRiskEvidenceRequirement] = [
            ListS2TFieldMappingRequirement(
                source_table=scope.source_table,
                source_field=scope.source_field,
                target_table=scope.target_table,
                target_field=scope.target_field,
            )
        ]
    else:
        requirements = [
            ReadS2TSourceToTargetRequirement(
                source_table=scope.source_table,
                target_table=scope.target_table,
            )
        ]
    if (
        aspect == "constraint_rejection"
        and scope.is_field_pair
        and scope.file_id is not None
    ):
        # ``is_field_pair`` makes the fields non-optional, but retaining this
        # assertion keeps the constructor statically and dynamically honest.
        assert scope.source_field is not None
        assert scope.target_field is not None
        requirements.append(
            GetSourceTargetColumnPairRequirement(
                file_id=scope.file_id,
                source_table=scope.source_table,
                source_column=scope.source_field,
                target_table=scope.target_table,
                target_column=scope.target_field,
            )
        )

    return SqlRiskScopeContract(
        scope=scope,
        aspects=(aspect,),
        requirements=tuple(requirements),
        execution_mode=execution_mode,
    )


def _requirement_sequence(
    contract_or_requirements: (
        SqlRiskScopeContract
        | Iterable[SqlRiskEvidenceRequirement]
        | None
    ),
) -> tuple[SqlRiskEvidenceRequirement, ...]:
    if contract_or_requirements is None:
        return ()
    if isinstance(contract_or_requirements, SqlRiskScopeContract):
        return contract_or_requirements.requirements
    return tuple(contract_or_requirements)


def _call_name_and_arguments(
    call: Mapping[str, Any] | object,
) -> tuple[str | None, Mapping[str, Any] | None, bool]:
    """Read accepted-evidence and tool-call shapes without coupling to them."""

    if isinstance(call, Mapping):
        name = call.get("tool_name", call.get("name"))
        arguments = call.get(
            "args",
            call.get("compact_args", call.get("arguments")),
        )
        truncated = bool(call.get("truncated", False))
    else:
        name = getattr(call, "tool_name", getattr(call, "name", None))
        arguments = getattr(
            call,
            "compact_args",
            getattr(call, "args", getattr(call, "arguments", None)),
        )
        truncated = bool(getattr(call, "truncated", False))
    return (
        str(name) if name is not None else None,
        arguments if isinstance(arguments, Mapping) else None,
        truncated,
    )


def missing_sql_risk_requirements(
    contract_or_requirements: (
        SqlRiskScopeContract
        | Iterable[SqlRiskEvidenceRequirement]
        | None
    ),
    calls: Iterable[Mapping[str, Any] | object],
) -> tuple[SqlRiskEvidenceRequirement, ...]:
    """Return exact requirements not satisfied by accepted tool evidence.

    A requirement matches only the same tool name and byte/value-equivalent
    argument mapping.  Truncated evidence never satisfies the contract.  The
    function understands both upstream payload mappings (``args``) and
    ``EvidenceArtifact``-like objects (``compact_args``), while remaining
    independent from runtime model classes.
    """

    requirements = _requirement_sequence(contract_or_requirements)
    observed = tuple(_call_name_and_arguments(call) for call in calls)
    return tuple(
        requirement
        for requirement in requirements
        if not any(
            name == requirement.tool_name
            and not truncated
            and dict(arguments or {}) == dict(requirement.arguments)
            for name, arguments, truncated in observed
        )
    )


__all__ = [
    "GetSourceTargetColumnPairArguments",
    "GetSourceTargetColumnPairRequirement",
    "ListS2TFieldMappingArguments",
    "ListS2TFieldMappingRequirement",
    "SqlRiskExactScope",
    "OPERATION_SQL_RISK_SCOPE_EVIDENCE_EXPERIMENT_ENV",
    "ReadS2TSourceToTargetArguments",
    "ReadS2TSourceToTargetRequirement",
    "SqlRiskEvidenceRequirement",
    "SqlRiskScopeContract",
    "SqlRiskToolName",
    "build_sql_risk_scope_contract",
    "missing_sql_risk_requirements",
    "sql_risk_scope_evidence_enabled",
]
