"""Opt-in exact-scope contracts for typed SQL-risk operations.

The helpers stay pure while coordinator and worker opt into them through one
environment flag.  Only a single literal technical ``source → target`` pair
from the original task becomes a typed reader requirement.  Missing or
ambiguous literals produce no requirements: this experiment must never resolve
or invent an identifier on the model's behalf.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, ClassVar, Iterable, Literal, Mapping, TypedDict

from .contracts import SqlRiskAspect, SqlRiskExecutionMode


OPERATION_SQL_RISK_SCOPE_EVIDENCE_EXPERIMENT_ENV = (
    "OPERATION_SQL_RISK_SCOPE_EVIDENCE_EXPERIMENT"
)

SqlRiskToolName = Literal[
    "read_s2t_source_to_target",
    "get_source_target_column_pair",
]
SqlRiskScopeEvidenceArchitecture = Literal["off", "typed_plan"]
SqlRiskScopeStage = Literal[
    "plan",
    "planner",
    "observer",
    "upstream_decision",
    "upstream",
]
SqlRiskEndpointKind = Literal["table", "field"]

_SUPPORTED_ASPECTS: tuple[SqlRiskAspect, ...] = (
    "row_filtering",
    "cardinality",
    "constraint_rejection",
    "value_changes",
    "write_semantics",
)
_DISABLED_VALUES = frozenset(
    {"", "0", "false", "no", "off", "disabled", "default", "current"}
)
_TYPED_PLAN_VALUE = "typed_plan"

_IDENTIFIER_ATOM = r"[A-Za-z_][A-Za-z0-9_$]*"
_TECHNICAL_ENDPOINT = rf"{_IDENTIFIER_ATOM}(?:\.{_IDENTIFIER_ATOM}){{0,2}}"
_ARROW_PAIR_RE = re.compile(
    rf"(?<![A-Za-z0-9_$.])"
    rf"`?(?P<source>{_TECHNICAL_ENDPOINT})`?"
    # The opt-in runtime contract accepts only the unambiguous relation
    # glyph.  ASCII ``->`` is also a SQL/JSON operator and must not silently
    # turn an expression into a physical S2T scope.
    rf"\s*→\s*"
    rf"`?(?P<target>{_TECHNICAL_ENDPOINT})`?"
    rf"(?![A-Za-z0-9_$]|\.[A-Za-z_])"
)
_FILE_ID_RE = re.compile(
    r"\bfile_id\s*(?:=|:)\s*(?P<file_id>[1-9][0-9]*)\b",
    re.IGNORECASE,
)
_ENDPOINT_BOUNDARY_CHARS = r"A-Za-z0-9_$\."


class ReadS2TSourceToTargetArguments(TypedDict):
    """Exact arguments of ``read_s2t_source_to_target``."""

    source_table: str
    target_table: str


class GetSourceTargetColumnPairArguments(TypedDict):
    """Exact arguments of ``get_source_target_column_pair``."""

    file_id: int
    source_table: str
    source_column: str
    target_table: str
    target_column: str


@dataclass(frozen=True)
class LiteralSqlRiskScope:
    """One unambiguous literal pair copied from the original task."""

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
    | GetSourceTargetColumnPairRequirement
)


@dataclass(frozen=True)
class SqlRiskScopeContract:
    """Typed, immutable runtime requirements for one literal scope."""

    scope: LiteralSqlRiskScope
    aspects: tuple[SqlRiskAspect, ...]
    requirements: tuple[SqlRiskEvidenceRequirement, ...]
    execution_mode: SqlRiskExecutionMode

    @property
    def tool_names(self) -> tuple[SqlRiskToolName, ...]:
        return tuple(requirement.tool_name for requirement in self.requirements)


def sql_risk_scope_evidence_architecture(
    value: str | None = None,
) -> SqlRiskScopeEvidenceArchitecture:
    """Return the explicitly selected scope/evidence architecture.

    ``typed_plan`` selects deterministic worker-plan synthesis while keeping
    the worker agentic. Disabled values preserve the default path. The
    rejected prompt-mediated experiment and unknown values raise instead of
    silently selecting another mode.

    An explicit ``value`` is useful for pure callers and tests; otherwise the
    environment is read.
    """

    configured = (
        os.getenv(OPERATION_SQL_RISK_SCOPE_EVIDENCE_EXPERIMENT_ENV)
        if value is None
        else value
    )
    normalized = str(configured or "").strip().casefold()
    if normalized == _TYPED_PLAN_VALUE:
        return "typed_plan"
    if normalized in _DISABLED_VALUES:
        return "off"
    raise ValueError(
        "Unknown OPERATION_SQL_RISK_SCOPE_EVIDENCE_EXPERIMENT value: "
        + repr(configured)
    )


def sql_risk_scope_evidence_enabled(value: str | None = None) -> bool:
    """Return whether either isolated scope/evidence architecture is active."""

    return sql_risk_scope_evidence_architecture(value) != "off"


def _endpoint_parts(
    value: str,
    *,
    endpoint_kind: SqlRiskEndpointKind,
) -> tuple[str, str | None]:
    if endpoint_kind == "table" or "." not in value:
        return value, None
    table, field = value.rsplit(".", 1)
    return table, field


def extract_literal_file_id(original_task: str) -> int | None:
    """Extract one unambiguous, explicitly labelled positive ``file_id``."""

    values = {
        int(match.group("file_id"))
        for match in _FILE_ID_RE.finditer(str(original_task or ""))
    }
    if len(values) != 1:
        return None
    return next(iter(values))


def extract_literal_sql_risk_scope(
    original_task: str,
    *,
    endpoint_kind: SqlRiskEndpointKind | None = None,
) -> LiteralSqlRiskScope | None:
    """Extract exactly one distinct literal technical directed pair.

    Both endpoints must use the same supported shape: either ``table`` or
    ``table.field``.  Repeating the same literal pair is harmless, while two
    different pairs are ambiguous and therefore return ``None``.
    """

    matches: dict[tuple[str, str], tuple[str, str]] = {}
    for match in _ARROW_PAIR_RE.finditer(str(original_task or "")):
        source = match.group("source")
        target = match.group("target")
        source_parts = source.split(".")
        target_parts = target.split(".")
        if len(source_parts) != len(target_parts):
            continue
        resolved_kind = endpoint_kind
        if resolved_kind is None:
            resolved_kind = "field" if len(source_parts) > 1 else "table"
        if resolved_kind == "table" and len(source_parts) > 2:
            continue
        if resolved_kind == "field" and len(source_parts) not in {1, 2, 3}:
            continue
        key = (source.casefold(), target.casefold())
        matches.setdefault(key, (source, target))

    if len(matches) != 1:
        return None

    source, target = next(iter(matches.values()))
    resolved_kind = endpoint_kind
    if resolved_kind is None:
        resolved_kind = "field" if "." in source else "table"
    source_table, source_field = _endpoint_parts(
        source,
        endpoint_kind=resolved_kind,
    )
    target_table, target_field = _endpoint_parts(
        target,
        endpoint_kind=resolved_kind,
    )
    return LiteralSqlRiskScope(
        source=source,
        target=target,
        source_table=source_table,
        target_table=target_table,
        source_field=source_field,
        target_field=target_field,
        file_id=extract_literal_file_id(original_task),
    )


def _selected_aspects(
    sql_risk_aspects: Iterable[SqlRiskAspect],
) -> tuple[SqlRiskAspect, ...]:
    requested = set(sql_risk_aspects)
    return tuple(aspect for aspect in _SUPPORTED_ASPECTS if aspect in requested)


def build_sql_risk_scope_contract(
    original_task: str,
    sql_risk_aspects: Iterable[SqlRiskAspect],
    *,
    enabled: bool | None = None,
    execution_mode: SqlRiskExecutionMode = "agentic",
) -> SqlRiskScopeContract | None:
    """Build typed reader requirements, or no contract when unsafe/off.

    Only a router-selected closed execution mode can create a contract.
    Conditional cardinality needs the full directed S2T mapping. Nullable
    constraint compatibility additionally needs exact endpoint metadata and
    one literal ``file_id``. No resolver, keyword matcher or inferred active
    file state is consulted.
    """

    aspects = _selected_aspects(sql_risk_aspects)
    if not aspects or execution_mode == "agentic":
        return None
    is_enabled = (
        sql_risk_scope_evidence_enabled()
        if enabled is None
        else bool(enabled)
    )
    if not is_enabled:
        return None
    typed_mode_spec: dict[
        SqlRiskExecutionMode,
        tuple[tuple[SqlRiskAspect, ...], SqlRiskEndpointKind] | None,
    ] = {
        "agentic": None,
        "conditional_cardinality": (("cardinality",), "table"),
        "nullable_constraint": (("constraint_rejection",), "field"),
    }
    typed_spec = typed_mode_spec[execution_mode]
    # The operation router owns semantic classification. This boundary checks
    # only enum/aspect consistency and literal origin; it never reclassifies
    # natural language with keywords, examples or sentence templates.
    if typed_spec is None or aspects != typed_spec[0]:
        return None
    endpoint_kind = typed_spec[1]
    scope = extract_literal_sql_risk_scope(
        original_task,
        endpoint_kind=endpoint_kind,
    )
    if scope is None:
        return None
    source_depth = scope.source.count(".") + 1
    target_depth = scope.target.count(".") + 1
    if endpoint_kind == "field":
        if source_depth == 1 or target_depth == 1:
            return None
        # Field semantics come from the structured router enum. Only the exact
        # endpoints and file_id are extracted from text.
        if scope.file_id is None:
            return None
    if execution_mode == "nullable_constraint" and (
        not scope.is_field_pair or scope.file_id is None
    ):
        return None

    requirements: list[SqlRiskEvidenceRequirement] = [
        ReadS2TSourceToTargetRequirement(
            source_table=scope.source_table,
            target_table=scope.target_table,
        )
    ]
    if (
        "constraint_rejection" in aspects
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
        aspects=aspects,
        requirements=tuple(requirements),
        execution_mode=execution_mode,
    )


def required_sql_risk_tools(
    original_task: str,
    sql_risk_aspects: Iterable[SqlRiskAspect],
    *,
    enabled: bool | None = None,
    execution_mode: SqlRiskExecutionMode = "agentic",
) -> tuple[SqlRiskToolName, ...]:
    """Return the ordered typed tool names required by the exact contract."""

    contract = build_sql_risk_scope_contract(
        original_task,
        sql_risk_aspects,
        enabled=enabled,
        execution_mode=execution_mode,
    )
    return contract.tool_names if contract is not None else ()


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


def render_sql_risk_scope_contract(
    contract: SqlRiskScopeContract | None,
    *,
    stage: SqlRiskScopeStage,
) -> str:
    """Render a short stage-specific instruction for an already safe scope."""

    if contract is None:
        return ""
    rendered_requirements = "; ".join(
        f"`{requirement.tool_name}`(" 
        + json.dumps(
            requirement.arguments,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + ")"
        for requirement in contract.requirements
    )
    scope = contract.scope.label
    by_stage: Mapping[SqlRiskScopeStage, str] = {
        "plan": (
            f"Exact scope `{scope}`: запланируй одну самодостаточную task "
            f"для обязательных чтений {rendered_requirements}; не добавляй "
            "analysis task."
        ),
        "planner": (
            f"Для exact scope `{scope}` выполни обязательные чтения "
            f"{rendered_requirements} с точными args без narrowing."
        ),
        "observer": (
            f"Для exact scope `{scope}` верни `complete` только когда "
            f"приняты все чтения name+args: {rendered_requirements}."
        ),
        "upstream_decision": (
            f"Для exact scope `{scope}` считай evidence полным только при "
            f"наличии всех name+args: {rendered_requirements}."
        ),
        "upstream": (
            f"Ответ относится строго к `{scope}`; обе точные endpoint-строки "
            "должны присутствовать в публичном ответе."
        ),
    }
    try:
        return by_stage[stage]
    except KeyError as exc:
        raise ValueError(f"Unknown SQL-risk scope stage: {stage!r}") from exc


def _contains_exact_directed_scope(
    answer: str,
    source: str,
    target: str,
) -> bool:
    return bool(
        re.search(
            rf"(?<![{_ENDPOINT_BOUNDARY_CHARS}])"
            rf"`?{re.escape(source)}`?"
            rf"\s*(?:→|->|=>)\s*"
            rf"`?{re.escape(target)}`?"
            rf"(?![{_ENDPOINT_BOUNDARY_CHARS}])",
            answer,
            re.IGNORECASE,
        )
    )


def ensure_sql_risk_answer_scope(
    answer: str,
    original_task: str,
    sql_risk_aspects: Iterable[SqlRiskAspect],
    *,
    enabled: bool | None = None,
    execution_mode: SqlRiskExecutionMode = "agentic",
) -> str:
    """Prepend the exact compact scope unless its directed pair is present.

    The line is derived solely from the original literal pair.  If both exact
    endpoints already occur in the correct source-to-target direction, or if
    the experiment cannot form a safe contract, the answer is returned
    byte-for-byte.  Merely mentioning both endpoints, including in the reverse
    direction, is not sufficient.  Applying the helper twice is idempotent.
    """

    contract = build_sql_risk_scope_contract(
        original_task,
        sql_risk_aspects,
        enabled=enabled,
        execution_mode=execution_mode,
    )
    if contract is None:
        return answer
    scope = contract.scope
    if _contains_exact_directed_scope(answer, scope.source, scope.target):
        return answer

    scope_line = f"Scope: {scope.label}"
    return scope_line if not answer else f"{scope_line}\n\n{answer}"


__all__ = [
    "GetSourceTargetColumnPairArguments",
    "GetSourceTargetColumnPairRequirement",
    "LiteralSqlRiskScope",
    "OPERATION_SQL_RISK_SCOPE_EVIDENCE_EXPERIMENT_ENV",
    "ReadS2TSourceToTargetArguments",
    "ReadS2TSourceToTargetRequirement",
    "SqlRiskEvidenceRequirement",
    "SqlRiskScopeEvidenceArchitecture",
    "SqlRiskScopeContract",
    "SqlRiskScopeStage",
    "SqlRiskToolName",
    "build_sql_risk_scope_contract",
    "ensure_sql_risk_answer_scope",
    "extract_literal_file_id",
    "extract_literal_sql_risk_scope",
    "missing_sql_risk_requirements",
    "render_sql_risk_scope_contract",
    "required_sql_risk_tools",
    "sql_risk_scope_evidence_architecture",
    "sql_risk_scope_evidence_enabled",
]
