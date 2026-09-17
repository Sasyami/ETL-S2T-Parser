"""Exact-reader orchestration for the closed SQL-risk scope pipeline.

The module is intentionally independent from :mod:`agents.coordinator`.  It
accepts an already structured :class:`SqlRiskScopeContract`, executes only the
exact readers declared by that contract and materializes their complete
results in the run-scoped saved-result store. SQLGlot emits a neutral structural
bundle; a caller-supplied bounded LLM stage owns the semantic risk conclusion.

The closed modes cover row filtering, conditional cardinality, nullable
constraints, value changes and write semantics. Reader outages,
malformed/partial payloads and missing data are returned as a structured
``unavailable`` result; they never trigger a broader read or an agentic
fallback here.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from time import perf_counter
from types import MappingProxyType
from typing import Any, Callable, Dict, List, Literal, Mapping, Sequence
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from .chat_graph import WorkerDisplayItem
from .contracts import EvidenceArtifact
from .sql_risk_assessment import (
    SqlRiskAssessmentContext,
    SqlRiskAssessmentEndpoint,
    SqlRiskAssessmentEvidence,
    SqlRiskAssessmentScope,
    SqlRiskAssessmentValidation,
    SqlRiskStructuralRule,
)
from .sql_risk_scope_contract import (
    GetSourceTargetColumnPairRequirement,
    ListS2TFieldMappingRequirement,
    ReadS2TSourceToTargetRequirement,
    SqlRiskScopeContract,
    SqlRiskToolName,
    missing_sql_risk_requirements,
)
from .tools.columns import get_source_target_column_pair
from .tools.s2t import list_s2t_field_mapping, read_s2t_source_to_target
from .tools.saved_results import (
    SavedResultStore,
    get_active_saved_result_store,
)
from .sql_risk_structure import (
    SqlRiskStructuralEvidence,
    build_sql_risk_structure_bundle,
    sql_risk_structure_assessment_payload,
)


SqlRiskOperationMode = Literal[
    "row_filtering",
    "conditional_cardinality",
    "nullable_constraint",
    "value_changes",
    "write_semantics",
]
SqlRiskOperationReportedMode = SqlRiskOperationMode | Literal["unresolved"]
SqlRiskOperationStatus = Literal["complete", "unavailable"]
SqlRiskOperationAnswerSource = Literal[
    "sql_risk_scope_llm",
    "sql_risk_scope_unavailable",
]
SqlRiskOperationIssueCode = Literal[
    "invalid_contract",
    "store_unavailable",
    "reader_unavailable",
    "reader_error",
    "invalid_reader_payload",
    "persistence_error",
    "unpersistable_payload",
    "incomplete_evidence",
    "structure_error",
    "analysis_unavailable",
]

SQL_RISK_SCOPE_PIPELINE = "sql_risk_scope"
MAX_SQL_RISK_OPERATION_MESSAGE_CHARS = 600
MAX_SQL_RISK_OPERATION_PREVIEW_CHARS = 6000
_MAX_IDENTIFIER_CHARS = 200

_MAPPING_CARDINALITY_COLUMNS = (
    "source_table",
    "target_table",
    "transformation_rule",
)
_MAPPING_NULLABLE_COLUMNS = (
    "source_table",
    "source_field",
    "target_table",
    "target_field",
    "transformation_rule",
)
_METADATA_NULLABLE_COLUMNS = (
    "column_role",
    "file_id",
    "table_name",
    "column_name",
    "not_null",
)

_DEFAULT_READERS: Mapping[str, Any] = MappingProxyType(
    {
        "read_s2t_source_to_target": read_s2t_source_to_target,
        "list_s2t_field_mapping": list_s2t_field_mapping,
        "get_source_target_column_pair": get_source_target_column_pair,
    }
)


class SqlRiskOperationReadSpec(BaseModel):
    """One ordered exact read owned by the direct pipeline."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    tool_name: SqlRiskToolName
    arguments: Dict[str, Any]
    required_columns: List[str]
    display_on_complete: bool = False


class SqlRiskOperationSpec(BaseModel):
    """Pure validated execution specification for one closed contract."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    pipeline: Literal["sql_risk_scope"] = SQL_RISK_SCOPE_PIPELINE
    execution_mode: SqlRiskOperationMode
    scope: str
    reads: List[SqlRiskOperationReadSpec]


class SqlRiskOperationIssue(BaseModel):
    """Bounded machine-readable reason why the direct lane is unavailable."""

    model_config = ConfigDict(extra="forbid")

    code: SqlRiskOperationIssueCode
    message: str = Field(min_length=1, max_length=MAX_SQL_RISK_OPERATION_MESSAGE_CHARS)
    tool_name: str | None = None
    requirement_index: int | None = Field(default=None, ge=1)


class SqlRiskOperationReadRecord(BaseModel):
    """Bounded read metric and runtime provenance for coordinator integration."""

    model_config = ConfigDict(extra="forbid")

    pipeline: Literal["sql_risk_scope"] = SQL_RISK_SCOPE_PIPELINE
    requirement_index: int = Field(ge=1)
    tool_name: SqlRiskToolName
    arguments: Dict[str, Any]
    call_id: str
    status: SqlRiskOperationStatus
    elapsed_seconds: float = Field(default=0.0, ge=0.0)
    row_count: int | None = Field(default=None, ge=0)
    source_total: int | None = Field(default=None, ge=0)
    truncated: bool = False
    evidence_id: str | None = None
    dataset_ref: str | None = Field(default=None, exclude=True)
    issue_code: SqlRiskOperationIssueCode | None = None


class SqlRiskOperationPipelineResult(BaseModel):
    """Complete or fail-closed outcome of the direct SQL-risk pipeline."""

    model_config = ConfigDict(extra="forbid")

    pipeline: Literal["sql_risk_scope"] = SQL_RISK_SCOPE_PIPELINE
    status: SqlRiskOperationStatus
    execution_mode: SqlRiskOperationReportedMode
    scope: str
    answer: str = Field(min_length=1)
    answer_source: SqlRiskOperationAnswerSource
    facts: List[Dict[str, Any]] = Field(default_factory=list)
    fact_payload: Dict[str, Any] = Field(default_factory=dict)
    evidence: List[EvidenceArtifact] = Field(default_factory=list)
    used_evidence_ids: List[str] = Field(default_factory=list)
    display_evidence_ids: List[str] = Field(default_factory=list)
    display_items: List[WorkerDisplayItem] = Field(default_factory=list)
    read_records: List[SqlRiskOperationReadRecord] = Field(default_factory=list)
    issues: List[SqlRiskOperationIssue] = Field(default_factory=list)

    def metrics_payload(self) -> Dict[str, Any]:
        """Return bounded records without raw results or runtime dataset refs."""

        return {
            "pipeline": self.pipeline,
            "status": self.status,
            "execution_mode": self.execution_mode,
            "scope": self.scope,
            "answer_source": self.answer_source,
            "reads": [
                item.model_dump(mode="json") for item in self.read_records
            ],
            "issues": [item.model_dump(mode="json") for item in self.issues],
            "facts": list(self.facts),
        }


@dataclass(frozen=True)
class _SuccessfulRead:
    artifact: EvidenceArtifact
    payload: Dict[str, Any]
    call_id: str
    display_on_complete: bool


@dataclass(frozen=True)
class _ReadExecution:
    record: SqlRiskOperationReadRecord
    successful: _SuccessfulRead | None = None
    issue: SqlRiskOperationIssue | None = None


def _bounded_message(value: object) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip() or "unknown error"
    if len(text) <= MAX_SQL_RISK_OPERATION_MESSAGE_CHARS:
        return text
    return text[: MAX_SQL_RISK_OPERATION_MESSAGE_CHARS - 1].rstrip() + "…"


def _identifiers_are_bounded(contract: SqlRiskScopeContract) -> bool:
    scope = contract.scope
    values = (
        scope.source,
        scope.target,
        scope.source_table,
        scope.target_table,
        *(value for value in (scope.source_field, scope.target_field) if value),
    )
    return all(
        bool(str(value or "").strip())
        and len(str(value)) <= _MAX_IDENTIFIER_CHARS
        for value in values
    )


def _nullable_contract_is_exact(contract: SqlRiskScopeContract) -> bool:
    scope = contract.scope
    valid_file_id = (
        isinstance(scope.file_id, int)
        and not isinstance(scope.file_id, bool)
        and scope.file_id > 0
    )
    if (
        contract.execution_mode != "nullable_constraint"
        or contract.aspects != ("constraint_rejection",)
        or not _identifiers_are_bounded(contract)
        or not scope.is_field_pair
        or not valid_file_id
        or scope.source_field is None
        or scope.target_field is None
        or scope.source != f"{scope.source_table}.{scope.source_field}"
        or scope.target != f"{scope.target_table}.{scope.target_field}"
        or len(contract.requirements) != 2
        or not isinstance(
            contract.requirements[0],
            ListS2TFieldMappingRequirement,
        )
        or not isinstance(
            contract.requirements[1],
            GetSourceTargetColumnPairRequirement,
        )
    ):
        return False
    mapping = contract.requirements[0]
    metadata = contract.requirements[1]
    return bool(
        mapping.source_table == scope.source_table
        and mapping.target_table == scope.target_table
        and mapping.source_field == scope.source_field
        and mapping.target_field == scope.target_field
        and metadata.file_id == scope.file_id
        and metadata.source_table == scope.source_table
        and metadata.source_column == scope.source_field
        and metadata.target_table == scope.target_table
        and metadata.target_column == scope.target_field
    )


def _single_mapping_contract_is_exact(
    contract: SqlRiskScopeContract,
    *,
    execution_mode: str,
    aspect: str,
    field_scope: bool,
) -> bool:
    """Validate a one-reader contract without inspecting natural language."""

    scope = contract.scope
    if (
        str(contract.execution_mode) != execution_mode
        or tuple(contract.aspects) != (aspect,)
        or not _identifiers_are_bounded(contract)
        or len(contract.requirements) != 1
    ):
        return False
    requirement = contract.requirements[0]
    expected_requirement_type = (
        ListS2TFieldMappingRequirement
        if field_scope
        else ReadS2TSourceToTargetRequirement
    )
    if not isinstance(requirement, expected_requirement_type):
        return False
    if (
        requirement.source_table != scope.source_table
        or requirement.target_table != scope.target_table
    ):
        return False
    if field_scope:
        return bool(
            isinstance(requirement, ListS2TFieldMappingRequirement)
            and scope.is_field_pair
            and scope.source_field
            and scope.target_field
            and requirement.source_field == scope.source_field
            and requirement.target_field == scope.target_field
            and scope.source
            == f"{scope.source_table}.{scope.source_field}"
            and scope.target
            == f"{scope.target_table}.{scope.target_field}"
        )
    return bool(
        not scope.is_field_pair
        and scope.source_field is None
        and scope.target_field is None
        and scope.source == scope.source_table
        and scope.target == scope.target_table
    )


def _operation_spec_from_contract(
    contract: SqlRiskScopeContract,
) -> SqlRiskOperationSpec | None:
    if _single_mapping_contract_is_exact(
        contract,
        execution_mode="row_filtering",
        aspect="row_filtering",
        field_scope=False,
    ):
        requirement = contract.requirements[0]
        return SqlRiskOperationSpec(
            execution_mode="row_filtering",
            scope=contract.scope.label,
            reads=[
                SqlRiskOperationReadSpec(
                    tool_name=requirement.tool_name,
                    arguments=dict(requirement.arguments),
                    required_columns=list(_MAPPING_CARDINALITY_COLUMNS),
                    display_on_complete=True,
                )
            ],
        )

    if _single_mapping_contract_is_exact(
        contract,
        execution_mode="conditional_cardinality",
        aspect="cardinality",
        field_scope=False,
    ):
        requirement = contract.requirements[0]
        return SqlRiskOperationSpec(
            execution_mode="conditional_cardinality",
            scope=contract.scope.label,
            reads=[
                SqlRiskOperationReadSpec(
                    tool_name=requirement.tool_name,
                    arguments=dict(requirement.arguments),
                    required_columns=list(_MAPPING_CARDINALITY_COLUMNS),
                    display_on_complete=True,
                )
            ],
        )
    if _nullable_contract_is_exact(contract):
        mapping, metadata = contract.requirements
        return SqlRiskOperationSpec(
            execution_mode="nullable_constraint",
            scope=contract.scope.label,
            reads=[
                SqlRiskOperationReadSpec(
                    tool_name=mapping.tool_name,
                    arguments=dict(mapping.arguments),
                    required_columns=list(_MAPPING_NULLABLE_COLUMNS),
                    display_on_complete=True,
                ),
                SqlRiskOperationReadSpec(
                    tool_name=metadata.tool_name,
                    arguments=dict(metadata.arguments),
                    required_columns=list(_METADATA_NULLABLE_COLUMNS),
                    display_on_complete=True,
                ),
            ],
        )
    if _single_mapping_contract_is_exact(
        contract,
        execution_mode="value_changes",
        aspect="value_changes",
        field_scope=True,
    ):
        requirement = contract.requirements[0]
        return SqlRiskOperationSpec(
            execution_mode="value_changes",
            scope=contract.scope.label,
            reads=[
                SqlRiskOperationReadSpec(
                    tool_name=requirement.tool_name,
                    arguments=dict(requirement.arguments),
                    required_columns=list(_MAPPING_NULLABLE_COLUMNS),
                    display_on_complete=True,
                )
            ],
        )
    if _single_mapping_contract_is_exact(
        contract,
        execution_mode="write_semantics",
        aspect="write_semantics",
        field_scope=False,
    ):
        requirement = contract.requirements[0]
        return SqlRiskOperationSpec(
            execution_mode="write_semantics",
            scope=contract.scope.label,
            reads=[
                SqlRiskOperationReadSpec(
                    tool_name=requirement.tool_name,
                    arguments=dict(requirement.arguments),
                    required_columns=list(_MAPPING_CARDINALITY_COLUMNS),
                    display_on_complete=True,
                )
            ],
        )
    return None


def operation_spec_from_contract(
    contract: SqlRiskScopeContract,
) -> SqlRiskOperationSpec | None:
    """Validate a structured contract and return its fixed reader plan.

    This boundary never examines natural language and never repairs a forged or
    stale contract. Invalid, malformed or unsupported contracts return ``None``
    before any reader can run.
    """

    if not isinstance(contract, SqlRiskScopeContract):
        return None
    try:
        return _operation_spec_from_contract(contract)
    except Exception:
        return None


def _issue(
    code: SqlRiskOperationIssueCode,
    message: object,
    *,
    tool_name: str | None = None,
    requirement_index: int | None = None,
) -> SqlRiskOperationIssue:
    return SqlRiskOperationIssue(
        code=code,
        message=_bounded_message(message),
        tool_name=tool_name,
        requirement_index=requirement_index,
    )


def _safe_mode(contract: object) -> SqlRiskOperationReportedMode:
    try:
        value = str(
            getattr(contract, "execution_mode", "unresolved") or "unresolved"
        )
    except Exception:
        return "unresolved"
    if value in {
        "row_filtering",
        "conditional_cardinality",
        "nullable_constraint",
        "value_changes",
        "write_semantics",
    }:
        return value  # type: ignore[return-value]
    return "unresolved"


def _scope_label(contract: object) -> str:
    try:
        scope = getattr(contract, "scope", None)
        label = str(getattr(scope, "label", "") or "").strip()
    except Exception:
        label = ""
    return label or "unknown exact scope"


def _unavailable_answer(scope: str) -> str:
    return (
        f"Для `{scope}` оценка SQL-риска недоступна: обязательные "
        "exact read-only данные не получены или не прошли проверку "
        "полноты. "
        "Сделать подтверждённый вывод о риске нельзя."
    )


def _unavailable_result(
    contract: object,
    *,
    spec: SqlRiskOperationSpec | None = None,
    answer: str | None = None,
    facts: Sequence[Mapping[str, Any]] = (),
    fact_payload: Mapping[str, Any] | None = None,
    evidence: Sequence[EvidenceArtifact] = (),
    used_evidence_ids: Sequence[str] = (),
    read_records: Sequence[SqlRiskOperationReadRecord] = (),
    issues: Sequence[SqlRiskOperationIssue],
) -> SqlRiskOperationPipelineResult:
    scope = spec.scope if spec is not None else _scope_label(contract)
    return SqlRiskOperationPipelineResult(
        status="unavailable",
        execution_mode=(
            spec.execution_mode if spec is not None else _safe_mode(contract)
        ),
        scope=scope,
        answer=(str(answer or "").strip() or _unavailable_answer(scope)),
        answer_source="sql_risk_scope_unavailable",
        facts=[dict(item) for item in facts],
        fact_payload=dict(fact_payload or {}),
        evidence=list(evidence),
        used_evidence_ids=list(dict.fromkeys(used_evidence_ids)),
        read_records=list(read_records),
        issues=list(issues),
    )


def _namespace(value: str | None) -> str:
    raw = str(value or "").strip() or uuid4().hex
    clean = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw).strip("_.-")
    return (clean or uuid4().hex)[:80]


def _invoke_reader(
    reader: Any,
    arguments: Mapping[str, Any],
    callbacks: Sequence[Any],
) -> Any:
    if hasattr(reader, "invoke"):
        config = {"callbacks": list(callbacks)} if callbacks else None
        return (
            reader.invoke(dict(arguments), config=config)
            if config is not None
            else reader.invoke(dict(arguments))
        )
    if callable(reader):
        return reader(**dict(arguments))
    raise TypeError("configured reader is not callable")


def _sql_literal(value: object) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _saved_scope_error(
    read_spec: SqlRiskOperationReadSpec,
    *,
    result_ref: str,
    row_count: int,
    store: SavedResultStore,
) -> str | None:
    """Verify that an exact-reader relation contains no out-of-scope rows."""

    args = read_spec.arguments
    if read_spec.tool_name == "read_s2t_source_to_target":
        predicate = " AND ".join(
            (
                "LOWER(source_table) = LOWER("
                + _sql_literal(args.get("source_table"))
                + ")",
                "LOWER(target_table) = LOWER("
                + _sql_literal(args.get("target_table"))
                + ")",
            )
        )
    elif read_spec.tool_name == "list_s2t_field_mapping":
        predicate = " AND ".join(
            (
                "LOWER(source_table) = LOWER("
                + _sql_literal(args.get("source_table"))
                + ")",
                "LOWER(source_field) = LOWER("
                + _sql_literal(args.get("source_field"))
                + ")",
                "LOWER(target_table) = LOWER("
                + _sql_literal(args.get("target_table"))
                + ")",
                "LOWER(target_field) = LOWER("
                + _sql_literal(args.get("target_field"))
                + ")",
            )
        )
    elif read_spec.tool_name == "get_source_target_column_pair":
        file_id = args.get("file_id")
        if (
            not isinstance(file_id, int)
            or isinstance(file_id, bool)
            or file_id <= 0
        ):
            return "Exact metadata reader has an invalid file_id."
        source = " AND ".join(
            (
                "LOWER(column_role) = 'source'",
                "LOWER(table_name) = LOWER("
                + _sql_literal(args.get("source_table"))
                + ")",
                "LOWER(column_name) = LOWER("
                + _sql_literal(args.get("source_column"))
                + ")",
            )
        )
        target = " AND ".join(
            (
                "LOWER(column_role) = 'target'",
                "LOWER(table_name) = LOWER("
                + _sql_literal(args.get("target_table"))
                + ")",
                "LOWER(column_name) = LOWER("
                + _sql_literal(args.get("target_column"))
                + ")",
            )
        )
        predicate = f"file_id = {file_id} AND (({source}) OR ({target}))"
    else:  # pragma: no cover - closed SqlRiskToolName union
        return "Unsupported exact-reader scope."

    try:
        checked = store.query(
            result_ref=result_ref,
            query=(
                "SELECT COUNT(*) AS matching_rows FROM result WHERE "
                + predicate
            ),
            preview_limit=1,
        )
    except Exception as exc:
        return (
            "Saved exact-reader scope verification failed: "
            f"{type(exc).__name__}: {_bounded_message(exc)}"
        )
    rows = checked.get("rows", [])
    if (
        checked.get("error")
        or checked.get("input_truncated")
        or checked.get("truncated")
        or len(rows) != 1
        or not isinstance(rows[0], Mapping)
    ):
        return "Saved exact-reader scope could not be verified."
    try:
        matching_rows = int(rows[0].get("matching_rows") or 0)
    except (TypeError, ValueError):
        return "Saved exact-reader scope count is invalid."
    if matching_rows != row_count:
        return "Saved exact-reader relation contains out-of-scope rows."
    return None


def _failed_read(
    read_spec: SqlRiskOperationReadSpec,
    *,
    index: int,
    call_id: str,
    started_at: float,
    code: SqlRiskOperationIssueCode,
    message: object,
    row_count: int | None = None,
    source_total: int | None = None,
    truncated: bool = False,
    dataset_ref: str | None = None,
) -> _ReadExecution:
    issue = _issue(
        code,
        message,
        tool_name=read_spec.tool_name,
        requirement_index=index,
    )
    return _ReadExecution(
        record=SqlRiskOperationReadRecord(
            requirement_index=index,
            tool_name=read_spec.tool_name,
            arguments=dict(read_spec.arguments),
            call_id=call_id,
            status="unavailable",
            elapsed_seconds=perf_counter() - started_at,
            row_count=row_count,
            source_total=source_total,
            truncated=truncated,
            dataset_ref=dataset_ref,
            issue_code=issue.code,
        ),
        issue=issue,
    )


def _execute_read(
    read_spec: SqlRiskOperationReadSpec,
    *,
    index: int,
    namespace: str,
    store: SavedResultStore,
    reader_registry: Mapping[str, Any],
    callbacks: Sequence[Any],
) -> _ReadExecution:
    call_id = f"sql_risk_scope_{namespace}_{index}"
    evidence_id = f"evidence_sql_risk_scope_{namespace}_{index}"
    started_at = perf_counter()
    reader = reader_registry.get(read_spec.tool_name)
    if reader is None:
        return _failed_read(
            read_spec,
            index=index,
            call_id=call_id,
            started_at=started_at,
            code="reader_unavailable",
            message=f"Required reader {read_spec.tool_name!r} is unavailable.",
        )
    try:
        is_tool_object = hasattr(reader, "invoke")
        reader_name = (
            str(getattr(reader, "name", "") or "")
            if is_tool_object
            else read_spec.tool_name
        )
    except Exception as exc:
        return _failed_read(
            read_spec,
            index=index,
            call_id=call_id,
            started_at=started_at,
            code="reader_unavailable",
            message=(
                "Configured reader identity could not be verified: "
                f"{type(exc).__name__}: {exc}"
            ),
        )
    # For plain callables the registry key is the explicit logical-name
    # adapter. BaseTool-like objects additionally have to attest the same name.
    if is_tool_object and reader_name != read_spec.tool_name:
        return _failed_read(
            read_spec,
            index=index,
            call_id=call_id,
            started_at=started_at,
            code="reader_unavailable",
            message=(
                "Configured tool identity does not match the required exact "
                f"reader {read_spec.tool_name!r}."
            ),
        )
    try:
        raw_payload = _invoke_reader(
            reader,
            read_spec.arguments,
            callbacks,
        )
    except Exception as exc:
        return _failed_read(
            read_spec,
            index=index,
            call_id=call_id,
            started_at=started_at,
            code="reader_error",
            message=f"{type(exc).__name__}: {exc}",
        )
    if not isinstance(raw_payload, Mapping):
        return _failed_read(
            read_spec,
            index=index,
            call_id=call_id,
            started_at=started_at,
            code="invalid_reader_payload",
            message=f"{read_spec.tool_name} returned a non-object payload.",
        )
    try:
        payload = dict(raw_payload)
        payload_error = payload.get("error")
    except Exception as exc:
        return _failed_read(
            read_spec,
            index=index,
            call_id=call_id,
            started_at=started_at,
            code="invalid_reader_payload",
            message=(
                "Reader payload could not be canonicalized: "
                f"{type(exc).__name__}: {exc}"
            ),
        )
    if payload_error:
        return _failed_read(
            read_spec,
            index=index,
            call_id=call_id,
            started_at=started_at,
            code="reader_error",
            message=payload_error,
        )

    try:
        descriptor = store.save_payload(
            source_tool=read_spec.tool_name,
            source_tool_call_id=call_id,
            payload=payload,
        )
    except Exception as exc:
        return _failed_read(
            read_spec,
            index=index,
            call_id=call_id,
            started_at=started_at,
            code="persistence_error",
            message=f"{type(exc).__name__}: {exc}",
        )
    if descriptor is None:
        return _failed_read(
            read_spec,
            index=index,
            call_id=call_id,
            started_at=started_at,
            code="unpersistable_payload",
            message=f"{read_spec.tool_name} did not return a tabular result.",
        )
    try:
        persisted_descriptor = store.descriptor(descriptor.result_ref)
    except Exception as exc:
        return _failed_read(
            read_spec,
            index=index,
            call_id=call_id,
            started_at=started_at,
            code="persistence_error",
            message=(
                "Persisted descriptor could not be verified: "
                f"{type(exc).__name__}: {exc}"
            ),
        )
    if persisted_descriptor is None:
        return _failed_read(
            read_spec,
            index=index,
            call_id=call_id,
            started_at=started_at,
            code="persistence_error",
            message="Saved result descriptor is missing from the active store.",
        )
    descriptor = persisted_descriptor

    available_columns = {
        column.name.casefold() for column in descriptor.columns
    }
    missing_columns = [
        column
        for column in read_spec.required_columns
        if column.casefold() not in available_columns
    ]
    complete_source = bool(
        descriptor.source_tool == read_spec.tool_name
        and descriptor.source_tool_call_id == call_id
        and not descriptor.truncated
        and not descriptor.input_truncated
        and descriptor.source_total is not None
        and descriptor.source_total == descriptor.row_count
        and not missing_columns
    )
    if not complete_source:
        detail = "Saved exact evidence is incomplete"
        if missing_columns:
            detail += "; missing columns: " + ", ".join(missing_columns)
        return _failed_read(
            read_spec,
            index=index,
            call_id=call_id,
            started_at=started_at,
            code="incomplete_evidence",
            message=detail,
            row_count=descriptor.row_count,
            source_total=descriptor.source_total,
            truncated=descriptor.truncated,
            dataset_ref=descriptor.result_ref,
        )

    scope_error = _saved_scope_error(
        read_spec,
        result_ref=descriptor.result_ref,
        row_count=descriptor.row_count,
        store=store,
    )
    if scope_error is not None:
        return _failed_read(
            read_spec,
            index=index,
            call_id=call_id,
            started_at=started_at,
            code="incomplete_evidence",
            message=scope_error,
            row_count=descriptor.row_count,
            source_total=descriptor.source_total,
            truncated=descriptor.truncated,
            dataset_ref=descriptor.result_ref,
        )

    artifact = EvidenceArtifact(
        evidence_id=evidence_id,
        tool_name=read_spec.tool_name,
        compact_args=dict(read_spec.arguments),
        truncated=False,
        dataset_ref=descriptor.result_ref,
    )
    return _ReadExecution(
        record=SqlRiskOperationReadRecord(
            requirement_index=index,
            tool_name=read_spec.tool_name,
            arguments=dict(read_spec.arguments),
            call_id=call_id,
            status="complete",
            elapsed_seconds=perf_counter() - started_at,
            row_count=descriptor.row_count,
            source_total=descriptor.source_total,
            truncated=False,
            evidence_id=evidence_id,
            dataset_ref=descriptor.result_ref,
        ),
        successful=_SuccessfulRead(
            artifact=artifact,
            payload=payload,
            call_id=call_id,
            display_on_complete=read_spec.display_on_complete,
        ),
    )


def _display_item(
    read: _SuccessfulRead,
) -> WorkerDisplayItem:
    artifact = read.artifact
    content = json.dumps(
        read.payload,
        ensure_ascii=False,
        default=str,
        separators=(",", ":"),
    )
    preview = content[:MAX_SQL_RISK_OPERATION_PREVIEW_CHARS]
    if len(content) > MAX_SQL_RISK_OPERATION_PREVIEW_CHARS:
        preview = preview[:-1].rstrip() + "…"
    return WorkerDisplayItem(
        name=artifact.tool_name,
        content=content,
        evidence_id=artifact.evidence_id,
        tool_call_id=read.call_id,
        arguments=dict(artifact.compact_args),
        preview=preview,
        truncated=artifact.truncated,
    )


def run_sql_risk_operation_pipeline(
    contract: SqlRiskScopeContract,
    *,
    original_task: str,
    stable_context: str = "",
    assessment_runner: Callable[
        [SqlRiskAssessmentContext], SqlRiskAssessmentValidation
    ] | None,
    store: SavedResultStore | None = None,
    readers: Mapping[str, Any] | None = None,
    callbacks: Sequence[Any] = (),
    evidence_namespace: str | None = None,
) -> SqlRiskOperationPipelineResult:
    """Execute exact reads, compile neutral structure and ask the scope LLM.

    ``readers`` is an optional complete registry used for dependency injection;
    when omitted, the production exact-reader tools are used.  The caller must
    provide a saved-result store or run inside ``saved_result_store_scope`` so
    that structural inputs remain complete and run-scoped. Semantic assessment
    is accepted only through ``assessment_runner`` and its validated typed
    provenance result; there is no deterministic answer or agentic fallback.
    """

    spec = operation_spec_from_contract(contract)
    if spec is None:
        return _unavailable_result(
            contract,
            issues=[
                _issue(
                    "invalid_contract",
                    "Unsupported or inconsistent structured SQL-risk contract.",
                )
            ],
        )

    active_store = store or get_active_saved_result_store()
    if active_store is None:
        return _unavailable_result(
            contract,
            spec=spec,
            issues=[
                _issue(
                    "store_unavailable",
                    "A run-scoped SavedResultStore is required.",
                )
            ],
        )

    reader_registry = _DEFAULT_READERS if readers is None else readers
    # Even an explicitly supplied human-readable namespace is only a prefix;
    # identifiers remain unique when the same operation is invoked twice in
    # one run-scoped store.
    namespace = f"{_namespace(evidence_namespace)}_{uuid4().hex[:12]}"
    successful_reads: List[_SuccessfulRead] = []
    artifacts: List[EvidenceArtifact] = []
    records: List[SqlRiskOperationReadRecord] = []
    issues: List[SqlRiskOperationIssue] = []

    for index, read_spec in enumerate(spec.reads, start=1):
        execution = _execute_read(
            read_spec,
            index=index,
            namespace=namespace,
            store=active_store,
            reader_registry=reader_registry,
            callbacks=callbacks,
        )
        records.append(execution.record)
        if execution.issue is not None:
            issues.append(execution.issue)
        if execution.successful is not None:
            successful_reads.append(execution.successful)
            artifacts.append(execution.successful.artifact)

    if issues:
        return _unavailable_result(
            contract,
            spec=spec,
            evidence=artifacts,
            read_records=records,
            issues=issues,
        )

    missing = missing_sql_risk_requirements(contract, artifacts)
    if missing:
        missing_names = ", ".join(item.tool_name for item in missing)
        return _unavailable_result(
            contract,
            spec=spec,
            evidence=artifacts,
            read_records=records,
            issues=[
                _issue(
                    "incomplete_evidence",
                    "Exact contract requirements remain unsatisfied: "
                    + missing_names,
                )
            ],
        )

    try:
        structural_evidence = [
            SqlRiskStructuralEvidence(
                evidence_id=read.artifact.evidence_id,
                tool_name=read.artifact.tool_name,
                arguments=dict(read.artifact.compact_args),
                payload=dict(read.payload),
            )
            for read in successful_reads
        ]
        bundle = build_sql_risk_structure_bundle(
            contract,
            structural_evidence,
        )
    except Exception as exc:
        return _unavailable_result(
            contract,
            spec=spec,
            evidence=artifacts,
            read_records=records,
            issues=[
                _issue(
                    "structure_error",
                    f"{type(exc).__name__}: {exc}",
                )
            ],
        )
    if bundle.status == "unavailable":
        return _unavailable_result(
            contract,
            spec=spec,
            evidence=artifacts,
            read_records=records,
            issues=[
                _issue(
                    "structure_error",
                    "; ".join(
                        f"{item.code}: {item.message}"
                        for item in bundle.issues
                    ) or "Neutral SQL structure is unavailable.",
                )
            ],
        )

    try:
        bundle_payload = sql_risk_structure_assessment_payload(bundle)
        evidence_summaries = {
            str(item["evidence_id"]): item
            for item in bundle_payload.get("evidence", [])
        }
        metadata_by_evidence: Dict[str, List[Dict[str, Any]]] = {}
        for row in bundle_payload.get("metadata_rows", []):
            metadata_by_evidence.setdefault(
                str(row["evidence_id"]), []
            ).append(row)
        mapping_by_evidence: Dict[str, List[Dict[str, Any]]] = {}
        for row in bundle_payload.get("mapping_rows", []):
            mapping_by_evidence.setdefault(
                str(row["evidence_id"]), []
            ).append(row)
        context = SqlRiskAssessmentContext(
            original_task=str(original_task or "").strip(),
            stable_context=str(stable_context or "").strip(),
            scope=SqlRiskAssessmentScope(
                execution_mode=spec.execution_mode,
                source=SqlRiskAssessmentEndpoint(
                    table_name=contract.scope.source_table,
                    field_name=contract.scope.source_field,
                ),
                target=SqlRiskAssessmentEndpoint(
                    table_name=contract.scope.target_table,
                    field_name=contract.scope.target_field,
                ),
                file_id=contract.scope.file_id,
            ),
            evidence=[
                SqlRiskAssessmentEvidence(
                    evidence_id=read.artifact.evidence_id,
                    tool_name=read.artifact.tool_name,
                    arguments=json.loads(
                        json.dumps(read.artifact.compact_args, default=str)
                    ),
                    content={
                        "reader_summary": evidence_summaries.get(
                            read.artifact.evidence_id,
                            {},
                        ),
                        "metadata_rows": metadata_by_evidence.get(
                            read.artifact.evidence_id,
                            [],
                        ),
                        "mapping_rows": mapping_by_evidence.get(
                            read.artifact.evidence_id,
                            [],
                        ),
                        "mapping_row_count": bundle.mapping_row_count,
                        "mapping_source_total": bundle.mapping_source_total,
                    },
                    required=True,
                    displayable=read.display_on_complete,
                )
                for read in successful_reads
            ],
            structural_rules=[
                SqlRiskStructuralRule(
                    rule_id=str(rule["rule_id"]),
                    evidence_ids=list(rule["evidence_ids"]),
                    structure=rule,
                    required=True,
                )
                for rule in bundle_payload.get("rules", [])
            ],
        )
    except Exception as exc:
        return _unavailable_result(
            contract,
            spec=spec,
            evidence=artifacts,
            read_records=records,
            issues=[
                _issue(
                    "structure_error",
                    f"Assessment context failed validation: {type(exc).__name__}: {exc}",
                )
            ],
        )

    if assessment_runner is None:
        return _unavailable_result(
            contract,
            spec=spec,
            evidence=artifacts,
            read_records=records,
            issues=[
                _issue(
                    "analysis_unavailable",
                    "The required SQL-risk assessment LLM stage is unavailable.",
                )
            ],
        )
    try:
        validation = assessment_runner(context)
    except Exception as exc:
        return _unavailable_result(
            contract,
            spec=spec,
            evidence=artifacts,
            read_records=records,
            issues=[
                _issue(
                    "analysis_unavailable",
                    f"Assessment stage failed: {type(exc).__name__}: {exc}",
                )
            ],
        )
    if not isinstance(validation, SqlRiskAssessmentValidation) or (
        validation.status != "valid" or validation.assessment is None
    ):
        detail = (
            "; ".join(
                f"{item.code}: {item.message}"
                for item in getattr(validation, "issues", [])
            )
            if validation is not None
            else "missing validation result"
        )
        return _unavailable_result(
            contract,
            spec=spec,
            evidence=artifacts,
            read_records=records,
            issues=[
                _issue(
                    "analysis_unavailable",
                    detail or "Assessment output remained invalid after repair.",
                )
            ],
        )

    assessment = validation.assessment
    used_evidence_ids = list(assessment.used_evidence_ids)
    display_evidence_ids = list(assessment.display_evidence_ids)
    try:
        display_items = [
            _display_item(read)
            for read in successful_reads
            if read.artifact.evidence_id in display_evidence_ids
        ]
    except (TypeError, ValueError, OverflowError) as exc:
        return _unavailable_result(
            contract,
            spec=spec,
            evidence=artifacts,
            used_evidence_ids=used_evidence_ids,
            read_records=records,
            issues=[
                _issue(
                    "invalid_reader_payload",
                    "Display payload is not safely serializable: "
                    f"{type(exc).__name__}: {exc}",
                )
            ],
        )
    summary_facts = [
        {
            "assessment_status": assessment.status,
            "outcome": assessment.outcome,
            "limitations": list(assessment.limitations),
            "structure_status": bundle.status,
            "reviewed_rule_ids": list(assessment.reviewed_rule_ids),
        }
    ]
    return SqlRiskOperationPipelineResult(
        status=assessment.status,
        execution_mode=spec.execution_mode,
        scope=spec.scope,
        answer=assessment.answer,
        answer_source="sql_risk_scope_llm",
        facts=summary_facts,
        fact_payload={"sql_risk_assessment": summary_facts[0]},
        evidence=artifacts,
        used_evidence_ids=used_evidence_ids,
        display_evidence_ids=display_evidence_ids,
        display_items=display_items,
        read_records=records,
        issues=[],
    )


__all__ = [
    "MAX_SQL_RISK_OPERATION_MESSAGE_CHARS",
    "MAX_SQL_RISK_OPERATION_PREVIEW_CHARS",
    "SQL_RISK_SCOPE_PIPELINE",
    "SqlRiskOperationAnswerSource",
    "SqlRiskOperationIssue",
    "SqlRiskOperationIssueCode",
    "SqlRiskOperationMode",
    "SqlRiskOperationReportedMode",
    "SqlRiskOperationPipelineResult",
    "SqlRiskOperationReadRecord",
    "SqlRiskOperationReadSpec",
    "SqlRiskOperationSpec",
    "SqlRiskOperationStatus",
    "operation_spec_from_contract",
    "run_sql_risk_operation_pipeline",
]
