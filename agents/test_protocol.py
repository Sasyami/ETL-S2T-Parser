"""Typed deterministic compiler for external Greenplum S2T test protocols."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Literal, Mapping, Optional, Sequence

import sqlglot
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from sqlglot import exp
from sqlglot.errors import SqlglotError

from services.sql_dialects import GREENPLUM_DIALECT  # noqa: F401

from .tools.saved_results import _tabular_payload
from .transformation_ast import (
    NormalizedTransformation,
    normalize_transformation,
    quote_dollar_schemas,
)


ProtocolCheck = Literal[
    "row_count",
    "key_uniqueness",
    "required_null_rate",
    "transformation_correctness",
    "key_reconciliation",
    "missing_rows",
    "extra_rows",
    "field_mismatch",
    "schema_compatibility",
    "expected_required_nulls",
    "aggregate_reconciliation",
    "duplicate_expected",
    "duplicate_actual",
]
ProtocolMode = Literal["explicit", "standard", "exhaustive"]
ProtocolCheckStatus = Literal["ready", "partial", "unavailable"]
ProtocolStatus = Literal["ready", "partial_protocol", "unavailable"]
ProtocolIssueCode = Literal[
    "unresolved_entity",
    "ambiguous_entity",
    "missing_parameter",
    "unsupported_check",
    "partial_protocol",
    "unavailable_check",
]

PROTOCOL_CHECKS: tuple[ProtocolCheck, ...] = (
    "row_count",
    "key_uniqueness",
    "required_null_rate",
    "transformation_correctness",
    "key_reconciliation",
    "missing_rows",
    "extra_rows",
    "field_mismatch",
    "schema_compatibility",
    "expected_required_nulls",
    "aggregate_reconciliation",
    "duplicate_expected",
    "duplicate_actual",
)
STANDARD_PROTOCOL_CHECKS: tuple[ProtocolCheck, ...] = (
    "row_count",
    "key_uniqueness",
    "required_null_rate",
    "transformation_correctness",
)
MAX_PROTOCOL_OBJECTS = 8
SOURCE_SCOPE_PREDICATE = "{{SOURCE_SCOPE_PREDICATE}}"
TARGET_SCOPE_PREDICATE = "{{TARGET_SCOPE_PREDICATE}}"
# Compatibility import for callers that used the old target-only placeholder.
LOAD_SCOPE_PREDICATE = TARGET_SCOPE_PREDICATE
_SIMPLE_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_$]*$")


def _clean_string_list(values: Sequence[str] | None) -> List[str]:
    return list(
        dict.fromkeys(
            str(value).strip()
            for value in values or []
            if str(value).strip()
        )
    )


class RawTestProtocolLoad(BaseModel):
    """Literal source/target mentions for one user-requested load."""

    model_config = ConfigDict(extra="forbid")

    source_mentions: List[str] = Field(
        min_length=1,
        max_length=MAX_PROTOCOL_OBJECTS,
    )
    target_mention: str = Field(min_length=1, max_length=300)
    requested_checks: List[ProtocolCheck] = Field(default_factory=list)
    explicit_key: Optional[List[str]] = None

    @field_validator("source_mentions", "requested_checks", mode="after")
    @classmethod
    def deduplicate_lists(cls, values: List[str]) -> List[str]:
        return _clean_string_list(values)

    @field_validator("explicit_key", mode="before")
    @classmethod
    def normalize_explicit_key(cls, value: Any) -> Any:
        if isinstance(value, str):
            return [value]
        return value


class RawTestProtocolContract(BaseModel):
    """User wording before deterministic file and table resolution."""

    model_config = ConfigDict(extra="forbid")

    file_id: Optional[int] = Field(default=None, gt=0)
    file_mention: Optional[str] = Field(default=None, max_length=500)
    loads: List[RawTestProtocolLoad] = Field(
        min_length=1,
        max_length=MAX_PROTOCOL_OBJECTS,
    )
    requested_checks: List[ProtocolCheck] = Field(default_factory=list)
    mode: ProtocolMode = "explicit"
    explicit_key: Optional[List[str]] = None

    @field_validator("requested_checks", mode="after")
    @classmethod
    def deduplicate_checks(cls, values: List[str]) -> List[str]:
        return _clean_string_list(values)

    @field_validator("explicit_key", mode="before")
    @classmethod
    def normalize_explicit_key(cls, value: Any) -> Any:
        if isinstance(value, str):
            return [value]
        return value

    @model_validator(mode="after")
    def keep_one_file_mention(self) -> "RawTestProtocolContract":
        if self.file_id is not None and str(self.file_mention or "").strip():
            raise ValueError("Укажи только file_id или file_mention.")
        if self.mode == "explicit" and any(
            not (load.requested_checks or self.requested_checks)
            for load in self.loads
        ):
            raise ValueError(
                "explicit mode requires requested_checks for every load"
            )
        return self

    @property
    def source_mentions(self) -> List[str]:
        return list(
            dict.fromkeys(
                value for load in self.loads for value in load.source_mentions
            )
        )

    @property
    def target_mentions(self) -> List[str]:
        return list(dict.fromkeys(load.target_mention for load in self.loads))


class EntityResolutionMetadata(BaseModel):
    """Auditable link from one literal mention to a canonical identifier."""

    model_config = ConfigDict(extra="forbid")

    entity_type: Literal["file", "source_table", "target_table"]
    mention: str
    method: Literal[
        "exact",
        "normalized_exact",
        "partial",
        "fuzzy",
        "semantic",
        "ambiguous",
        "none",
    ]
    status: Literal["resolved", "ambiguous", "unresolved"] = "resolved"
    error_code: Optional[
        Literal["unresolved_entity", "ambiguous_entity"]
    ] = None
    canonical: Optional[str] = None
    candidates: List[str] = Field(default_factory=list)


class ProtocolIssue(BaseModel):
    """Machine-readable reason for a partial or unavailable protocol result."""

    model_config = ConfigDict(extra="forbid")

    code: ProtocolIssueCode
    message: str
    load_index: Optional[int] = Field(default=None, ge=1)
    check: Optional[ProtocolCheck] = None
    candidates: List[str] = Field(default_factory=list)


class TestProtocolLoad(BaseModel):
    """One independently scoped canonical source-set to target validation load."""

    model_config = ConfigDict(extra="forbid")

    sources: List[str] = Field(
        min_length=1,
        max_length=MAX_PROTOCOL_OBJECTS,
    )
    target: str = Field(min_length=1, max_length=300)
    checks: List[ProtocolCheck] = Field(default_factory=list)
    explicit_key: Optional[List[str]] = None

    @field_validator("sources", "checks", mode="after")
    @classmethod
    def deduplicate_lists(cls, values: List[str]) -> List[str]:
        return _clean_string_list(values)

    @field_validator("explicit_key", mode="before")
    @classmethod
    def normalize_explicit_key(cls, value: Any) -> Any:
        if isinstance(value, str):
            return [value]
        return value


class ResolvedTestProtocolContract(BaseModel):
    """Canonical contract consumed by deterministic readers and compilers."""

    model_config = ConfigDict(extra="forbid")

    file_id: Optional[int] = Field(default=None, gt=0)
    filename: Optional[str] = Field(default=None, min_length=1, max_length=500)
    loads: List[TestProtocolLoad] = Field(
        min_length=1,
        max_length=MAX_PROTOCOL_OBJECTS,
    )
    checks: List[ProtocolCheck] = Field(default_factory=list)
    mode: ProtocolMode = "explicit"
    explicit_key: Optional[List[str]] = None
    resolution_metadata: List[EntityResolutionMetadata] = Field(
        default_factory=list
    )

    @field_validator("checks", mode="after")
    @classmethod
    def deduplicate_checks(cls, values: List[str]) -> List[str]:
        return _clean_string_list(values)

    @field_validator("explicit_key", mode="before")
    @classmethod
    def normalize_explicit_key(cls, value: Any) -> Any:
        if isinstance(value, str):
            return [value]
        return value

    @model_validator(mode="after")
    def keep_one_file_selector(self) -> "ResolvedTestProtocolContract":
        if self.file_id is not None and str(self.filename or "").strip():
            raise ValueError("Укажи только file_id или filename.")
        return self

    @property
    def source_tables(self) -> List[str]:
        return list(
            dict.fromkeys(source for load in self.loads for source in load.sources)
        )

    @property
    def target_tables(self) -> List[str]:
        return list(dict.fromkeys(load.target for load in self.loads))

    def checks_for_load(self, load: TestProtocolLoad) -> List[ProtocolCheck]:
        if self.mode == "exhaustive":
            return list(PROTOCOL_CHECKS)
        if self.mode == "standard":
            return list(STANDARD_PROTOCOL_CHECKS)
        return list(load.checks or self.checks)

    def explicit_key_for_load(self, load: TestProtocolLoad) -> List[str]:
        return _clean_string_list(load.explicit_key or self.explicit_key)


class TestProtocolContract(ResolvedTestProtocolContract):
    """Backward-compatible name for the resolved protocol contract."""


class ProtocolPreflightItem(BaseModel):
    """One deterministic phase-zero contract/S2T verification."""

    model_config = ConfigDict(extra="forbid")

    kind: str
    status: Literal["pass", "warning", "fail", "unavailable"]
    conclusion: str
    evidence: List[str] = Field(default_factory=list)


class CompiledProtocolCheck(BaseModel):
    """One non-executed Greenplum validation check."""

    model_config = ConfigDict(extra="forbid")

    kind: ProtocolCheck
    phase: int = Field(default=1, ge=1, le=3)
    status: ProtocolCheckStatus = "ready"
    dependencies: List[str] = Field(default_factory=list)
    missing_dependencies: List[str] = Field(default_factory=list)
    goal: str
    sql_template: str
    pass_criterion: str
    limitations: List[str] = Field(default_factory=list)


class CompiledTargetProtocol(BaseModel):
    """Validation protocol for one exact target table."""

    model_config = ConfigDict(extra="forbid")

    source_tables: List[str]
    target_table: str
    mode: ProtocolMode = "explicit"
    status: ProtocolStatus = "ready"
    comparison_key: List[str] = Field(default_factory=list)
    comparison_key_source: Optional[
        Literal["explicit", "target_primary_key"]
    ] = None
    preflight: List[ProtocolPreflightItem] = Field(default_factory=list)
    checks: List[CompiledProtocolCheck]
    evidence: List[str] = Field(default_factory=list)
    limitations: List[str] = Field(default_factory=list)
    normalized_transformation: Optional[NormalizedTransformation] = None
    s2t_evidence_rows: List[Dict[str, Any]] = Field(
        default_factory=list,
        exclude=True,
    )
    catalog_evidence_rows: List[Dict[str, Any]] = Field(
        default_factory=list,
        exclude=True,
    )
    source_catalog_evidence_rows: List[Dict[str, Any]] = Field(
        default_factory=list,
        exclude=True,
    )


class ProtocolPhaseSummary(BaseModel):
    """Machine-readable phase inventory for execution orchestration."""

    model_config = ConfigDict(extra="forbid")

    phase: int = Field(ge=0, le=3)
    name: str
    item_count: int = Field(ge=0)
    ready_count: int = Field(ge=0)
    partial_count: int = Field(ge=0)
    unavailable_count: int = Field(ge=0)
    failed_count: int = Field(default=0, ge=0)


class CompiledTestProtocol(BaseModel):
    """Complete deterministic protocol for all requested targets."""

    model_config = ConfigDict(extra="forbid")

    mode: ProtocolMode = "explicit"
    status: ProtocolStatus = "ready"
    issues: List[ProtocolIssue] = Field(default_factory=list)
    phases: List[ProtocolPhaseSummary] = Field(default_factory=list)
    targets: List[CompiledTargetProtocol]


def _rows(payload: Any) -> List[Dict[str, Any]]:
    decoded = _tabular_payload(payload)
    if decoded is None:
        return []
    return [dict(row) for row in decoded.get("rows") or []]


def _truthy(value: Any) -> bool:
    if value is True or value == 1:
        return True
    return str(value or "").strip().casefold() in {
        "true",
        "истина",
        "да",
        "yes",
    }


def _quote_identifier(value: str) -> str:
    return '"' + str(value).replace('"', '""') + '"'


def _quote_qualified_name(value: str) -> str:
    parts = [part.strip() for part in str(value).split(".") if part.strip()]
    if not parts:
        raise ValueError("Пустое имя таблицы в typed contract.")
    if any(not _SIMPLE_IDENTIFIER.fullmatch(part) for part in parts):
        raise ValueError(f"Небезопасное имя таблицы: {value}")
    return ".".join(_quote_identifier(part) for part in parts)


def _quote_literal(value: Any) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    return "'" + str(value).replace("'", "''") + "'"


def _parse_full_query(rule: str) -> tuple[exp.Query | None, str | None]:
    """Compatibility helper accepting one complete Greenplum read query."""

    try:
        statements = sqlglot.parse(
            quote_dollar_schemas(str(rule or "").strip()),
            read=GREENPLUM_DIALECT,
        )
    except (SqlglotError, ValueError) as exc:
        return None, (
            "SQLGlot не разобрал transformation_rule: "
            f"{type(exc).__name__}"
        )
    if len(statements) != 1 or not isinstance(statements[0], exp.Query):
        return None, (
            "transformation_rule не является одним полным SELECT/WITH query"
        )
    return statements[0], None


@dataclass
class _ReaderBundle:
    pair_rows: Dict[str, List[Dict[str, Any]]]
    target_rows: List[Dict[str, Any]]
    target_catalog_rows: List[Dict[str, Any]]
    source_catalog_rows: List[Dict[str, Any]]
    target_catalog_available: bool
    source_catalog_available: bool


def _load_results(
    load_index: int,
    target_table: str,
    reader_results: Sequence[Dict[str, Any]],
) -> _ReaderBundle:
    pair_rows: Dict[str, List[Dict[str, Any]]] = {}
    target_rows: List[Dict[str, Any]] = []
    target_catalog_rows: List[Dict[str, Any]] = []
    source_catalog_rows: List[Dict[str, Any]] = []
    target_catalog_available = False
    source_catalog_available = False
    for result in reader_results:
        result_load_index = result.get("load_index")
        if result_load_index is not None and result_load_index != load_index:
            continue
        kind = str(result.get("kind") or "")
        args = dict(result.get("args") or {})
        if kind == "s2t_pair" and args.get("target_table") == target_table:
            source_table = str(args.get("source_table") or "").casefold()
            pair_rows[source_table] = _rows(result.get("payload"))
        elif kind == "s2t_target" and args.get("target_table") == target_table:
            target_rows = _rows(result.get("payload"))
        elif kind in {"target_column_catalog", "target_catalog"} and (
            not args.get("table_name")
            or args.get("table_name") == target_table
        ):
            target_catalog_available = True
            target_catalog_rows.extend(_rows(result.get("payload")))
        elif kind in {"source_column_catalog", "source_catalog"}:
            source_catalog_available = True
            source_catalog_rows.extend(_rows(result.get("payload")))
    for source_name in {
        str(row.get("source_table") or "").strip()
        for row in target_rows
        if str(row.get("source_table") or "").strip()
    }:
        pair_rows.setdefault(
            source_name.casefold(),
            [
                row
                for row in target_rows
                if str(row.get("source_table") or "").strip().casefold()
                == source_name.casefold()
            ],
        )
    return _ReaderBundle(
        pair_rows=pair_rows,
        target_rows=target_rows,
        target_catalog_rows=target_catalog_rows,
        source_catalog_rows=source_catalog_rows,
        target_catalog_available=target_catalog_available,
        source_catalog_available=source_catalog_available,
    )


def _row_rule(row: Mapping[str, Any]) -> str:
    return str(row.get("transformation_rule") or "").strip()


def _select_load_rule(
    sources: Sequence[str],
    pair_rows: Mapping[str, Sequence[Dict[str, Any]]],
) -> tuple[str, List[str], str | None]:
    """Select one complete query shared by every exact requested source pair."""

    missing_sources: List[str] = []
    sources_without_full_query: List[str] = []
    full_rule_sets: List[set[str]] = []
    ordered_rules: List[str] = []
    for source in sources:
        rows = pair_rows.get(source.casefold(), [])
        if not rows:
            missing_sources.append(source)
            continue
        source_rules: set[str] = set()
        for row in rows:
            rule = _row_rule(row)
            if not rule or normalize_transformation(rule).parse_status != "ok":
                continue
            source_rules.add(rule)
            if rule not in ordered_rules:
                ordered_rules.append(rule)
        if not source_rules:
            sources_without_full_query.append(source)
            continue
        full_rule_sets.append(source_rules)
    if missing_sources:
        return "", missing_sources, None
    if sources_without_full_query:
        return (
            "",
            [],
            "Exact source→target строки не содержат полного SELECT/WITH для: "
            + ", ".join(sources_without_full_query)
            + ".",
        )
    common_rules = set.intersection(*full_rule_sets) if full_rule_sets else set()
    selected = [rule for rule in ordered_rules if rule in common_rules]
    if len(selected) != 1:
        return (
            "",
            [],
            "Для exact source→target пар требуется одно общее полное правило "
            f"SELECT; найдено: {len(selected)}.",
        )
    return selected[0], [], None


def _mapping_rows_for_rule(
    rule: str,
    load: TestProtocolLoad,
    bundle: _ReaderBundle,
) -> List[Dict[str, Any]]:
    candidates = bundle.target_rows or [
        row
        for source in load.sources
        for row in bundle.pair_rows.get(source.casefold(), [])
    ]
    return [
        dict(row)
        for row in candidates
        if rule and _row_rule(row) == rule
    ]


def _mapping_pairs(
    rows: Sequence[Mapping[str, Any]],
) -> List[tuple[str, str]]:
    pairs: List[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for row in rows:
        source = str(row.get("source_field") or "").strip()
        target = str(row.get("target_field") or "").strip()
        folded = (source.casefold(), target.casefold())
        if not source or not target or folded in seen:
            continue
        seen.add(folded)
        pairs.append((source, target))
    return pairs


def _projection_plan(
    normalized: NormalizedTransformation,
    pairs: Sequence[tuple[str, str]],
) -> tuple[List[tuple[str, str]], List[str], List[str]]:
    outputs = {name.casefold(): name for name in normalized.projections}
    grouped: Dict[str, Dict[str, Any]] = {}
    for source, target in pairs:
        item = grouped.setdefault(
            target.casefold(),
            {"target": target, "sources": []},
        )
        folded_sources = {value.casefold() for value in item["sources"]}
        if source.casefold() not in folded_sources:
            item["sources"].append(source)
    plan: List[tuple[str, str]] = []
    missing: List[str] = []
    ambiguous: List[str] = []
    for item in grouped.values():
        target = str(item["target"])
        sources = list(item["sources"])
        raw_output = outputs.get(target.casefold())
        if raw_output is None and len(sources) == 1:
            raw_output = outputs.get(sources[0].casefold())
            if raw_output is None and normalized.has_wildcard:
                raw_output = sources[0]
        if raw_output is None and len(sources) > 1:
            ambiguous.append(target)
        elif raw_output is None:
            missing.append(target)
        else:
            plan.append((raw_output, target))
    return plan, missing, ambiguous


def _expected_ctes(
    normalized: NormalizedTransformation,
    projection_plan: Sequence[tuple[str, str]],
    *,
    project_fields: bool = True,
) -> str:
    if project_fields:
        projection = ",\n        ".join(
            f"src.{_quote_identifier(raw)} AS {_quote_identifier(target)}"
            for raw, target in projection_plan
        )
    else:
        projection = "src.*"
    return (
        "WITH expected_raw AS (\n"
        f"{normalized.query_sql}\n"
        "),\nexpected AS (\n"
        "    SELECT\n        "
        f"{projection}\n"
        "    FROM expected_raw AS src\n"
        f"    WHERE {SOURCE_SCOPE_PREDICATE}\n"
        ")"
    )


def _actual_cte(
    target_sql: str,
    fields: Sequence[str],
    *,
    marker: bool = False,
) -> str:
    projections = [f"t.{_quote_identifier(field)}" for field in fields]
    if marker:
        projections.append('TRUE AS "__actual_present"')
    return (
        "actual AS (\n"
        "    SELECT "
        + ", ".join(projections)
        + "\n"
        + f"    FROM {target_sql} AS t\n"
        + f"    WHERE {TARGET_SCOPE_PREDICATE}\n"
        + ")"
    )


def _join_on(
    keys: Sequence[str],
    left: str = "e",
    right: str = "a",
) -> str:
    return " AND ".join(
        f"{left}.{_quote_identifier(key)} IS NOT DISTINCT FROM "
        f"{right}.{_quote_identifier(key)}"
        for key in keys
    )


def _normalized_type(value: Any) -> str:
    text = re.sub(r"\s+", " ", str(value or "").strip().casefold())
    aliases = {
        "int": "integer",
        "int4": "integer",
        "int8": "bigint",
        "decimal": "numeric",
        "character varying": "varchar",
        "bool": "boolean",
    }
    base = re.sub(r"\s*\(.*\)$", "", text)
    return aliases.get(base, base)


def _types_compatible(source_type: str, target_type: str) -> bool:
    source = _normalized_type(source_type)
    target = _normalized_type(target_type)
    if not source or not target:
        return False
    if source == target:
        return True
    widening = {
        "smallint": {"integer", "bigint", "numeric"},
        "integer": {"bigint", "numeric"},
        "bigint": {"numeric"},
        "char": {"varchar", "text"},
        "varchar": {"text"},
        "date": {"timestamp", "timestamp without time zone"},
    }
    return target in widening.get(source, set())


@dataclass
class _CompilerContext:
    target_sql: str
    normalized: NormalizedTransformation
    projection_plan: List[tuple[str, str]]
    target_fields: List[str]
    comparison_key: List[str]
    comparison_key_source: Optional[str]
    required_fields: List[str]
    target_catalog_available: bool
    source_catalog_available: bool
    schema_rows: List[Dict[str, Any]]
    schema_missing: List[str]

    @property
    def transformation_ready(self) -> bool:
        return self.normalized.parse_status == "ok"

    @property
    def projection_ready(self) -> bool:
        return self.transformation_ready and bool(self.projection_plan)

    @property
    def projected_fields(self) -> set[str]:
        return {target.casefold() for _, target in self.projection_plan}

    @property
    def expected_ctes(self) -> str:
        return _expected_ctes(self.normalized, self.projection_plan)

    @property
    def expected_raw_ctes(self) -> str:
        return _expected_ctes(self.normalized, (), project_fields=False)


CheckCompiler = Callable[
    [ProtocolCheck, "CheckDefinition", _CompilerContext],
    CompiledProtocolCheck,
]


@dataclass(frozen=True)
class CheckDefinition:
    """Declarative requirements and compiler for one supported check."""

    dependencies: tuple[str, ...]
    phase: int
    goal: str
    compiler: CheckCompiler


def _unavailable_check(
    kind: ProtocolCheck,
    definition: CheckDefinition,
    reason: str,
    *,
    missing: Sequence[str] = (),
) -> CompiledProtocolCheck:
    return CompiledProtocolCheck(
        kind=kind,
        phase=definition.phase,
        status="unavailable",
        dependencies=list(definition.dependencies),
        missing_dependencies=list(missing),
        goal=definition.goal,
        sql_template=f"-- SQL-шаблон не сформирован: {reason}",
        pass_criterion="Проверка требует устранить указанное ограничение.",
        limitations=[reason],
    )


def _compiled_check(
    kind: ProtocolCheck,
    definition: CheckDefinition,
    sql: str,
    pass_criterion: str,
    *,
    status: ProtocolCheckStatus = "ready",
    limitations: Sequence[str] = (),
    missing: Sequence[str] = (),
) -> CompiledProtocolCheck:
    return CompiledProtocolCheck(
        kind=kind,
        phase=definition.phase,
        status=status,
        dependencies=list(definition.dependencies),
        missing_dependencies=list(missing),
        goal=definition.goal,
        sql_template=sql,
        pass_criterion=pass_criterion,
        limitations=list(limitations),
    )


def _compile_row_count(
    kind: ProtocolCheck,
    definition: CheckDefinition,
    ctx: _CompilerContext,
) -> CompiledProtocolCheck:
    if not ctx.transformation_ready:
        return _unavailable_check(
            kind,
            definition,
            "нет однозначного полного SELECT",
            missing=["transformation"],
        )
    sql = (
        f"{ctx.expected_raw_ctes}\n"
        "SELECT\n"
        "    (SELECT COUNT(*) FROM expected) AS expected_row_count,\n"
        f"    (SELECT COUNT(*) FROM {ctx.target_sql} WHERE "
        f"{TARGET_SCOPE_PREDICATE}) AS actual_row_count;"
    )
    return _compiled_check(
        kind,
        definition,
        sql,
        "expected_row_count = actual_row_count.",
    )


def _compile_actual_duplicates(
    kind: ProtocolCheck,
    definition: CheckDefinition,
    ctx: _CompilerContext,
) -> CompiledProtocolCheck:
    if not ctx.comparison_key:
        return _unavailable_check(
            kind,
            definition,
            "не задан explicit key и target PK не подтверждён",
            missing=["comparison_key"],
        )
    keys = ", ".join(
        _quote_identifier(value) for value in ctx.comparison_key
    )
    sql = (
        f"SELECT {keys}, COUNT(*) AS duplicate_count\n"
        f"FROM {ctx.target_sql}\n"
        f"WHERE {TARGET_SCOPE_PREDICATE}\n"
        f"GROUP BY {keys}\n"
        "HAVING COUNT(*) > 1;"
    )
    return _compiled_check(
        kind,
        definition,
        sql,
        "Запрос не возвращает строк.",
    )


def _compile_required_null_rate(
    kind: ProtocolCheck,
    definition: CheckDefinition,
    ctx: _CompilerContext,
) -> CompiledProtocolCheck:
    if not ctx.target_catalog_available:
        return _unavailable_check(
            kind,
            definition,
            "target-каталог недоступен без file scope",
            missing=["target_catalog"],
        )
    if not ctx.required_fields:
        return _unavailable_check(
            kind,
            definition,
            "в target-каталоге нет полей not_null=true",
            missing=["required_fields"],
        )
    null_counts = ",\n    ".join(
        "COUNT(*) FILTER (WHERE "
        f"{_quote_identifier(value)} IS NULL) AS "
        f"{_quote_identifier(value + '_null_count')}"
        for value in ctx.required_fields
    )
    sql = (
        "SELECT\n    COUNT(*) AS total_rows,\n    "
        f"{null_counts}\nFROM {ctx.target_sql}\n"
        f"WHERE {TARGET_SCOPE_PREDICATE};"
    )
    return _compiled_check(
        kind,
        definition,
        sql,
        "Каждый *_null_count равен 0.",
    )


def _compile_transformation_correctness(
    kind: ProtocolCheck,
    definition: CheckDefinition,
    ctx: _CompilerContext,
) -> CompiledProtocolCheck:
    if not ctx.projection_ready:
        return _unavailable_check(
            kind,
            definition,
            "нет однозначного полного SELECT и target-проекции",
            missing=["transformation", "mapped_fields"],
        )
    target_fields = ", ".join(
        _quote_identifier(value) for value in ctx.target_fields
    )
    sql = (
        f"{ctx.expected_ctes},\n"
        "actual AS (\n"
        f"    SELECT {target_fields}\n"
        f"    FROM {ctx.target_sql}\n"
        f"    WHERE {TARGET_SCOPE_PREDICATE}\n"
        "),\ndifferences AS (\n"
        "    (SELECT * FROM expected EXCEPT ALL SELECT * FROM actual)\n"
        "    UNION ALL\n"
        "    (SELECT * FROM actual EXCEPT ALL SELECT * FROM expected)\n"
        ")\nSELECT COUNT(*) AS difference_count\nFROM differences;"
    )
    return _compiled_check(
        kind,
        definition,
        sql,
        "difference_count = 0.",
    )


def _expected_key_ready(ctx: _CompilerContext) -> bool:
    return bool(ctx.comparison_key) and all(
        key.casefold() in ctx.projected_fields for key in ctx.comparison_key
    )


def _compile_key_reconciliation(
    kind: ProtocolCheck,
    definition: CheckDefinition,
    ctx: _CompilerContext,
) -> CompiledProtocolCheck:
    if not ctx.projection_ready or not _expected_key_ready(ctx):
        return _unavailable_check(
            kind,
            definition,
            "comparison key отсутствует в однозначной expected-проекции",
            missing=["transformation", "comparison_key"],
        )
    actual = _actual_cte(ctx.target_sql, ctx.target_fields, marker=True)
    join = _join_on(ctx.comparison_key)
    sql = (
        f"{ctx.expected_ctes},\n"
        "expected_keyed AS (\n"
        "    SELECT e.*, TRUE AS \"__expected_present\" FROM expected AS e\n"
        "),\n"
        f"{actual}\n"
        "SELECT\n"
        "    COUNT(*) FILTER (WHERE a.\"__actual_present\" IS NULL) "
        "AS missing_key_count,\n"
        "    COUNT(*) FILTER (WHERE e.\"__expected_present\" IS NULL) "
        "AS extra_key_count\n"
        "FROM expected_keyed AS e\n"
        f"FULL OUTER JOIN actual AS a ON {join};"
    )
    return _compiled_check(
        kind,
        definition,
        sql,
        "missing_key_count = 0 и extra_key_count = 0.",
    )


def _compile_missing_rows(
    kind: ProtocolCheck,
    definition: CheckDefinition,
    ctx: _CompilerContext,
) -> CompiledProtocolCheck:
    if not ctx.projection_ready or not _expected_key_ready(ctx):
        return _unavailable_check(
            kind,
            definition,
            "нет expected-проекции с comparison key",
            missing=["transformation", "comparison_key"],
        )
    actual = _actual_cte(ctx.target_sql, ctx.target_fields)
    sql = (
        f"{ctx.expected_ctes},\n{actual}\n"
        "SELECT e.*\nFROM expected AS e\n"
        "WHERE NOT EXISTS (\n"
        "    SELECT 1 FROM actual AS a WHERE "
        + _join_on(ctx.comparison_key)
        + "\n);"
    )
    return _compiled_check(
        kind,
        definition,
        sql,
        "Запрос не возвращает строк.",
    )


def _compile_extra_rows(
    kind: ProtocolCheck,
    definition: CheckDefinition,
    ctx: _CompilerContext,
) -> CompiledProtocolCheck:
    if not ctx.projection_ready or not _expected_key_ready(ctx):
        return _unavailable_check(
            kind,
            definition,
            "нет expected-проекции с comparison key",
            missing=["transformation", "comparison_key"],
        )
    actual = _actual_cte(ctx.target_sql, ctx.target_fields)
    sql = (
        f"{ctx.expected_ctes},\n{actual}\n"
        "SELECT a.*\nFROM actual AS a\n"
        "WHERE NOT EXISTS (\n"
        "    SELECT 1 FROM expected AS e WHERE "
        + _join_on(ctx.comparison_key)
        + "\n);"
    )
    return _compiled_check(
        kind,
        definition,
        sql,
        "Запрос не возвращает строк.",
    )


def _compile_field_mismatch(
    kind: ProtocolCheck,
    definition: CheckDefinition,
    ctx: _CompilerContext,
) -> CompiledProtocolCheck:
    key_names = {key.casefold() for key in ctx.comparison_key}
    compared = [
        field for field in ctx.target_fields if field.casefold() not in key_names
    ]
    if not ctx.projection_ready or not _expected_key_ready(ctx) or not compared:
        return _unavailable_check(
            kind,
            definition,
            "нет comparison key либо сравниваемых mapped fields",
            missing=["transformation", "comparison_key", "mapped_fields"],
        )
    actual = _actual_cte(ctx.target_sql, ctx.target_fields)
    left_row = ", ".join(
        f"e.{_quote_identifier(field)}" for field in compared
    )
    right_row = ", ".join(
        f"a.{_quote_identifier(field)}" for field in compared
    )
    key_select = ", ".join(
        f"e.{_quote_identifier(key)}" for key in ctx.comparison_key
    )
    sql = (
        f"{ctx.expected_ctes},\n{actual}\n"
        f"SELECT {key_select}, e.*, a.*\n"
        "FROM expected AS e\n"
        f"JOIN actual AS a ON {_join_on(ctx.comparison_key)}\n"
        f"WHERE ROW({left_row}) IS DISTINCT FROM ROW({right_row});"
    )
    return _compiled_check(
        kind,
        definition,
        sql,
        "Запрос не возвращает строк.",
    )


def _compile_schema_compatibility(
    kind: ProtocolCheck,
    definition: CheckDefinition,
    ctx: _CompilerContext,
) -> CompiledProtocolCheck:
    if not ctx.source_catalog_available or not ctx.target_catalog_available:
        missing = []
        if not ctx.source_catalog_available:
            missing.append("source_catalog")
        if not ctx.target_catalog_available:
            missing.append("target_catalog")
        return _unavailable_check(
            kind,
            definition,
            "source/target catalog metadata недоступны",
            missing=missing,
        )
    if not ctx.schema_rows:
        return _unavailable_check(
            kind,
            definition,
            "нет типизированных source→target пар",
            missing=["typed_mapping"],
        )
    values = ",\n        ".join(
        "("
        + ", ".join(
            _quote_literal(row[field])
            for field in (
                "source_column",
                "source_type",
                "target_column",
                "target_type",
                "compatible",
            )
        )
        + ")"
        for row in ctx.schema_rows
    )
    sql = (
        "WITH schema_check(source_column, source_type, target_column, "
        "target_type, compatible) AS (\n"
        f"    VALUES\n        {values}\n"
        ")\nSELECT * FROM schema_check WHERE NOT compatible;"
    )
    status: ProtocolCheckStatus = (
        "partial" if ctx.schema_missing else "ready"
    )
    limitations = (
        ["Нет type metadata для: " + ", ".join(ctx.schema_missing)]
        if ctx.schema_missing
        else []
    )
    return _compiled_check(
        kind,
        definition,
        sql,
        "Запрос не возвращает строк; partial metadata требует ручной проверки.",
        status=status,
        limitations=limitations,
        missing=["typed_mapping"] if ctx.schema_missing else [],
    )


def _compile_expected_required_nulls(
    kind: ProtocolCheck,
    definition: CheckDefinition,
    ctx: _CompilerContext,
) -> CompiledProtocolCheck:
    if not ctx.target_catalog_available:
        return _unavailable_check(
            kind,
            definition,
            "target-каталог недоступен",
            missing=["target_catalog"],
        )
    missing_projection = [
        field
        for field in ctx.required_fields
        if field.casefold() not in ctx.projected_fields
    ]
    if not ctx.projection_ready or not ctx.required_fields or missing_projection:
        return _unavailable_check(
            kind,
            definition,
            "обязательные поля отсутствуют в expected-проекции",
            missing=["transformation", "required_fields"],
        )
    null_counts = ",\n    ".join(
        f"COUNT(*) FILTER (WHERE {_quote_identifier(field)} IS NULL) AS "
        f"{_quote_identifier(field + '_expected_null_count')}"
        for field in ctx.required_fields
    )
    sql = (
        f"{ctx.expected_ctes}\n"
        f"SELECT\n    {null_counts}\nFROM expected;"
    )
    return _compiled_check(
        kind,
        definition,
        sql,
        "Каждый *_expected_null_count равен 0.",
    )


def _compile_aggregate_reconciliation(
    kind: ProtocolCheck,
    definition: CheckDefinition,
    ctx: _CompilerContext,
) -> CompiledProtocolCheck:
    if not ctx.projection_ready:
        return _unavailable_check(
            kind,
            definition,
            "нет однозначной expected-проекции",
            missing=["transformation", "mapped_fields"],
        )
    expected_metrics = ["COUNT(*) AS row_count"] + [
        f"COUNT({_quote_identifier(field)}) AS "
        f"{_quote_identifier(field + '_non_null_count')}"
        for field in ctx.target_fields
    ]
    metrics_sql = ",\n        ".join(expected_metrics)
    sql = (
        f"{ctx.expected_ctes},\n"
        "expected_aggregates AS (\n    SELECT\n        "
        + metrics_sql
        + "\n    FROM expected\n),\n"
        + "actual_aggregates AS (\n    SELECT\n        "
        + metrics_sql
        + f"\n    FROM {ctx.target_sql}\n"
        + f"    WHERE {TARGET_SCOPE_PREDICATE}\n)\n"
        + "SELECT e.*, a.* FROM expected_aggregates AS e "
        + "CROSS JOIN actual_aggregates AS a;"
    )
    return _compiled_check(
        kind,
        definition,
        sql,
        "Все одноимённые expected/actual агрегаты равны.",
    )


def _compile_duplicate_expected(
    kind: ProtocolCheck,
    definition: CheckDefinition,
    ctx: _CompilerContext,
) -> CompiledProtocolCheck:
    if not ctx.projection_ready or not _expected_key_ready(ctx):
        return _unavailable_check(
            kind,
            definition,
            "comparison key отсутствует в expected-проекции",
            missing=["transformation", "comparison_key"],
        )
    keys = ", ".join(
        _quote_identifier(value) for value in ctx.comparison_key
    )
    sql = (
        f"{ctx.expected_ctes}\n"
        f"SELECT {keys}, COUNT(*) AS duplicate_count\n"
        "FROM expected\n"
        f"GROUP BY {keys}\nHAVING COUNT(*) > 1;"
    )
    return _compiled_check(
        kind,
        definition,
        sql,
        "Запрос не возвращает строк.",
    )


CHECKS: Dict[ProtocolCheck, CheckDefinition] = {
    "row_count": CheckDefinition(
        ("transformation",),
        1,
        "Сравнить количество ожидаемых и загруженных строк.",
        _compile_row_count,
    ),
    "key_uniqueness": CheckDefinition(
        ("comparison_key",),
        1,
        "Проверить уникальность target-ключа.",
        _compile_actual_duplicates,
    ),
    "required_null_rate": CheckDefinition(
        ("target_catalog", "required_fields"),
        1,
        "Проверить NULL в обязательных target-полях.",
        _compile_required_null_rate,
    ),
    "transformation_correctness": CheckDefinition(
        ("transformation", "mapped_fields"),
        2,
        "Сравнить полную ожидаемую S2T-проекцию с target.",
        _compile_transformation_correctness,
    ),
    "key_reconciliation": CheckDefinition(
        ("transformation", "comparison_key"),
        2,
        "Сверить множества expected и actual ключей.",
        _compile_key_reconciliation,
    ),
    "missing_rows": CheckDefinition(
        ("transformation", "comparison_key"),
        3,
        "Показать ожидаемые строки, отсутствующие в target.",
        _compile_missing_rows,
    ),
    "extra_rows": CheckDefinition(
        ("transformation", "comparison_key"),
        3,
        "Показать лишние строки target.",
        _compile_extra_rows,
    ),
    "field_mismatch": CheckDefinition(
        ("transformation", "comparison_key", "mapped_fields"),
        3,
        "Показать расхождения mapped fields по ключу.",
        _compile_field_mismatch,
    ),
    "schema_compatibility": CheckDefinition(
        ("source_catalog", "target_catalog"),
        1,
        "Проверить совместимость source/target типов.",
        _compile_schema_compatibility,
    ),
    "expected_required_nulls": CheckDefinition(
        ("transformation", "target_catalog", "required_fields"),
        1,
        "Проверить NULL обязательных полей до загрузки.",
        _compile_expected_required_nulls,
    ),
    "aggregate_reconciliation": CheckDefinition(
        ("transformation", "mapped_fields"),
        2,
        "Сверить row-count и non-null агрегаты expected/actual.",
        _compile_aggregate_reconciliation,
    ),
    "duplicate_expected": CheckDefinition(
        ("transformation", "comparison_key"),
        3,
        "Найти дубли ключа в ожидаемом наборе.",
        _compile_duplicate_expected,
    ),
    "duplicate_actual": CheckDefinition(
        ("comparison_key",),
        1,
        "Найти дубли ключа в target.",
        _compile_actual_duplicates,
    ),
}


def check_dependencies(checks: Sequence[ProtocolCheck]) -> List[str]:
    """Return minimal declarative dependencies in stable order."""

    return list(
        dict.fromkeys(
            dependency
            for check in checks
            for dependency in CHECKS[check].dependencies
        )
    )


def _catalog_fields(
    rows: Sequence[Mapping[str, Any]],
    flag: str,
    mapped_names: Mapping[str, str],
) -> List[str]:
    return list(
        dict.fromkeys(
            mapped_names.get(name.casefold(), name)
            for row in rows
            if _truthy(row.get(flag))
            and (name := str(row.get("column_name") or "").strip())
        )
    )


def _schema_comparisons(
    s2t_rows: Sequence[Mapping[str, Any]],
    source_catalog: Sequence[Mapping[str, Any]],
    target_catalog: Sequence[Mapping[str, Any]],
) -> tuple[List[Dict[str, Any]], List[str]]:
    source_types = {
        (
            str(row.get("table_name") or "").strip().casefold(),
            str(row.get("column_name") or "").strip().casefold(),
        ): str(row.get("data_type") or "").strip()
        for row in source_catalog
        if str(row.get("column_name") or "").strip()
    }
    target_types = {
        str(row.get("column_name") or "").strip().casefold(): str(
            row.get("data_type") or ""
        ).strip()
        for row in target_catalog
        if str(row.get("column_name") or "").strip()
    }
    comparisons: List[Dict[str, Any]] = []
    missing: List[str] = []
    seen: set[tuple[str, str, str]] = set()
    for row in s2t_rows:
        source_table = str(row.get("source_table") or "").strip()
        source_field = str(row.get("source_field") or "").strip()
        target_field = str(row.get("target_field") or "").strip()
        identity = (
            source_table.casefold(),
            source_field.casefold(),
            target_field.casefold(),
        )
        if not source_field or not target_field or identity in seen:
            continue
        seen.add(identity)
        source_type = source_types.get(
            (source_table.casefold(), source_field.casefold()),
            "",
        )
        target_type = target_types.get(target_field.casefold(), "")
        if not source_type or not target_type:
            missing.append(f"{source_table}.{source_field}→{target_field}")
            continue
        comparisons.append(
            {
                "source_column": f"{source_table}.{source_field}",
                "source_type": source_type,
                "target_column": target_field,
                "target_type": target_type,
                "compatible": _types_compatible(source_type, target_type),
            }
        )
    return comparisons, missing


def _preflight(
    bundle: _ReaderBundle,
    normalized: NormalizedTransformation,
    pairs: Sequence[tuple[str, str]],
    missing_sources: Sequence[str],
    rule_selection_error: Optional[str],
    missing_projection: Sequence[str],
    ambiguous_projection: Sequence[str],
    required_fields: Sequence[str],
) -> List[ProtocolPreflightItem]:
    mapped = list(dict.fromkeys(target for _, target in pairs))
    catalog = list(
        dict.fromkeys(
            str(row.get("column_name") or "").strip()
            for row in bundle.target_catalog_rows
            if str(row.get("column_name") or "").strip()
        )
    )
    mapped_folded = {value.casefold() for value in mapped}
    catalog_folded = {value.casefold() for value in catalog}
    mapped_not_catalog = [
        value for value in mapped if value.casefold() not in catalog_folded
    ]
    unmapped_required = [
        value
        for value in required_fields
        if value.casefold() not in mapped_folded
    ]
    catalog_status: Literal["pass", "warning", "fail", "unavailable"] = (
        "pass" if bundle.target_catalog_available else "unavailable"
    )
    return [
        ProtocolPreflightItem(
            kind="mapping_coverage",
            status=catalog_status,
            conclusion=(
                f"S2T покрывает {len(mapped_folded)} из "
                f"{len(catalog_folded)} catalog fields."
                if bundle.target_catalog_available
                else "Target catalog metadata недоступны; покрытие не вычислено."
            ),
        ),
        ProtocolPreflightItem(
            kind="target_fields_exist",
            status=(
                "fail"
                if bundle.target_catalog_available and mapped_not_catalog
                else catalog_status
            ),
            conclusion=(
                "Mapped fields вне target-каталога: "
                + ", ".join(mapped_not_catalog)
                if mapped_not_catalog
                else "Все mapped target fields подтверждены каталогом."
                if bundle.target_catalog_available
                else "Проверка target fields недоступна без каталога."
            ),
        ),
        ProtocolPreflightItem(
            kind="unmapped_required_fields",
            status=(
                "fail"
                if bundle.target_catalog_available and unmapped_required
                else catalog_status
            ),
            conclusion=(
                "Обязательные поля без маппинга: "
                + ", ".join(unmapped_required)
                if unmapped_required
                else "Обязательные поля покрыты S2T."
                if bundle.target_catalog_available
                else "Проверка обязательных полей недоступна без каталога."
            ),
        ),
        ProtocolPreflightItem(
            kind="requested_sources_exist",
            status="fail" if missing_sources else "pass",
            conclusion=(
                "Запрошенные sources отсутствуют в S2T: "
                + ", ".join(missing_sources)
                if missing_sources
                else "Все запрошенные sources подтверждены S2T."
            ),
        ),
        ProtocolPreflightItem(
            kind="transformation_sql_parses",
            status="pass" if normalized.parse_status == "ok" else "fail",
            conclusion=(
                "Transformation SQL разобран SQLGlot."
                if normalized.parse_status == "ok"
                else normalized.error or "Transformation SQL недоступен."
            ),
        ),
        ProtocolPreflightItem(
            kind="mapping_ambiguity",
            status=(
                "fail"
                if rule_selection_error or ambiguous_projection
                else "pass"
            ),
            conclusion=rule_selection_error
            or (
                "Неоднозначные target projections: "
                + ", ".join(ambiguous_projection)
                if ambiguous_projection
                else "Неоднозначность mapping rule не обнаружена."
            ),
        ),
        ProtocolPreflightItem(
            kind="target_projection_consistency",
            status="fail" if missing_projection else "pass",
            conclusion=(
                "Transformation output не содержит target fields: "
                + ", ".join(missing_projection)
                if missing_projection
                else "Target projection согласована с S2T mapping."
            ),
        ),
    ]


def _target_status(
    checks: Sequence[CompiledProtocolCheck],
    preflight: Sequence[ProtocolPreflightItem],
) -> ProtocolStatus:
    if not checks or all(check.status == "unavailable" for check in checks):
        return "unavailable"
    if any(check.status != "ready" for check in checks) or any(
        item.status == "fail" for item in preflight
    ):
        return "partial_protocol"
    return "ready"


def _phase_summaries(
    targets: Sequence[CompiledTargetProtocol],
) -> List[ProtocolPhaseSummary]:
    names = {
        0: "static/preflight",
        1: "smoke",
        2: "reconciliation",
        3: "diagnostics",
    }
    preflight = [item for target in targets for item in target.preflight]
    summaries = [
        ProtocolPhaseSummary(
            phase=0,
            name=names[0],
            item_count=len(preflight),
            ready_count=sum(item.status == "pass" for item in preflight),
            partial_count=sum(item.status == "warning" for item in preflight),
            unavailable_count=sum(
                item.status == "unavailable" for item in preflight
            ),
            failed_count=sum(item.status == "fail" for item in preflight),
        )
    ]
    for phase in (1, 2, 3):
        checks = [
            check
            for target in targets
            for check in target.checks
            if check.phase == phase
        ]
        summaries.append(
            ProtocolPhaseSummary(
                phase=phase,
                name=names[phase],
                item_count=len(checks),
                ready_count=sum(check.status == "ready" for check in checks),
                partial_count=sum(
                    check.status == "partial" for check in checks
                ),
                unavailable_count=sum(
                    check.status == "unavailable" for check in checks
                ),
                failed_count=0,
            )
        )
    return summaries


def compile_test_protocol(
    contract: ResolvedTestProtocolContract,
    *,
    reader_results: Sequence[Dict[str, Any]],
) -> CompiledTestProtocol:
    """Compile phased Greenplum SQL templates without executing ETL data."""

    targets: List[CompiledTargetProtocol] = []
    issues: List[ProtocolIssue] = []
    for load_index, load in enumerate(contract.loads, start=1):
        target_table = load.target
        bundle = _load_results(load_index, target_table, reader_results)
        rule, missing_sources, rule_selection_error = _select_load_rule(
            load.sources,
            bundle.pair_rows,
        )
        normalized = normalize_transformation(rule)
        s2t_rows = _mapping_rows_for_rule(rule, load, bundle)
        pairs = _mapping_pairs(s2t_rows)
        (
            projection_plan,
            missing_projection,
            ambiguous_projection,
        ) = _projection_plan(normalized, pairs)
        target_fields = [target for _, target in projection_plan]
        mapped_names = {target.casefold(): target for _, target in pairs}
        pk_fields = _catalog_fields(
            bundle.target_catalog_rows,
            "primary_key",
            mapped_names,
        )
        required_fields = _catalog_fields(
            bundle.target_catalog_rows,
            "not_null",
            mapped_names,
        )
        explicit_key = contract.explicit_key_for_load(load)
        comparison_key = explicit_key or pk_fields
        comparison_key_source = (
            "explicit"
            if explicit_key
            else "target_primary_key"
            if pk_fields
            else None
        )
        invalid_keys = [
            value
            for value in comparison_key
            if not _SIMPLE_IDENTIFIER.fullmatch(value)
        ]
        if invalid_keys:
            comparison_key = []
            comparison_key_source = None
        target_sql = _quote_qualified_name(target_table)
        schema_rows, schema_missing = _schema_comparisons(
            s2t_rows,
            bundle.source_catalog_rows,
            bundle.target_catalog_rows,
        )
        ctx = _CompilerContext(
            target_sql=target_sql,
            normalized=normalized,
            projection_plan=projection_plan,
            target_fields=target_fields,
            comparison_key=comparison_key,
            comparison_key_source=comparison_key_source,
            required_fields=required_fields,
            target_catalog_available=bundle.target_catalog_available,
            source_catalog_available=bundle.source_catalog_available,
            schema_rows=schema_rows,
            schema_missing=schema_missing,
        )
        selected_checks = contract.checks_for_load(load)
        checks = [
            CHECKS[kind].compiler(kind, CHECKS[kind], ctx)
            for kind in selected_checks
        ]
        preflight = _preflight(
            bundle,
            normalized,
            pairs,
            missing_sources,
            rule_selection_error,
            missing_projection,
            ambiguous_projection,
            required_fields,
        )
        limitations: List[str] = []
        if missing_sources:
            limitations.append(
                "В S2T не подтверждены sources: "
                + ", ".join(missing_sources)
                + "."
            )
        if rule_selection_error:
            limitations.append(rule_selection_error)
        if invalid_keys:
            limitations.append(
                "Небезопасные имена explicit key: "
                + ", ".join(invalid_keys)
                + "."
            )
        status = _target_status(checks, preflight)
        if status != "ready":
            issues.append(
                ProtocolIssue(
                    code=(
                        "partial_protocol"
                        if status == "partial_protocol"
                        else "unavailable_check"
                    ),
                    message=(
                        f"Protocol для {target_table} имеет статус {status}."
                    ),
                    load_index=load_index,
                )
            )
        for check in checks:
            if check.status == "unavailable":
                issues.append(
                    ProtocolIssue(
                        code="unavailable_check",
                        message="; ".join(check.limitations),
                        load_index=load_index,
                        check=check.kind,
                    )
                )
        source_tables = list(
            dict.fromkeys(
                str(row.get("source_table") or "").strip()
                for row in s2t_rows
                if str(row.get("source_table") or "").strip()
            )
        )
        targets.append(
            CompiledTargetProtocol(
                source_tables=list(load.sources),
                target_table=target_table,
                mode=contract.mode,
                status=status,
                comparison_key=comparison_key,
                comparison_key_source=comparison_key_source,
                preflight=preflight,
                checks=checks,
                evidence=[
                    f"S2T sources: {source_tables!r}",
                    f"mapped target fields: {target_fields!r}",
                    "comparison key "
                    f"({comparison_key_source or 'none'}): {comparison_key!r}",
                    f"required fields: {required_fields!r}",
                ],
                limitations=limitations,
                normalized_transformation=normalized,
                s2t_evidence_rows=s2t_rows,
                catalog_evidence_rows=bundle.target_catalog_rows,
                source_catalog_evidence_rows=bundle.source_catalog_rows,
            )
        )
    target_statuses = {target.status for target in targets}
    status: ProtocolStatus = (
        "ready"
        if target_statuses <= {"ready"}
        else "unavailable"
        if target_statuses <= {"unavailable"}
        else "partial_protocol"
    )
    return CompiledTestProtocol(
        mode=contract.mode,
        status=status,
        issues=issues,
        phases=_phase_summaries(targets),
        targets=targets,
    )


_CHECK_TITLES: Dict[ProtocolCheck, str] = {
    "row_count": "Проверка количества строк",
    "key_uniqueness": "Проверка уникальности ключа",
    "required_null_rate": "Проверка null-rate обязательных полей",
    "transformation_correctness": "Проверка корректности трансформаций",
    "key_reconciliation": "Сверка ключей",
    "missing_rows": "Отсутствующие строки",
    "extra_rows": "Лишние строки",
    "field_mismatch": "Расхождения полей",
    "schema_compatibility": "Совместимость схем",
    "expected_required_nulls": "NULL до загрузки",
    "aggregate_reconciliation": "Сверка агрегатов",
    "duplicate_expected": "Дубли expected",
    "duplicate_actual": "Дубли target",
}


def render_test_protocol_answer(
    contract: ResolvedTestProtocolContract,
    protocol: CompiledTestProtocol,
) -> str:
    """Render phased exact compiler output without a final LLM rewrite."""

    sources = ", ".join(f"`{value}`" for value in contract.source_tables)
    targets = ", ".join(f"`{value}`" for value in contract.target_tables)
    sections = [
        "Тест-протокол проверки ETL-загрузки во внешней Greenplum СУБД "
        f"для sources [{sources}] → targets [{targets}]. Режим: "
        f"{protocol.mode}; статус: {protocol.status}. SQL-шаблоны не "
        "исполнялись; фактические метрики не вычислялись. Замени "
        f"`{SOURCE_SCOPE_PREDICATE}` условием expected/source scope, а "
        f"`{TARGET_SCOPE_PREDICATE}` условием target scope; используй "
        "`TRUE` для полного снимка."
    ]
    phase_titles = {
        1: "Phase 1 — cheap smoke checks",
        2: "Phase 2 — full reconciliation",
        3: "Phase 3 — diagnostics",
    }
    for target in protocol.targets:
        load_sources = ", ".join(
            f"`{value}`" for value in target.source_tables
        )
        target_parts = [
            f"Load sources [{load_sources}] → Target `{target.target_table}`",
            "Phase 0 — static/preflight\n"
            + "\n".join(
                f"- [{item.status}] {item.kind}: {item.conclusion}"
                for item in target.preflight
            ),
        ]
        check_index = 0
        for phase in (1, 2, 3):
            phase_checks = [
                check for check in target.checks if check.phase == phase
            ]
            if not phase_checks:
                continue
            target_parts.append(phase_titles[phase])
            for check in phase_checks:
                check_index += 1
                block = (
                    f"{check_index}. {_CHECK_TITLES[check.kind]}\n"
                    f"Статус: {check.status}\n"
                    f"Цель: {check.goal}\n"
                    "SQL-шаблон:\n```sql\n"
                    f"{check.sql_template}\n```\n"
                    f"Критерий прохождения: {check.pass_criterion}"
                )
                if check.limitations:
                    block += "\nОграничения: " + "; ".join(
                        check.limitations
                    )
                target_parts.append(block)
        target_parts.append(
            "Подтверждённые основания: " + "; ".join(target.evidence)
        )
        if target.limitations:
            target_parts.append(
                "Ограничения target: " + "; ".join(target.limitations)
            )
        sections.append("\n\n".join(target_parts))
    return "\n\n".join(sections)


def build_test_protocol_display_payloads(
    protocol: CompiledTestProtocol,
) -> List[Dict[str, str]]:
    """Build user-visible copies of the exact evidence used by the compiler."""

    displays: List[Dict[str, str]] = []
    for target in protocol.targets:
        for name, rows in (
            ("read_s2t_source_to_target", target.s2t_evidence_rows),
            ("list_target_column_catalog", target.catalog_evidence_rows),
            (
                "list_source_column_catalog",
                target.source_catalog_evidence_rows,
            ),
        ):
            if not rows:
                continue
            columns = list(dict.fromkeys(key for row in rows for key in row))
            payload = {
                "load": {
                    "sources": list(target.source_tables),
                    "target": target.target_table,
                },
                "columns": columns,
                "rows": [
                    [row.get(column) for column in columns] for row in rows
                ],
            }
            displays.append(
                {
                    "name": name,
                    "content": json.dumps(
                        payload,
                        ensure_ascii=False,
                        default=str,
                        separators=(",", ":"),
                    ),
                }
            )
    return displays


__all__ = [
    "CHECKS",
    "LOAD_SCOPE_PREDICATE",
    "MAX_PROTOCOL_OBJECTS",
    "PROTOCOL_CHECKS",
    "SOURCE_SCOPE_PREDICATE",
    "STANDARD_PROTOCOL_CHECKS",
    "TARGET_SCOPE_PREDICATE",
    "CheckDefinition",
    "CompiledProtocolCheck",
    "CompiledTargetProtocol",
    "CompiledTestProtocol",
    "EntityResolutionMetadata",
    "ProtocolCheck",
    "ProtocolIssue",
    "ProtocolMode",
    "ProtocolPhaseSummary",
    "ProtocolPreflightItem",
    "RawTestProtocolContract",
    "RawTestProtocolLoad",
    "ResolvedTestProtocolContract",
    "TestProtocolContract",
    "TestProtocolLoad",
    "build_test_protocol_display_payloads",
    "check_dependencies",
    "compile_test_protocol",
    "render_test_protocol_answer",
]
