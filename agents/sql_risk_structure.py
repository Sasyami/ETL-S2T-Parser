"""Neutral SQLGlot structure for the bounded SQL-risk scope pipeline.

This module has deliberately narrow responsibilities: it verifies that the
provided reader evidence is the exact evidence declared by an already
validated :class:`SqlRiskScopeContract`, decodes the complete tabular
payloads, and exposes syntax/metadata facts.  It does not classify natural
language, infer risk, choose an outcome, or render a public answer.

The returned bundle is suitable as input to a separate structured LLM call.
SQL text is deduplicated only by exact stored value; original S2T row
occurrences remain visible through ``occurrence_count`` and mapping counts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Mapping, Sequence

import sqlglot
from pydantic import BaseModel, ConfigDict, Field, JsonValue, ValidationError
from sqlglot import exp
from sqlglot.errors import SqlglotError

from services.sql_dialects import GREENPLUM_DIALECT

from .sql_risk_assessment import MAX_SQL_RISK_ASSESSMENT_INPUT_CHARS
from .sql_risk_scope_contract import SqlRiskScopeContract, SqlRiskToolName
from .sql_risk_scope_extraction import SqlRiskScopeExecutionMode
from .transformation_ast import quote_dollar_schemas


# The assessment envelope also contains up to 16k characters of original task,
# 4k of stable context, provenance wrappers and a second copy of rule/evidence
# identifiers. Reserve enough room for that envelope instead of allowing the
# neutral bundle alone to consume almost the complete assessment boundary.
SQL_RISK_ASSESSMENT_ENVELOPE_RESERVE_CHARS = 45_000
SQL_RISK_STRUCTURE_MAX_CHARS = (
    MAX_SQL_RISK_ASSESSMENT_INPUT_CHARS
    - SQL_RISK_ASSESSMENT_ENVELOPE_RESERVE_CHARS
)

SqlRiskStructureStatus = Literal["ready", "unavailable"]
SqlRiskStructureIssueCode = Literal[
    "invalid_contract",
    "missing_evidence",
    "unexpected_evidence",
    "duplicate_evidence",
    "invalid_payload",
    "incomplete_payload",
    "sql_parse_error",
    "bundle_too_large",
]

_CLOSED_EXECUTION_MODES = {
    "row_filtering",
    "conditional_cardinality",
    "nullable_constraint",
    "value_changes",
    "write_semantics",
}
_MAPPING_TOOLS = {
    "read_s2t_source_to_target",
    "list_s2t_field_mapping",
}
_METADATA_TOOL = "get_source_target_column_pair"
_MAPPING_REQUIRED_COLUMNS = {
    "source_table",
    "target_table",
    "transformation_rule",
}
_METADATA_REQUIRED_COLUMNS = {
    "column_role",
    "file_id",
    "table_name",
    "column_name",
    "not_null",
}


class SqlRiskStructuralEvidence(BaseModel):
    """One complete exact-reader payload and its runtime provenance."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    evidence_id: str = Field(min_length=1)
    tool_name: SqlRiskToolName
    arguments: Dict[str, Any]
    payload: Dict[str, Any]


class SqlRiskStructureIssue(BaseModel):
    """Explicit reason for missing structure or a partial SQL parse."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    code: SqlRiskStructureIssueCode
    message: str = Field(min_length=1)
    evidence_id: str | None = None
    rule_id: str | None = None


class SqlRiskScopeStructure(BaseModel):
    """The exact role-preserving scope copied from the validated contract."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source: str
    target: str
    source_table: str
    target_table: str
    source_field: str | None = None
    target_field: str | None = None
    file_id: int | None = None


class SqlRiskEvidenceStructure(BaseModel):
    """Counts and identity of one complete reader payload."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    evidence_id: str
    tool_name: SqlRiskToolName
    arguments: Dict[str, Any]
    returned_rows: int = Field(ge=0)
    source_total: int | None = Field(default=None, ge=0)


class SqlRiskColumnMetadataRow(BaseModel):
    """Relevant raw catalog attributes for one exact endpoint row."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    evidence_id: str
    row_index: int = Field(ge=1)
    column_role: str
    file_id: int
    table_name: str
    column_name: str
    data_type: str | None = None
    primary_key: str | int | float | bool | None = None
    not_null: str | int | float | bool | None = None
    description: str | None = None


class SqlRiskMappingRow(BaseModel):
    """One losslessly decoded exact-reader row with its evidence identity."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    evidence_id: str
    row_index: int = Field(ge=1)
    values: Dict[str, JsonValue]


class SqlRiskRelationStructure(BaseModel):
    """One physical table occurrence in a parsed statement."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    alias: str | None = None
    sql: str
    query_block_id: str


class SqlRiskEqualityStructure(BaseModel):
    """One equality expression retained from a complete JOIN predicate."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    sql: str
    left: str
    right: str


class SqlRiskJoinStructure(BaseModel):
    """One SQLGlot JOIN node without semantic interpretation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    query_block_id: str
    join_type: str
    relation: str
    predicate: str | None = None
    using_columns: List[str] = Field(default_factory=list)
    equalities: List[SqlRiskEqualityStructure] = Field(default_factory=list)
    referenced_columns: List[str] = Field(default_factory=list)


class SqlRiskProjectionStructure(BaseModel):
    """One SELECT projection and the columns syntactically referenced by it."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    query_block_id: str
    sql: str
    expression: str
    expression_kind: str
    output_name: str | None = None
    explicit_alias: str | None = None
    referenced_columns: List[str] = Field(default_factory=list)
    has_wildcard: bool = False


class SqlRiskScopedExpression(BaseModel):
    """One clause expression tied to its nearest SELECT or statement root."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    query_block_id: str
    sql: str


class SqlRiskSetOperationStructure(BaseModel):
    """One UNION/INTERSECT/EXCEPT node."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    operation: str
    distinct: bool | None = None
    by_name: bool = False


class SqlRiskStatementStructure(BaseModel):
    """Neutral structure of one parsed SQL statement."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    statement_index: int = Field(ge=1)
    statement_kind: str
    normalized_sql: str
    write_targets: List[str] = Field(default_factory=list)
    sources: List[SqlRiskRelationStructure] = Field(default_factory=list)
    cte_names: List[str] = Field(default_factory=list)
    joins: List[SqlRiskJoinStructure] = Field(default_factory=list)
    filters: List[SqlRiskScopedExpression] = Field(default_factory=list)
    having: List[SqlRiskScopedExpression] = Field(default_factory=list)
    qualify: List[SqlRiskScopedExpression] = Field(default_factory=list)
    distinct_query_blocks: List[str] = Field(default_factory=list)
    group_by: List[SqlRiskScopedExpression] = Field(default_factory=list)
    order_by: List[SqlRiskScopedExpression] = Field(default_factory=list)
    limits: List[SqlRiskScopedExpression] = Field(default_factory=list)
    offsets: List[SqlRiskScopedExpression] = Field(default_factory=list)
    set_operations: List[SqlRiskSetOperationStructure] = Field(
        default_factory=list
    )
    projections: List[SqlRiskProjectionStructure] = Field(default_factory=list)
    referenced_columns: List[str] = Field(default_factory=list)


class SqlRiskMappingEndpoint(BaseModel):
    """Exact S2T endpoint tuple represented by one deduplicated SQL rule."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source_table: str
    source_field: str | None = None
    target_table: str
    target_field: str | None = None
    occurrence_count: int = Field(ge=1)
    evidence_ids: List[str] = Field(min_length=1)


class SqlRiskRuleStructure(BaseModel):
    """One exact stored transformation-rule value and its parsed structure."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    rule_id: str
    raw_sql: str | None = None
    occurrence_count: int = Field(ge=1)
    evidence_ids: List[str] = Field(min_length=1)
    mappings: List[SqlRiskMappingEndpoint] = Field(default_factory=list)
    parse_status: Literal["ok", "error"]
    parse_error: str | None = None
    normalized_sql: str | None = None
    statements: List[SqlRiskStatementStructure] = Field(default_factory=list)


class SqlRiskStructureBundle(BaseModel):
    """Complete neutral input for a later SQL-risk assessment call."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    status: SqlRiskStructureStatus
    execution_mode: SqlRiskScopeExecutionMode
    scope: SqlRiskScopeStructure
    evidence: List[SqlRiskEvidenceStructure] = Field(default_factory=list)
    mapping_row_count: int = Field(default=0, ge=0)
    mapping_source_total: int | None = Field(default=None, ge=0)
    metadata_row_count: int = Field(default=0, ge=0)
    metadata_source_total: int | None = Field(default=None, ge=0)
    mapping_rows: List[SqlRiskMappingRow] = Field(default_factory=list)
    rules: List[SqlRiskRuleStructure] = Field(default_factory=list)
    metadata_rows: List[SqlRiskColumnMetadataRow] = Field(default_factory=list)
    issues: List[SqlRiskStructureIssue] = Field(default_factory=list)


@dataclass(frozen=True)
class _DecodedRows:
    rows: tuple[Dict[str, Any], ...]
    columns: tuple[str, ...]
    source_total: int | None


def _scope_structure(contract: SqlRiskScopeContract) -> SqlRiskScopeStructure:
    scope = contract.scope
    return SqlRiskScopeStructure(
        source=scope.source,
        target=scope.target,
        source_table=scope.source_table,
        target_table=scope.target_table,
        source_field=scope.source_field,
        target_field=scope.target_field,
        file_id=scope.file_id,
    )


def _issue(
    code: SqlRiskStructureIssueCode,
    message: str,
    *,
    evidence_id: str | None = None,
    rule_id: str | None = None,
) -> SqlRiskStructureIssue:
    return SqlRiskStructureIssue(
        code=code,
        message=message,
        evidence_id=evidence_id,
        rule_id=rule_id,
    )


def _payload_total(payload: Mapping[str, Any]) -> int | None:
    for key in ("total", "total_matches", "total_candidates", "row_count"):
        value = payload.get(key)
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            return value
    return None


def _decode_rows(evidence: SqlRiskStructuralEvidence) -> _DecodedRows:
    payload = evidence.payload
    if payload.get("error"):
        raise ValueError(f"reader returned error: {payload['error']}")
    raw_rows = payload.get("rows")
    if not isinstance(raw_rows, list):
        raise ValueError("payload.rows must be a list")
    raw_columns = payload.get("columns", [])
    if raw_columns is None:
        raw_columns = []
    if not isinstance(raw_columns, list) or any(
        not isinstance(item, str) or not item for item in raw_columns
    ):
        raise ValueError("payload.columns must contain non-empty strings")
    if len(set(raw_columns)) != len(raw_columns):
        raise ValueError("payload.columns must be unique")

    rows: list[Dict[str, Any]] = []
    if payload.get("row_format") == "arrays_in_column_order":
        if not raw_columns:
            raise ValueError("packed payload requires columns")
        dictionaries = payload.get("dictionaries", {})
        if not isinstance(dictionaries, Mapping):
            raise ValueError("packed payload dictionaries must be an object")
        for key, values in dictionaries.items():
            if key not in raw_columns or not isinstance(values, list):
                raise ValueError("packed payload contains an invalid dictionary")
        for raw_row in raw_rows:
            if (
                not isinstance(raw_row, Sequence)
                or isinstance(raw_row, (str, bytes, bytearray))
                or len(raw_row) != len(raw_columns)
            ):
                raise ValueError("packed row does not match declared columns")
            row: Dict[str, Any] = {}
            for column, raw_value in zip(raw_columns, raw_row):
                value = raw_value
                if column in dictionaries:
                    dictionary = dictionaries[column]
                    if (
                        isinstance(value, bool)
                        or not isinstance(value, int)
                        or value < 0
                        or value >= len(dictionary)
                    ):
                        raise ValueError(
                            f"invalid dictionary index for column {column!r}"
                        )
                    value = dictionary[value]
                row[column] = value
            rows.append(row)
    else:
        for raw_row in raw_rows:
            if not isinstance(raw_row, Mapping):
                raise ValueError("unpacked payload rows must be objects")
            rows.append(dict(raw_row))

    columns = list(raw_columns)
    for row in rows:
        for column in row:
            if not isinstance(column, str) or not column:
                raise ValueError("row columns must be non-empty strings")
            if column not in columns:
                columns.append(column)

    returned_rows = payload.get("returned_rows")
    if returned_rows is not None and (
        isinstance(returned_rows, bool)
        or not isinstance(returned_rows, int)
        or returned_rows != len(rows)
    ):
        raise ValueError("returned_rows does not match payload.rows")
    source_total = _payload_total(payload)
    if payload.get("truncated") is True or (
        source_total is not None and source_total != len(rows)
    ):
        raise RuntimeError("reader payload is truncated or incomplete")
    return _DecodedRows(tuple(rows), tuple(columns), source_total)


def _sql(node: exp.Expression | None) -> str:
    return node.sql(dialect=GREENPLUM_DIALECT) if node is not None else ""


def _qualified_table_name(table: exp.Table) -> str:
    return ".".join(
        part for part in (table.catalog, table.db, table.name) if part
    )


def _nearest_select(node: exp.Expression) -> exp.Select | None:
    current = node.parent
    while current is not None:
        if isinstance(current, exp.Select):
            return current
        current = current.parent
    return None


def _unique(values: Sequence[str]) -> List[str]:
    return list(dict.fromkeys(value for value in values if value))


def _write_target_nodes(statement: exp.Expression) -> List[exp.Table]:
    target_roots: list[exp.Expression] = []
    key = str(getattr(statement, "key", "") or "").casefold()
    if key in {
        "insert",
        "update",
        "delete",
        "merge",
        "create",
        "replace",
        "alter",
        "drop",
        "truncate",
        "copy",
    }:
        root = statement.args.get("this")
        if isinstance(root, exp.Expression):
            target_roots.append(root)
    into = statement.args.get("into")
    if isinstance(into, exp.Expression):
        target_roots.append(into)

    targets: list[exp.Table] = []
    for root in target_roots:
        if isinstance(root, exp.Table):
            targets.append(root)
            continue
        table = root.find(exp.Table)
        if table is not None:
            targets.append(table)
    return list(dict.fromkeys(targets))


def _set_operations(statement: exp.Expression) -> List[exp.Expression]:
    nodes: list[exp.Expression] = []
    if isinstance(statement, (exp.Union, exp.Intersect, exp.Except)):
        nodes.append(statement)
    for node in statement.find_all(exp.SetOperation):
        if node not in nodes:
            nodes.append(node)
    return nodes


def _statement_structure(
    statement: exp.Expression,
    *,
    statement_index: int,
) -> SqlRiskStatementStructure:
    selects = list(statement.find_all(exp.Select))
    if isinstance(statement, exp.Select) and statement not in selects:
        selects.insert(0, statement)
    query_ids = {id(node): f"query_{index}" for index, node in enumerate(selects, 1)}

    def query_id(node: exp.Expression) -> str:
        select = _nearest_select(node)
        return query_ids.get(id(select), "statement")

    cte_names = _unique(
        [
            str(cte.alias_or_name or "").strip()
            for cte in statement.find_all(exp.CTE)
            if str(cte.alias_or_name or "").strip()
        ]
    )
    folded_ctes = {name.casefold() for name in cte_names}
    target_nodes = _write_target_nodes(statement)
    target_node_ids = {id(node) for node in target_nodes}
    write_targets = _unique(
        [_qualified_table_name(node) or _sql(node) for node in target_nodes]
    )

    sources: list[SqlRiskRelationStructure] = []
    seen_sources: set[tuple[str, str | None, str, str]] = set()
    for table in statement.find_all(exp.Table):
        name = _qualified_table_name(table)
        if (
            not name
            or id(table) in target_node_ids
            or table.name.casefold() in folded_ctes
        ):
            continue
        alias = str(table.alias or "").strip() or None
        item = (name, alias, _sql(table), query_id(table))
        if item in seen_sources:
            continue
        seen_sources.add(item)
        sources.append(
            SqlRiskRelationStructure(
                name=name,
                alias=alias,
                sql=item[2],
                query_block_id=item[3],
            )
        )

    joins: list[SqlRiskJoinStructure] = []
    for join in statement.find_all(exp.Join):
        on = join.args.get("on")
        condition = on if isinstance(on, exp.Expression) else None
        using = join.args.get("using") or []
        if not isinstance(using, list):
            using = [using]
        join_type = " ".join(
            part
            for part in (
                str(join.side or "").upper(),
                str(join.kind or "").upper(),
                "JOIN",
            )
            if part
        )
        joins.append(
            SqlRiskJoinStructure(
                query_block_id=query_id(join),
                join_type=join_type,
                relation=_sql(join.this),
                predicate=_sql(condition) or None,
                using_columns=[
                    _sql(item) if isinstance(item, exp.Expression) else str(item)
                    for item in using
                ],
                equalities=[
                    SqlRiskEqualityStructure(
                        sql=_sql(equality),
                        left=_sql(equality.this),
                        right=_sql(equality.expression),
                    )
                    for equality in (
                        condition.find_all(exp.EQ) if condition is not None else ()
                    )
                ],
                referenced_columns=(
                    _unique([_sql(column) for column in condition.find_all(exp.Column)])
                    if condition is not None
                    else []
                ),
            )
        )

    projections: list[SqlRiskProjectionStructure] = []
    distinct_query_blocks: list[str] = []
    for select in selects:
        block_id = query_ids[id(select)]
        if select.args.get("distinct") is not None:
            distinct_query_blocks.append(block_id)
        for projection in select.expressions:
            value = projection.this if isinstance(projection, exp.Alias) else projection
            columns = _unique(
                [_sql(column) for column in value.find_all(exp.Column)]
            )
            has_wildcard = isinstance(value, exp.Star) or any(
                isinstance(node, exp.Star)
                or (isinstance(node, exp.Column) and node.is_star)
                for node in value.walk()
            )
            projections.append(
                SqlRiskProjectionStructure(
                    query_block_id=block_id,
                    sql=_sql(projection),
                    expression=_sql(value),
                    expression_kind=type(value).__name__,
                    output_name=str(projection.alias_or_name or "").strip() or None,
                    explicit_alias=str(projection.alias or "").strip() or None,
                    referenced_columns=columns,
                    has_wildcard=has_wildcard,
                )
            )

    def scoped_nodes(
        node_type: type[exp.Expression],
        *,
        expressions: bool = False,
        argument: str = "this",
    ) -> List[SqlRiskScopedExpression]:
        result: list[SqlRiskScopedExpression] = []
        for node in statement.find_all(node_type):
            candidates: Sequence[Any]
            if expressions:
                candidates = list(node.expressions)
            else:
                candidates = [node.args.get(argument)]
            for candidate in candidates:
                if isinstance(candidate, exp.Expression):
                    result.append(
                        SqlRiskScopedExpression(
                            query_block_id=query_id(node),
                            sql=_sql(candidate),
                        )
                    )
        return result

    set_operations = [
        SqlRiskSetOperationStructure(
            operation=str(getattr(node, "key", "") or type(node).__name__).upper(),
            distinct=(
                bool(node.args["distinct"])
                if node.args.get("distinct") is not None
                else None
            ),
            by_name=bool(node.args.get("by_name")),
        )
        for node in _set_operations(statement)
    ]
    return SqlRiskStatementStructure(
        statement_index=statement_index,
        statement_kind=str(
            getattr(statement, "key", "") or type(statement).__name__
        ).upper(),
        normalized_sql=_sql(statement),
        write_targets=write_targets,
        sources=sources,
        cte_names=cte_names,
        joins=joins,
        filters=scoped_nodes(exp.Where),
        having=scoped_nodes(exp.Having),
        qualify=scoped_nodes(exp.Qualify),
        distinct_query_blocks=distinct_query_blocks,
        group_by=scoped_nodes(exp.Group, expressions=True),
        order_by=scoped_nodes(exp.Order, expressions=True),
        limits=scoped_nodes(exp.Limit, argument="expression"),
        offsets=scoped_nodes(exp.Offset, argument="expression"),
        set_operations=set_operations,
        projections=projections,
        referenced_columns=_unique(
            [_sql(column) for column in statement.find_all(exp.Column)]
        ),
    )


def _parse_rule(
    raw_sql: str | None,
    *,
    rule_id: str,
    occurrence_count: int,
    evidence_ids: Sequence[str],
    mappings: Sequence[SqlRiskMappingEndpoint],
) -> tuple[SqlRiskRuleStructure, SqlRiskStructureIssue | None]:
    if raw_sql is None or not raw_sql.strip():
        message = "transformation_rule is empty"
        return (
            SqlRiskRuleStructure(
                rule_id=rule_id,
                raw_sql=raw_sql,
                occurrence_count=occurrence_count,
                evidence_ids=list(evidence_ids),
                mappings=list(mappings),
                parse_status="error",
                parse_error=message,
            ),
            _issue("sql_parse_error", message, rule_id=rule_id),
        )
    try:
        statements = sqlglot.parse(
            quote_dollar_schemas(raw_sql),
            read=GREENPLUM_DIALECT,
        )
        if not statements or any(statement is None for statement in statements):
            raise ValueError("SQLGlot returned no complete statement")
        structures = [
            _statement_structure(statement, statement_index=index)
            for index, statement in enumerate(statements, 1)
            if statement is not None
        ]
    except (SqlglotError, TypeError, ValueError) as exc:
        message = f"{type(exc).__name__}: {exc}"
        return (
            SqlRiskRuleStructure(
                rule_id=rule_id,
                raw_sql=raw_sql,
                occurrence_count=occurrence_count,
                evidence_ids=list(evidence_ids),
                mappings=list(mappings),
                parse_status="error",
                parse_error=message,
            ),
            _issue("sql_parse_error", message, rule_id=rule_id),
        )
    normalized_sql = "; ".join(item.normalized_sql for item in structures)
    return (
        SqlRiskRuleStructure(
            rule_id=rule_id,
            raw_sql=raw_sql,
            occurrence_count=occurrence_count,
            evidence_ids=list(evidence_ids),
            mappings=list(mappings),
            parse_status="ok",
            normalized_sql=normalized_sql,
            statements=structures,
        ),
        None,
    )


def _metadata_value(
    row: Mapping[str, Any],
    key: str,
) -> str | int | float | bool | None:
    value = row.get(key)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise ValueError(f"metadata {key!r} must be a JSON scalar")


def _required_text(row: Mapping[str, Any], key: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key!r} must be a non-empty string")
    return value


def _optional_text(row: Mapping[str, Any], key: str) -> str | None:
    value = row.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{key!r} must be a string or null")
    return value


def _unavailable_bundle(
    contract: SqlRiskScopeContract,
    *,
    evidence: Sequence[SqlRiskEvidenceStructure] = (),
    mapping_row_count: int = 0,
    mapping_source_total: int | None = None,
    metadata_row_count: int = 0,
    metadata_source_total: int | None = None,
    issues: Sequence[SqlRiskStructureIssue],
) -> SqlRiskStructureBundle:
    return SqlRiskStructureBundle(
        status="unavailable",
        execution_mode=contract.execution_mode,
        scope=_scope_structure(contract),
        evidence=list(evidence),
        mapping_row_count=mapping_row_count,
        mapping_source_total=mapping_source_total,
        metadata_row_count=metadata_row_count,
        metadata_source_total=metadata_source_total,
        issues=list(issues),
    )


def build_sql_risk_structure_bundle(
    contract: SqlRiskScopeContract,
    evidence: Sequence[SqlRiskStructuralEvidence],
    *,
    max_bundle_chars: int = SQL_RISK_STRUCTURE_MAX_CHARS,
) -> SqlRiskStructureBundle:
    """Build a complete neutral structure bundle from exact reader payloads.

    Evidence must match every contract requirement by tool name and exact
    argument mapping. Packed reader payloads are decoded losslessly. Any
    missing, duplicated, truncated, malformed, or unexpected evidence makes
    the bundle unavailable. SQL parse errors also make it unavailable: the
    assessment model is never asked to infer risk from unparsed SQL.
    """

    if not isinstance(contract, SqlRiskScopeContract) or str(
        getattr(contract, "execution_mode", "")
    ) not in _CLOSED_EXECUTION_MODES:
        # A non-contract object has no safe shape from which to build the
        # required immutable scope. Keep this programmer error explicit.
        if not isinstance(contract, SqlRiskScopeContract):
            raise TypeError("contract must be a SqlRiskScopeContract")
        return _unavailable_bundle(
            contract,
            issues=[
                _issue(
                    "invalid_contract",
                    "execution_mode is not a closed SQL-risk scope mode",
                )
            ],
        )
    if isinstance(max_bundle_chars, bool) or max_bundle_chars <= 0:
        raise ValueError("max_bundle_chars must be a positive integer")

    requirements = list(contract.requirements)
    requirement_keys = [
        (requirement.tool_name, dict(requirement.arguments))
        for requirement in requirements
    ]
    if len({(name, repr(sorted(args.items()))) for name, args in requirement_keys}) != len(
        requirement_keys
    ):
        return _unavailable_bundle(
            contract,
            issues=[
                _issue(
                    "invalid_contract",
                    "contract contains duplicate exact-reader requirements",
                )
            ],
        )

    input_items = list(evidence)
    issues: list[SqlRiskStructureIssue] = []
    evidence_ids: set[str] = set()
    matched: dict[int, SqlRiskStructuralEvidence] = {}
    for item in input_items:
        if not isinstance(item, SqlRiskStructuralEvidence):
            raise TypeError("evidence items must be SqlRiskStructuralEvidence")
        if item.evidence_id in evidence_ids:
            issues.append(
                _issue(
                    "duplicate_evidence",
                    "evidence_id is repeated",
                    evidence_id=item.evidence_id,
                )
            )
            continue
        evidence_ids.add(item.evidence_id)
        indexes = [
            index
            for index, (tool_name, arguments) in enumerate(requirement_keys)
            if item.tool_name == tool_name and item.arguments == arguments
        ]
        if len(indexes) != 1:
            issues.append(
                _issue(
                    "unexpected_evidence",
                    "reader identity or arguments do not match the contract",
                    evidence_id=item.evidence_id,
                )
            )
            continue
        index = indexes[0]
        if index in matched:
            issues.append(
                _issue(
                    "duplicate_evidence",
                    "more than one payload satisfies the same requirement",
                    evidence_id=item.evidence_id,
                )
            )
            continue
        matched[index] = item

    for index, requirement in enumerate(requirements):
        if index not in matched:
            issues.append(
                _issue(
                    "missing_evidence",
                    f"missing exact payload for {requirement.tool_name}",
                )
            )
    if issues:
        return _unavailable_bundle(contract, issues=issues)

    decoded: dict[int, _DecodedRows] = {}
    summaries: list[SqlRiskEvidenceStructure] = []
    mapping_rows: list[tuple[str, Mapping[str, Any]]] = []
    metadata_pairs: list[tuple[str, Mapping[str, Any]]] = []
    mapping_source_total: int | None = None
    metadata_source_total: int | None = None
    for index in range(len(requirements)):
        item = matched[index]
        try:
            result = _decode_rows(item)
        except RuntimeError as exc:
            issues.append(
                _issue(
                    "incomplete_payload",
                    str(exc),
                    evidence_id=item.evidence_id,
                )
            )
            continue
        except ValueError as exc:
            issues.append(
                _issue(
                    "invalid_payload",
                    str(exc),
                    evidence_id=item.evidence_id,
                )
            )
            continue
        decoded[index] = result
        summaries.append(
            SqlRiskEvidenceStructure(
                evidence_id=item.evidence_id,
                tool_name=item.tool_name,
                arguments=dict(item.arguments),
                returned_rows=len(result.rows),
                source_total=result.source_total,
            )
        )
        columns = set(result.columns)
        if item.tool_name in _MAPPING_TOOLS:
            if not _MAPPING_REQUIRED_COLUMNS.issubset(columns):
                missing = sorted(_MAPPING_REQUIRED_COLUMNS - columns)
                issues.append(
                    _issue(
                        "invalid_payload",
                        "mapping payload is missing columns: " + ", ".join(missing),
                        evidence_id=item.evidence_id,
                    )
                )
                continue
            mapping_rows.extend((item.evidence_id, row) for row in result.rows)
            mapping_source_total = result.source_total
        elif item.tool_name == _METADATA_TOOL:
            if not _METADATA_REQUIRED_COLUMNS.issubset(columns):
                missing = sorted(_METADATA_REQUIRED_COLUMNS - columns)
                issues.append(
                    _issue(
                        "invalid_payload",
                        "metadata payload is missing columns: " + ", ".join(missing),
                        evidence_id=item.evidence_id,
                    )
                )
                continue
            metadata_pairs.extend((item.evidence_id, row) for row in result.rows)
            metadata_source_total = result.source_total
        else:  # pragma: no cover - closed SqlRiskToolName union
            issues.append(
                _issue(
                    "unexpected_evidence",
                    f"unsupported reader {item.tool_name}",
                    evidence_id=item.evidence_id,
                )
            )

    if issues:
        return _unavailable_bundle(
            contract,
            evidence=summaries,
            mapping_row_count=len(mapping_rows),
            mapping_source_total=mapping_source_total,
            metadata_row_count=len(metadata_pairs),
            metadata_source_total=metadata_source_total,
            issues=issues,
        )

    if not mapping_rows:
        return _unavailable_bundle(
            contract,
            evidence=summaries,
            mapping_source_total=mapping_source_total,
            metadata_row_count=len(metadata_pairs),
            metadata_source_total=metadata_source_total,
            issues=[
                _issue(
                    "incomplete_payload",
                    "exact directed S2T scope contains no mapping rows",
                )
            ],
        )

    expected_source = contract.scope.source_table.casefold()
    expected_target = contract.scope.target_table.casefold()
    if any(
        str(row.get("source_table") or "").casefold() != expected_source
        or str(row.get("target_table") or "").casefold() != expected_target
        for _, row in mapping_rows
    ):
        return _unavailable_bundle(
            contract,
            evidence=summaries,
            mapping_row_count=len(mapping_rows),
            mapping_source_total=mapping_source_total,
            metadata_row_count=len(metadata_pairs),
            metadata_source_total=metadata_source_total,
            issues=[
                _issue(
                    "unexpected_evidence",
                    "mapping payload contains rows outside the exact directed scope",
                )
            ],
        )

    if contract.scope.is_field_pair:
        expected_source_field = str(contract.scope.source_field).casefold()
        expected_target_field = str(contract.scope.target_field).casefold()
        if not mapping_rows or not all(
            str(row.get("source_field") or "").casefold()
            == expected_source_field
            and str(row.get("target_field") or "").casefold()
            == expected_target_field
            for _, row in mapping_rows
        ):
            return _unavailable_bundle(
                contract,
                evidence=summaries,
                mapping_row_count=len(mapping_rows),
                mapping_source_total=mapping_source_total,
                metadata_row_count=len(metadata_pairs),
                metadata_source_total=metadata_source_total,
                issues=[
                    _issue(
                        "incomplete_payload",
                        "exact field pair is absent from the directed S2T mapping",
                    )
                ],
            )

    if contract.execution_mode == "nullable_constraint":
        expected_file_id = contract.scope.file_id
        expected_metadata = {
            (
                "source",
                expected_file_id,
                contract.scope.source_table.casefold(),
                str(contract.scope.source_field).casefold(),
            ),
            (
                "target",
                expected_file_id,
                contract.scope.target_table.casefold(),
                str(contract.scope.target_field).casefold(),
            ),
        }
        observed_metadata = [
            (
                str(row.get("column_role") or "").casefold(),
                row.get("file_id"),
                str(row.get("table_name") or "").casefold(),
                str(row.get("column_name") or "").casefold(),
            )
            for _, row in metadata_pairs
        ]
        if set(observed_metadata) != expected_metadata:
            return _unavailable_bundle(
                contract,
                evidence=summaries,
                mapping_row_count=len(mapping_rows),
                mapping_source_total=mapping_source_total,
                metadata_row_count=len(metadata_pairs),
                metadata_source_total=metadata_source_total,
                issues=[
                    _issue(
                        "incomplete_payload",
                        "nullable scope requires at least one exact source and "
                        "target metadata row and no rows outside that scope",
                    )
                ],
            )

    # Preserve exact SQL identity and first-occurrence order. ``None`` is an
    # explicit empty rule value; no whitespace normalization is used for the
    # grouping key.
    grouped_rules: list[dict[str, Any]] = []
    rule_indexes: dict[str | None, int] = {}
    try:
        for evidence_id, row in mapping_rows:
            raw_sql = row.get("transformation_rule")
            if raw_sql is not None and not isinstance(raw_sql, str):
                raise ValueError("transformation_rule must be a string or null")
            source_table = _required_text(row, "source_table")
            target_table = _required_text(row, "target_table")
            source_field = _optional_text(row, "source_field")
            target_field = _optional_text(row, "target_field")
            if raw_sql not in rule_indexes:
                rule_indexes[raw_sql] = len(grouped_rules)
                grouped_rules.append(
                    {
                        "raw_sql": raw_sql,
                        "count": 0,
                        "evidence_ids": [],
                        "mapping_counts": {},
                    }
                )
            group = grouped_rules[rule_indexes[raw_sql]]
            group["count"] += 1
            if evidence_id not in group["evidence_ids"]:
                group["evidence_ids"].append(evidence_id)
            mapping_key = (
                source_table,
                source_field,
                target_table,
                target_field,
            )
            current = group["mapping_counts"].setdefault(
                mapping_key,
                {"count": 0, "evidence_ids": []},
            )
            current["count"] += 1
            if evidence_id not in current["evidence_ids"]:
                current["evidence_ids"].append(evidence_id)
    except ValueError as exc:
        return _unavailable_bundle(
            contract,
            evidence=summaries,
            mapping_row_count=len(mapping_rows),
            mapping_source_total=mapping_source_total,
            metadata_row_count=len(metadata_pairs),
            metadata_source_total=metadata_source_total,
            issues=[_issue("invalid_payload", str(exc))],
        )

    rules: list[SqlRiskRuleStructure] = []
    parse_issues: list[SqlRiskStructureIssue] = []
    for index, group in enumerate(grouped_rules, 1):
        mappings = [
            SqlRiskMappingEndpoint(
                source_table=key[0],
                source_field=key[1],
                target_table=key[2],
                target_field=key[3],
                occurrence_count=value["count"],
                evidence_ids=list(value["evidence_ids"]),
            )
            for key, value in group["mapping_counts"].items()
        ]
        rule, parse_issue = _parse_rule(
            group["raw_sql"],
            rule_id=f"sql_rule_{index}",
            occurrence_count=group["count"],
            evidence_ids=group["evidence_ids"],
            mappings=mappings,
        )
        rules.append(rule)
        if parse_issue is not None:
            parse_issues.append(parse_issue)

    decoded_mapping_rows: list[SqlRiskMappingRow] = []
    metadata_rows: list[SqlRiskColumnMetadataRow] = []
    try:
        for row_index, (evidence_id, row) in enumerate(mapping_rows, 1):
            decoded_mapping_rows.append(
                SqlRiskMappingRow(
                    evidence_id=evidence_id,
                    row_index=row_index,
                    values=dict(row),
                )
            )
        for row_index, (evidence_id, row) in enumerate(metadata_pairs, 1):
            file_id = row.get("file_id")
            if (
                isinstance(file_id, bool)
                or not isinstance(file_id, int)
                or file_id <= 0
            ):
                raise ValueError("metadata file_id must be a positive integer")
            data_type = _optional_text(row, "data_type")
            description = _optional_text(row, "description")
            metadata_rows.append(
                SqlRiskColumnMetadataRow(
                    evidence_id=evidence_id,
                    row_index=row_index,
                    column_role=_required_text(row, "column_role"),
                    file_id=file_id,
                    table_name=_required_text(row, "table_name"),
                    column_name=_required_text(row, "column_name"),
                    data_type=data_type,
                    primary_key=_metadata_value(row, "primary_key"),
                    not_null=_metadata_value(row, "not_null"),
                    description=description,
                )
            )
    except (ValueError, ValidationError) as exc:
        return _unavailable_bundle(
            contract,
            evidence=summaries,
            mapping_row_count=len(mapping_rows),
            mapping_source_total=mapping_source_total,
            metadata_row_count=len(metadata_pairs),
            metadata_source_total=metadata_source_total,
            issues=[_issue("invalid_payload", str(exc))],
        )

    bundle = SqlRiskStructureBundle(
        status="unavailable" if parse_issues else "ready",
        execution_mode=contract.execution_mode,
        scope=_scope_structure(contract),
        evidence=summaries,
        mapping_row_count=len(mapping_rows),
        mapping_source_total=mapping_source_total,
        metadata_row_count=len(metadata_pairs),
        metadata_source_total=metadata_source_total,
        mapping_rows=decoded_mapping_rows,
        rules=rules,
        metadata_rows=metadata_rows,
        issues=parse_issues,
    )
    serialized_chars = len(bundle.model_dump_json())
    if serialized_chars > max_bundle_chars:
        return _unavailable_bundle(
            contract,
            evidence=summaries,
            mapping_row_count=len(mapping_rows),
            mapping_source_total=mapping_source_total,
            metadata_row_count=len(metadata_pairs),
            metadata_source_total=metadata_source_total,
            issues=[
                *parse_issues,
                _issue(
                    "bundle_too_large",
                    "complete structural bundle contains "
                    f"{serialized_chars} characters; limit is {max_bundle_chars}",
                ),
            ],
        )
    return bundle


def sql_risk_structure_assessment_payload(
    bundle: SqlRiskStructureBundle,
) -> Dict[str, Any]:
    """Convert a usable neutral bundle to the bounded assessment input shape.

    No fields are reinterpreted or summarized here. An unavailable bundle is
    rejected so a caller cannot silently ask the model to assess partial or
    invalid evidence as though it were complete.
    """

    if bundle.status == "unavailable":
        raise ValueError("unavailable structural bundle cannot be assessed")
    return bundle.model_dump(mode="json")


__all__ = [
    "SQL_RISK_ASSESSMENT_ENVELOPE_RESERVE_CHARS",
    "SQL_RISK_STRUCTURE_MAX_CHARS",
    "SqlRiskColumnMetadataRow",
    "SqlRiskEqualityStructure",
    "SqlRiskEvidenceStructure",
    "SqlRiskJoinStructure",
    "SqlRiskMappingRow",
    "SqlRiskMappingEndpoint",
    "SqlRiskProjectionStructure",
    "SqlRiskRelationStructure",
    "SqlRiskRuleStructure",
    "SqlRiskScopeStructure",
    "SqlRiskScopedExpression",
    "SqlRiskSetOperationStructure",
    "SqlRiskStatementStructure",
    "SqlRiskStructuralEvidence",
    "SqlRiskStructureBundle",
    "SqlRiskStructureIssue",
    "build_sql_risk_structure_bundle",
    "sql_risk_structure_assessment_payload",
]
