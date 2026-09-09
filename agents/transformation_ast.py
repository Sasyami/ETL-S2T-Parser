"""Deterministic normalization of complete Greenplum transformation queries."""

from __future__ import annotations

import re
from typing import Dict, List, Literal, Optional

import sqlglot
from pydantic import BaseModel, ConfigDict, Field
from sqlglot import exp
from sqlglot.errors import SqlglotError

from services.sql_dialects import GREENPLUM_DIALECT  # noqa: F401


class NormalizedProjection(BaseModel):
    """One named output of the outer transformation query."""

    model_config = ConfigDict(extra="forbid")

    output_name: str
    expression: str
    source_columns: List[str] = Field(default_factory=list)


class NormalizedJoin(BaseModel):
    """One join retained from the parsed query."""

    model_config = ConfigDict(extra="forbid")

    join_type: str
    relation: str
    condition: str = ""


class NormalizedTransformation(BaseModel):
    """Stable SQLGlot-derived facts used by protocol compilers."""

    model_config = ConfigDict(extra="forbid")

    parse_status: Literal["ok", "error"]
    original_sql: str
    query_sql: str = ""
    sources: List[str] = Field(default_factory=list)
    outer_source_aliases: Dict[str, str] = Field(default_factory=dict)
    projections: Dict[str, str] = Field(default_factory=dict)
    projection_details: List[NormalizedProjection] = Field(default_factory=list)
    joins: List[NormalizedJoin] = Field(default_factory=list)
    filters: List[str] = Field(default_factory=list)
    grouping: List[str] = Field(default_factory=list)
    has_wildcard: bool = False
    has_set_operation: bool = False
    error: Optional[str] = None


class FieldValueChangeAnalysis(BaseModel):
    """Field-scoped value semantics derived from one outer SQL projection."""

    model_config = ConfigDict(extra="forbid")

    source_field: str
    target_field: str
    status: Literal[
        "direct_projection",
        "value_expression",
        "different_source_column",
        "source_provenance_unknown",
        "target_projection_missing",
        "ambiguous_target_projection",
        "set_operation_unsupported",
        "parse_error",
    ]
    may_change_value: Optional[bool] = None
    target_expression: Optional[str] = None
    source_columns: List[str] = Field(default_factory=list)
    expression_kind: Optional[str] = None


def quote_dollar_schemas(sql: str) -> str:
    """Quote project-specific ``$$schema`` names for Greenplum parsing."""

    return re.sub(
        r"(?<![\w$])\$\$([A-Za-z0-9_]+)(?=\.)",
        lambda match: f'"$${match.group(1)}"',
        str(sql or ""),
    )


def _qualified_table_name(table: exp.Table) -> str:
    return ".".join(
        part for part in (table.catalog, table.db, table.name) if part
    )


def _outer_select(query: exp.Query) -> exp.Select | None:
    current: exp.Expression = query
    while isinstance(current, (exp.Subquery, exp.Paren)):
        current = current.this
    if isinstance(current, exp.Select):
        return current
    if isinstance(current, (exp.Union, exp.Intersect, exp.Except)):
        branch = current.this
        while isinstance(branch, (exp.Subquery, exp.Paren)):
            branch = branch.this
        return branch if isinstance(branch, exp.Select) else branch.find(exp.Select)
    return current.find(exp.Select)


def _sql(node: exp.Expression | None) -> str:
    return node.sql(dialect=GREENPLUM_DIALECT) if node is not None else ""


def _nearest_select(node: exp.Expression) -> exp.Select | None:
    current = node.parent
    while current is not None:
        if isinstance(current, exp.Select):
            return current
        current = current.parent
    return None


def _outer_source_aliases(
    outer: exp.Select,
    *,
    cte_names: set[str],
) -> Dict[str, str]:
    """Map unambiguous outer-scope SQL aliases to physical relations."""

    candidates: Dict[str, set[str]] = {}
    for table in outer.find_all(exp.Table):
        if _nearest_select(table) is not outer:
            continue
        if table.name.casefold() in cte_names:
            # Resolving a CTE projection to a physical source requires lineage,
            # not a local name comparison.  Stay conservative here.
            continue
        relation = _qualified_table_name(table)
        if not relation:
            continue
        keys = {
            str(table.alias_or_name or "").strip(),
            str(table.name or "").strip(),
            relation,
        }
        for key in keys:
            if key:
                candidates.setdefault(key.casefold(), set()).add(relation)
    return {
        alias: next(iter(relations))
        for alias, relations in candidates.items()
        if len(relations) == 1
    }


def normalize_transformation(rule: str) -> NormalizedTransformation:
    """Parse exactly one complete read query and expose reusable normalized facts."""

    original = str(rule or "").strip()
    parseable = quote_dollar_schemas(original)
    try:
        statements = sqlglot.parse(parseable, read=GREENPLUM_DIALECT)
    except (SqlglotError, ValueError) as exc:
        return NormalizedTransformation(
            parse_status="error",
            original_sql=original,
            error=(
                "SQLGlot не разобрал transformation_rule: "
                f"{type(exc).__name__}: {str(exc)[:300]}"
            ),
        )
    if len(statements) != 1 or not isinstance(statements[0], exp.Query):
        return NormalizedTransformation(
            parse_status="error",
            original_sql=original,
            error="transformation_rule не является одним полным SELECT/WITH query",
        )

    query = statements[0]
    outer = _outer_select(query)
    if outer is None:
        return NormalizedTransformation(
            parse_status="error",
            original_sql=original,
            error="SQLGlot не нашёл внешний SELECT в transformation_rule",
        )

    cte_names = {
        str(cte.alias_or_name or "").casefold()
        for cte in query.find_all(exp.CTE)
        if str(cte.alias_or_name or "").strip()
    }
    projections: Dict[str, str] = {}
    details: List[NormalizedProjection] = []
    has_wildcard = False
    for projection in outer.expressions:
        value = projection.this if isinstance(projection, exp.Alias) else projection
        if isinstance(value, exp.Star) or (
            isinstance(value, exp.Column) and value.is_star
        ):
            has_wildcard = True
            continue
        output_name = str(projection.alias_or_name or "").strip()
        if not output_name:
            continue
        expression_sql = _sql(value)
        source_columns = list(
            dict.fromkeys(
                _sql(column)
                for column in value.find_all(exp.Column)
                if not column.is_star
            )
        )
        projections[output_name] = expression_sql
        details.append(
            NormalizedProjection(
                output_name=output_name,
                expression=expression_sql,
                source_columns=source_columns,
            )
        )

    sources = list(
        dict.fromkeys(
            name
            for table in query.find_all(exp.Table)
            if (name := _qualified_table_name(table))
            and table.name.casefold() not in cte_names
        )
    )
    joins = [
        NormalizedJoin(
            join_type=" ".join(
                part
                for part in (
                    str(join.side or "").upper(),
                    str(join.kind or "").upper(),
                    "JOIN",
                )
                if part
            ),
            relation=_sql(join.this),
            condition=_sql(join.args.get("on")),
        )
        for join in query.find_all(exp.Join)
    ]
    filters = [
        *(_sql(where.this) for where in query.find_all(exp.Where)),
        *(_sql(having.this) for having in query.find_all(exp.Having)),
    ]
    grouping = [
        _sql(item)
        for group in query.find_all(exp.Group)
        for item in group.expressions
    ]
    return NormalizedTransformation(
        parse_status="ok",
        original_sql=original,
        query_sql=_sql(query),
        sources=sources,
        outer_source_aliases=_outer_source_aliases(
            outer,
            cte_names=cte_names,
        ),
        projections=projections,
        projection_details=details,
        joins=joins,
        filters=filters,
        grouping=grouping,
        has_wildcard=has_wildcard,
        has_set_operation=isinstance(
            query,
            (exp.Union, exp.Intersect, exp.Except),
        ),
    )


def analyze_field_value_change(
    normalized: NormalizedTransformation,
    *,
    source_table: str,
    source_field: str,
    target_field: str,
) -> FieldValueChangeAnalysis:
    """Classify only the exact target projection for one S2T field pair.

    The classifier is deliberately structural.  It never transfers a function
    from a neighbouring output alias to the requested target field and it does
    not claim semantic equivalence for arbitrary expressions.  A plain column
    with the requested source-field name is the sole ``direct_projection``
    case; everything else remains an explicit expression or an unavailable
    assessment.
    """

    clean_source_table = str(source_table or "").strip()
    clean_source = str(source_field or "").strip()
    clean_target = str(target_field or "").strip()
    if normalized.parse_status != "ok":
        return FieldValueChangeAnalysis(
            source_field=clean_source,
            target_field=clean_target,
            status="parse_error",
        )
    if normalized.has_set_operation:
        return FieldValueChangeAnalysis(
            source_field=clean_source,
            target_field=clean_target,
            status="set_operation_unsupported",
        )

    matching = [
        detail
        for detail in normalized.projection_details
        if detail.output_name.casefold() == clean_target.casefold()
    ]
    if not matching:
        return FieldValueChangeAnalysis(
            source_field=clean_source,
            target_field=clean_target,
            status="target_projection_missing",
        )
    if len(matching) != 1:
        return FieldValueChangeAnalysis(
            source_field=clean_source,
            target_field=clean_target,
            status="ambiguous_target_projection",
        )

    detail = matching[0]
    try:
        expression = sqlglot.parse_one(
            detail.expression,
            read=GREENPLUM_DIALECT,
        )
    except (SqlglotError, TypeError, ValueError):
        return FieldValueChangeAnalysis(
            source_field=clean_source,
            target_field=clean_target,
            status="parse_error",
            target_expression=detail.expression,
            source_columns=list(detail.source_columns),
        )

    expression_kind = type(expression).__name__
    if isinstance(expression, exp.Column):
        qualifier = str(expression.table or "").strip().casefold()
        aliases = normalized.outer_source_aliases
        resolved_relation: Optional[str] = None
        if qualifier:
            resolved_relation = aliases.get(qualifier)
        else:
            relations = set(aliases.values())
            if len(relations) == 1:
                resolved_relation = next(iter(relations))

        expected = clean_source_table.casefold()
        resolved = str(resolved_relation or "").casefold()
        table_matches = bool(resolved and expected) and (
            resolved == expected
            or ("." not in expected and resolved.endswith("." + expected))
        )
        if expression.name.casefold() == clean_source.casefold():
            if table_matches:
                return FieldValueChangeAnalysis(
                    source_field=clean_source,
                    target_field=clean_target,
                    status="direct_projection",
                    may_change_value=False,
                    target_expression=detail.expression,
                    source_columns=list(detail.source_columns),
                    expression_kind=expression_kind,
                )
            return FieldValueChangeAnalysis(
                source_field=clean_source,
                target_field=clean_target,
                status="source_provenance_unknown",
                target_expression=detail.expression,
                source_columns=list(detail.source_columns),
                expression_kind=expression_kind,
            )
        return FieldValueChangeAnalysis(
            source_field=clean_source,
            target_field=clean_target,
            status="different_source_column",
            may_change_value=True,
            target_expression=detail.expression,
            source_columns=list(detail.source_columns),
            expression_kind=expression_kind,
        )

    return FieldValueChangeAnalysis(
        source_field=clean_source,
        target_field=clean_target,
        status="value_expression",
        may_change_value=True,
        target_expression=detail.expression,
        source_columns=list(detail.source_columns),
        expression_kind=expression_kind,
    )


__all__ = [
    "FieldValueChangeAnalysis",
    "NormalizedJoin",
    "NormalizedProjection",
    "NormalizedTransformation",
    "analyze_field_value_change",
    "normalize_transformation",
    "quote_dollar_schemas",
]
