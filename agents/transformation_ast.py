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
    projections: Dict[str, str] = Field(default_factory=dict)
    projection_details: List[NormalizedProjection] = Field(default_factory=list)
    joins: List[NormalizedJoin] = Field(default_factory=list)
    filters: List[str] = Field(default_factory=list)
    grouping: List[str] = Field(default_factory=list)
    has_wildcard: bool = False
    error: Optional[str] = None


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

    cte_names = {
        str(cte.alias_or_name or "").casefold()
        for cte in query.find_all(exp.CTE)
        if str(cte.alias_or_name or "").strip()
    }
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
        projections=projections,
        projection_details=details,
        joins=joins,
        filters=filters,
        grouping=grouping,
        has_wildcard=has_wildcard,
    )


__all__ = [
    "NormalizedJoin",
    "NormalizedProjection",
    "NormalizedTransformation",
    "normalize_transformation",
    "quote_dollar_schemas",
]
