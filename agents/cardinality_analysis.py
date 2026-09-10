"""Deterministic conditional-cardinality facts from exact saved S2T rows.

The analysis is deliberately narrow.  It identifies JOINs in the outer
transformation SELECT as a possible row-multiplication mechanism, but never
turns an unknown join-key uniqueness property into a factual duplicate claim.
A provably false outer predicate fails closed; other filters and scalar
projections are outside this contract.
"""

from __future__ import annotations

import json
from typing import Dict, List, Literal, Sequence

import sqlglot
from pydantic import BaseModel, ConfigDict, Field, model_validator
from sqlglot import exp
from sqlglot.errors import SqlglotError

from services.sql_dialects import GREENPLUM_DIALECT  # noqa: F401

from .cardinality_sufficiency import (
    ExactTablePair,
    cardinality_pair_from_contract,
)
from .contracts import EvidenceArtifact
from .sql_risk_scope_contract import SqlRiskScopeContract
from .tools.saved_results import SavedResultStore
from .transformation_ast import quote_dollar_schemas


CardinalityConclusion = Literal[
    "conditional_duplicate_risk",
    "join_mechanism_not_detected",
    "not_assessed",
    "conflicting",
]
CardinalityMechanism = Literal[
    "join_fanout",
    "no_join",
    "constant_false_predicate",
    "unparseable_rule",
    "set_operation",
    "unsupported_join",
    "no_exact_mapping",
    "incomplete_evidence",
    "missing_transformation_rule",
    "scope_mismatch",
    "conflicting_rules",
]
CardinalityCondition = Literal[
    "full_join_key_uniqueness_unknown",
    "join_match_multiplicity_unknown",
    "not_applicable",
    "unknown",
]
CardinalityRuleStatus = Literal[
    "join_detected",
    "no_join",
    "constant_false_predicate",
    "unparseable_rule",
    "set_operation",
    "unsupported_join",
]

_EXACT_DIRECTED_TOOLS = frozenset(
    {"read_s2t_mapping", "read_s2t_source_to_target"}
)
_REQUIRED_COLUMNS = frozenset(
    {"source_table", "target_table", "transformation_rule"}
)
_SUPPORTED_JOIN_SIDES = frozenset({"", "LEFT", "RIGHT", "FULL"})
_SUPPORTED_JOIN_KINDS = frozenset({"", "INNER", "CROSS", "OUTER"})
_SUPPORTED_JOIN_METHODS = frozenset({"", "NATURAL"})
_MAX_FACTS = 4
_MAX_JOINS = 4
_MAX_JOIN_KEYS = 8
_MAX_DISTINCT_RULES = 100
_MAX_IDENTIFIER_CHARS = 200
_MAX_RELATION_CHARS = 240
_MAX_PREDICATE_CHARS = 600
_MAX_JOIN_KEY_CHARS = 240
_MAX_EVIDENCE_IDS = 8
MAX_CARDINALITY_PAYLOAD_CHARS = 12_000


class CardinalityJoinFact(BaseModel):
    """One outer-query JOIN and the condition still needed to prove fanout."""

    model_config = ConfigDict(extra="forbid")

    join_type: str = Field(min_length=1, max_length=48)
    relation: str = Field(min_length=1, max_length=_MAX_RELATION_CHARS)
    predicate: str = Field(default="", max_length=_MAX_PREDICATE_CHARS)
    join_key_equalities: List[str] = Field(
        default_factory=list,
        max_length=_MAX_JOIN_KEYS,
    )
    uniqueness_condition: Literal[
        "full_join_key_uniqueness_unknown",
        "join_match_multiplicity_unknown",
    ]


class CardinalityRuleAnalysis(BaseModel):
    """SQLGlot classification of one complete transformation rule."""

    model_config = ConfigDict(extra="forbid")

    status: CardinalityRuleStatus
    joins: List[CardinalityJoinFact] = Field(
        default_factory=list,
        max_length=_MAX_JOINS,
    )


class CardinalityFact(BaseModel):
    """Code-derived cardinality conclusion for one exact directed mapping."""

    model_config = ConfigDict(extra="forbid")

    source_table: str = Field(min_length=1, max_length=_MAX_IDENTIFIER_CHARS)
    target_table: str = Field(min_length=1, max_length=_MAX_IDENTIFIER_CHARS)
    conclusion: CardinalityConclusion
    mechanism: CardinalityMechanism
    condition: CardinalityCondition
    matching_rows: int = Field(ge=0)
    joins: List[CardinalityJoinFact] = Field(
        default_factory=list,
        max_length=_MAX_JOINS,
    )
    evidence_ids: List[str] = Field(
        default_factory=list,
        max_length=_MAX_EVIDENCE_IDS,
    )

    @model_validator(mode="after")
    def validate_cardinality_claim(self) -> "CardinalityFact":
        """Prevent a positive claim without a concrete outer JOIN."""

        if self.conclusion == "conditional_duplicate_risk" and (
            self.mechanism != "join_fanout"
            or not self.joins
            or self.condition
            not in {
                "full_join_key_uniqueness_unknown",
                "join_match_multiplicity_unknown",
            }
        ):
            raise ValueError("conditional duplicate risk requires a concrete JOIN")
        if self.conclusion == "join_mechanism_not_detected" and (
            self.mechanism != "no_join"
            or self.condition != "not_applicable"
            or self.joins
        ):
            raise ValueError("no-join conclusion cannot carry JOIN facts")
        return self

    @property
    def directed_scope(self) -> str:
        return f"{self.source_table} → {self.target_table}"


def _sql(node: exp.Expression | None) -> str:
    return node.sql(dialect=GREENPLUM_DIALECT) if node is not None else ""


def _bounded(value: object, limit: int) -> str | None:
    clean = str(value or "").strip()
    return clean if clean and len(clean) <= limit else None


def _outer_select(query: exp.Query) -> exp.Select | None:
    current: exp.Expression = query
    while isinstance(current, (exp.Subquery, exp.Paren)):
        current = current.this
    return current if isinstance(current, exp.Select) else None


def _has_unsupported_outer_shape(
    query: exp.Query,
    outer: exp.Select,
) -> bool:
    """Reject query shapes that can mask or reshape JOIN multiplicity."""

    if query.args.get("with_") is not None:
        return True
    if any(select is not outer for select in query.find_all(exp.Select)):
        return True
    if any(query.find_all(exp.Subquery)):
        return True
    if any(
        outer.args.get(name) is not None
        for name in (
            "distinct",
            "group",
            "having",
            "qualify",
            "limit",
            "offset",
        )
    ):
        return True
    return any(
        isinstance(projection, exp.AggFunc)
        or projection.find(exp.AggFunc) is not None
        for projection in outer.expressions
    )


def _top_level_conjuncts(node: exp.Expression) -> List[exp.Expression]:
    if isinstance(node, exp.And):
        return [
            *_top_level_conjuncts(node.this),
            *_top_level_conjuncts(node.expression),
        ]
    return [node]


def _constant_boolean_value(node: exp.Expression | None) -> bool | None:
    """Evaluate only boolean constants and their boolean AST composition."""

    current = node
    while isinstance(current, exp.Paren):
        current = current.this
    if isinstance(current, exp.Boolean):
        return bool(current.this)
    if isinstance(current, exp.Not):
        value = _constant_boolean_value(current.this)
        return None if value is None else not value
    if isinstance(current, exp.And):
        left = _constant_boolean_value(current.this)
        right = _constant_boolean_value(current.expression)
        if left is False or right is False:
            return False
        if left is True and right is True:
            return True
        return None
    if isinstance(current, exp.Or):
        left = _constant_boolean_value(current.this)
        right = _constant_boolean_value(current.expression)
        if left is True or right is True:
            return True
        if left is False and right is False:
            return False
    return None


def _column_equality(
    node: exp.Expression,
    *,
    joined_qualifiers: frozenset[str],
) -> str | None:
    current = node
    while isinstance(current, exp.Paren):
        current = current.this
    if not isinstance(current, exp.EQ):
        return None
    left = current.this
    right = current.expression
    while isinstance(left, exp.Paren):
        left = left.this
    while isinstance(right, exp.Paren):
        right = right.this
    if not isinstance(left, exp.Column) or not isinstance(right, exp.Column):
        return None
    left_owner = str(left.table or "").casefold()
    right_owner = str(right.table or "").casefold()
    if not left_owner or not right_owner:
        return None
    if (left_owner in joined_qualifiers) == (right_owner in joined_qualifiers):
        return None
    return _bounded(_sql(current), _MAX_JOIN_KEY_CHARS)


def _join_type(join: exp.Join) -> str | None:
    method = str(join.args.get("method") or "").upper()
    side = str(join.args.get("side") or "").upper()
    kind = str(join.args.get("kind") or "").upper()
    if (
        method not in _SUPPORTED_JOIN_METHODS
        or side not in _SUPPORTED_JOIN_SIDES
        or kind not in _SUPPORTED_JOIN_KINDS
    ):
        return None
    has_on = isinstance(join.args.get("on"), exp.Expression)
    has_using = bool(join.args.get("using") or [])
    if has_on and has_using:
        return None
    if method == "NATURAL":
        if has_on or has_using or kind not in {"", "OUTER"}:
            return None
    elif kind == "CROSS":
        if side or has_on or has_using:
            return None
    elif side:
        if kind not in {"", "OUTER"} or not (has_on or has_using):
            return None
    elif kind in {"", "INNER"}:
        if not (has_on or has_using):
            return None
    else:
        return None
    return " ".join(part for part in (method, side, kind, "JOIN") if part)


def _join_fact(join: exp.Join) -> CardinalityJoinFact | None:
    join_type = _join_type(join)
    # Derived, lateral and function relations need lineage-aware analysis.
    if join_type is None or not isinstance(join.this, exp.Table):
        return None
    relation = _bounded(_sql(join.this), _MAX_RELATION_CHARS)
    if relation is None:
        return None
    joined_qualifiers = frozenset(
        value.casefold()
        for value in (join.this.alias_or_name, join.this.name)
        if value
    )
    if not joined_qualifiers:
        return None

    predicate = ""
    equalities: List[str] = []
    on = join.args.get("on")
    if isinstance(on, exp.Expression):
        if _constant_boolean_value(on) is False:
            return None
        predicate_value = _bounded(_sql(on), _MAX_PREDICATE_CHARS)
        if predicate_value is None:
            return None
        predicate = predicate_value
        for conjunct in _top_level_conjuncts(on):
            equality = _column_equality(
                conjunct,
                joined_qualifiers=joined_qualifiers,
            )
            if equality and equality not in equalities:
                equalities.append(equality)
            if len(equalities) >= _MAX_JOIN_KEYS:
                break
    else:
        using = join.args.get("using") or []
        if using:
            rendered = [_bounded(_sql(item), _MAX_JOIN_KEY_CHARS) for item in using]
            if any(item is None for item in rendered) or len(rendered) > _MAX_JOIN_KEYS:
                return None
            values = [item for item in rendered if item is not None]
            predicate = "USING (" + ", ".join(values) + ")"
            equalities = [f"USING ({item})" for item in values]

    condition = (
        "full_join_key_uniqueness_unknown"
        if equalities
        else "join_match_multiplicity_unknown"
    )
    return CardinalityJoinFact(
        join_type=join_type,
        relation=relation,
        predicate=predicate,
        join_key_equalities=equalities,
        uniqueness_condition=condition,
    )


def analyze_cardinality_rule(rule: str) -> CardinalityRuleAnalysis:
    """Classify only JOINs attached directly to the outer SELECT."""

    original = str(rule or "").strip()
    try:
        statements = sqlglot.parse(
            quote_dollar_schemas(original),
            read=GREENPLUM_DIALECT,
        )
    except (SqlglotError, ValueError):
        return CardinalityRuleAnalysis(status="unparseable_rule")
    if len(statements) != 1 or not isinstance(statements[0], exp.Query):
        return CardinalityRuleAnalysis(status="unparseable_rule")
    query = statements[0]
    if isinstance(query, (exp.Union, exp.Intersect, exp.Except)):
        return CardinalityRuleAnalysis(status="set_operation")
    outer = _outer_select(query)
    if outer is None:
        return CardinalityRuleAnalysis(status="unparseable_rule")
    if _has_unsupported_outer_shape(query, outer):
        return CardinalityRuleAnalysis(status="unsupported_join")

    where = outer.args.get("where")
    if isinstance(where, exp.Where) and _constant_boolean_value(where.this) is False:
        return CardinalityRuleAnalysis(status="constant_false_predicate")

    raw_joins = list(outer.args.get("joins") or [])
    if len(raw_joins) > _MAX_JOINS:
        return CardinalityRuleAnalysis(status="unsupported_join")
    joins: List[CardinalityJoinFact] = []
    for raw_join in raw_joins:
        if not isinstance(raw_join, exp.Join):
            return CardinalityRuleAnalysis(status="unsupported_join")
        on = raw_join.args.get("on")
        if isinstance(on, exp.Expression) and _constant_boolean_value(on) is False:
            return CardinalityRuleAnalysis(status="constant_false_predicate")
        join = _join_fact(raw_join)
        if join is None:
            return CardinalityRuleAnalysis(status="unsupported_join")
        joins.append(join)
    return CardinalityRuleAnalysis(
        status="join_detected" if joins else "no_join",
        joins=joins,
    )


def _sql_literal(value: str) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _nonnegative_int(value: object, *, none_as_zero: bool = False) -> int | None:
    if value is None and none_as_zero:
        return 0
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        return None
    if isinstance(value, str) and not value.isascii():
        return None
    try:
        parsed = int(value)
    except (ValueError, OverflowError):
        return None
    if isinstance(value, str) and value.strip() != str(parsed):
        return None
    return parsed if parsed >= 0 else None


def _matches_pair(artifact: EvidenceArtifact, pair: ExactTablePair) -> bool:
    args = artifact.compact_args
    return bool(
        artifact.tool_name in _EXACT_DIRECTED_TOOLS
        and str(args.get("source_table") or "").casefold()
        == pair.source_table.casefold()
        and str(args.get("target_table") or "").casefold()
        == pair.target_table.casefold()
    )


def _base_fact(
    pair: ExactTablePair,
    *,
    conclusion: CardinalityConclusion,
    mechanism: CardinalityMechanism,
    condition: CardinalityCondition = "unknown",
    matching_rows: int = 0,
    joins: Sequence[CardinalityJoinFact] = (),
    evidence_ids: Sequence[str] = (),
) -> CardinalityFact:
    return CardinalityFact(
        source_table=pair.source_table,
        target_table=pair.target_table,
        conclusion=conclusion,
        mechanism=mechanism,
        condition=condition,
        matching_rows=matching_rows,
        joins=list(joins),
        evidence_ids=list(evidence_ids),
    )


def _descriptor_rules(
    pair: ExactTablePair,
    artifact: EvidenceArtifact,
    store: SavedResultStore,
) -> tuple[CardinalityMechanism | None, int, Dict[str, int]]:
    if artifact.dataset_ref is None:
        return "incomplete_evidence", 0, {}
    descriptor = store.descriptor(artifact.dataset_ref)
    if (
        descriptor is None
        or descriptor.source_tool != artifact.tool_name
        or descriptor.truncated
        or descriptor.source_total is None
        or descriptor.source_total != descriptor.row_count
        or not _REQUIRED_COLUMNS.issubset(
            {column.name for column in descriptor.columns}
        )
    ):
        return "incomplete_evidence", 0, {}

    source = _sql_literal(pair.source_table)
    target = _sql_literal(pair.target_table)
    aggregate = store.query(
        result_ref=descriptor.result_ref,
        query=(
            "SELECT COUNT(*) AS saved_rows, "
            "SUM(CASE WHEN LOWER(TRIM(source_table)) = "
            f"LOWER(TRIM({source})) AND LOWER(TRIM(target_table)) = "
            f"LOWER(TRIM({target})) THEN 1 ELSE 0 END) AS exact_rows, "
            "SUM(CASE WHEN NULLIF(TRIM(transformation_rule), '') IS NULL "
            "OR TRIM(transformation_rule) = '-' THEN 1 ELSE 0 END) "
            "AS missing_rule_rows FROM result"
        ),
        preview_limit=1,
    )
    rows = aggregate.get("rows") or []
    if (
        aggregate.get("error")
        or aggregate.get("input_truncated")
        or aggregate.get("truncated")
        or len(rows) != 1
        or not isinstance(rows[0], dict)
    ):
        return "incomplete_evidence", 0, {}
    saved_rows = _nonnegative_int(rows[0].get("saved_rows"))
    exact_rows = _nonnegative_int(
        rows[0].get("exact_rows"),
        none_as_zero=True,
    )
    missing_rows = _nonnegative_int(
        rows[0].get("missing_rule_rows"),
        none_as_zero=True,
    )
    if saved_rows is None or exact_rows is None or missing_rows is None:
        return "incomplete_evidence", 0, {}
    if saved_rows != descriptor.row_count:
        return "incomplete_evidence", 0, {}
    if exact_rows != saved_rows:
        return "scope_mismatch", saved_rows, {}
    if missing_rows:
        return "missing_transformation_rule", saved_rows, {}
    if saved_rows == 0:
        return None, 0, {}

    grouped = store.query(
        result_ref=descriptor.result_ref,
        query=(
            "SELECT transformation_rule, COUNT(*) AS occurrence_count "
            "FROM result GROUP BY transformation_rule "
            "ORDER BY transformation_rule"
        ),
        preview_limit=_MAX_DISTINCT_RULES,
    )
    grouped_rows = grouped.get("rows") or []
    if (
        grouped.get("error")
        or grouped.get("input_truncated")
        or grouped.get("truncated")
        or not all(isinstance(row, dict) for row in grouped_rows)
    ):
        return "incomplete_evidence", saved_rows, {}
    rules: Dict[str, int] = {}
    for row in grouped_rows:
        occurrence_count = _nonnegative_int(row.get("occurrence_count"))
        rule = str(row.get("transformation_rule") or "").strip()
        if occurrence_count is None or not rule:
            return "incomplete_evidence", saved_rows, {}
        rules[rule] = occurrence_count
    if not rules or sum(rules.values()) != saved_rows:
        return "incomplete_evidence", saved_rows, {}
    return None, saved_rows, rules


def _analysis_signature(analysis: CardinalityRuleAnalysis) -> tuple[object, ...]:
    return (
        analysis.status,
        tuple(
            (
                join.join_type,
                join.relation,
                join.predicate,
                tuple(join.join_key_equalities),
                join.uniqueness_condition,
            )
            for join in analysis.joins
        ),
    )


def _fact_for_pair(
    pair: ExactTablePair,
    artifacts: Sequence[EvidenceArtifact],
    store: SavedResultStore,
) -> CardinalityFact:
    matching = [artifact for artifact in artifacts if _matches_pair(artifact, pair)]
    evidence_ids = [
        str(value)
        for value in list(
            dict.fromkeys(artifact.evidence_id for artifact in matching)
        )[:_MAX_EVIDENCE_IDS]
    ]
    if not matching:
        return _base_fact(
            pair,
            conclusion="not_assessed",
            mechanism="incomplete_evidence",
        )

    snapshots: List[tuple[int, Dict[str, int]]] = []
    seen_refs: set[str] = set()
    for artifact in matching:
        dataset_ref = str(artifact.dataset_ref or "")
        if dataset_ref in seen_refs:
            continue
        seen_refs.add(dataset_ref)
        error, row_count, rules = _descriptor_rules(pair, artifact, store)
        if error is not None:
            return _base_fact(
                pair,
                conclusion="not_assessed",
                mechanism=error,
                matching_rows=row_count,
                evidence_ids=evidence_ids,
            )
        snapshots.append((row_count, rules))

    if not snapshots:
        return _base_fact(
            pair,
            conclusion="not_assessed",
            mechanism="incomplete_evidence",
            evidence_ids=evidence_ids,
        )
    first_rows, first_rules = snapshots[0]
    if any(snapshot != snapshots[0] for snapshot in snapshots[1:]):
        return _base_fact(
            pair,
            conclusion="conflicting",
            mechanism="conflicting_rules",
            matching_rows=first_rows,
            evidence_ids=evidence_ids,
        )
    if first_rows == 0:
        return _base_fact(
            pair,
            conclusion="not_assessed",
            mechanism="no_exact_mapping",
            evidence_ids=evidence_ids,
        )

    analyses = [analyze_cardinality_rule(rule) for rule in first_rules]
    unknown_statuses = {
        analysis.status
        for analysis in analyses
        if analysis.status
        in {
            "constant_false_predicate",
            "unparseable_rule",
            "set_operation",
            "unsupported_join",
        }
    }
    if unknown_statuses:
        mechanism: CardinalityMechanism = next(
            item
            for item in (
                "constant_false_predicate",
                "set_operation",
                "unsupported_join",
                "unparseable_rule",
            )
            if item in unknown_statuses
        )
        return _base_fact(
            pair,
            conclusion="not_assessed",
            mechanism=mechanism,
            matching_rows=first_rows,
            evidence_ids=evidence_ids,
        )

    signatures = {_analysis_signature(analysis) for analysis in analyses}
    if len(signatures) != 1:
        return _base_fact(
            pair,
            conclusion="conflicting",
            mechanism="conflicting_rules",
            matching_rows=first_rows,
            evidence_ids=evidence_ids,
        )
    analysis = analyses[0]
    if analysis.status == "no_join":
        return _base_fact(
            pair,
            conclusion="join_mechanism_not_detected",
            mechanism="no_join",
            condition="not_applicable",
            matching_rows=first_rows,
            evidence_ids=evidence_ids,
        )
    condition: CardinalityCondition = (
        "full_join_key_uniqueness_unknown"
        if all(
            join.uniqueness_condition
            == "full_join_key_uniqueness_unknown"
            for join in analysis.joins
        )
        else "join_match_multiplicity_unknown"
    )
    return _base_fact(
        pair,
        conclusion="conditional_duplicate_risk",
        mechanism="join_fanout",
        condition=condition,
        matching_rows=first_rows,
        joins=analysis.joins,
        evidence_ids=evidence_ids,
    )


def derive_cardinality_facts(
    contract: SqlRiskScopeContract,
    artifacts: Sequence[EvidenceArtifact],
    store: SavedResultStore,
) -> List[CardinalityFact]:
    """Derive bounded facts from complete accepted exact mapping relations."""

    pair = cardinality_pair_from_contract(contract)
    if (
        pair is None
        or len(pair.source_table) > _MAX_IDENTIFIER_CHARS
        or len(pair.target_table) > _MAX_IDENTIFIER_CHARS
    ):
        return []
    return [_fact_for_pair(pair, artifacts, store)]


def cardinality_payload(
    facts: Sequence[CardinalityFact],
) -> Dict[str, object]:
    """Return a bounded payload without raw mapping rows or full SQL."""

    payload: Dict[str, object] = {
        "authority": "deterministic_sqlglot_full_saved_result",
        "scope_rule": (
            "JOIN is a potential row-multiplication mechanism only. The "
            "saved S2T mapping does not establish full join-key uniqueness "
            "or prove that duplicate output rows actually occur."
        ),
        "facts": [
            fact.model_dump(mode="json")
            for fact in list(facts)[:_MAX_FACTS]
        ],
    }
    if len(json.dumps(payload, ensure_ascii=False)) > MAX_CARDINALITY_PAYLOAD_CHARS:
        compact_facts: List[Dict[str, object]] = []
        for fact in list(facts)[:2]:
            item = fact.model_dump(mode="json")
            compact_joins = []
            for join in item["joins"][:1]:
                compact_join = dict(join)
                compact_join["relation"] = compact_join["relation"][:120]
                compact_join["predicate"] = compact_join["predicate"][:240]
                compact_join["join_key_equalities"] = [
                    value[:120]
                    for value in compact_join["join_key_equalities"][:2]
                ]
                compact_joins.append(compact_join)
            item["joins"] = compact_joins
            item["evidence_ids"] = item["evidence_ids"][:2]
            compact_facts.append(item)
        payload["facts"] = compact_facts
        payload["payload_truncated"] = True
    if len(json.dumps(payload, ensure_ascii=False)) > MAX_CARDINALITY_PAYLOAD_CHARS:
        compact = []
        for fact in list(facts)[:1]:
            item = fact.model_dump(mode="json")
            item["joins"] = []
            item["evidence_ids"] = []
            compact.append(item)
        payload["facts"] = compact
        payload["payload_truncated"] = True
    if len(json.dumps(payload, ensure_ascii=False)) > MAX_CARDINALITY_PAYLOAD_CHARS:
        payload = {
            "authority": "deterministic_sqlglot_full_saved_result",
            "scope_rule": "Payload omitted because its bounded representation is too large.",
            "facts": [],
            "payload_truncated": True,
        }
    return payload


def _render_join(join: CardinalityJoinFact) -> str:
    clause = f"{join.join_type} {join.relation}"
    if join.predicate:
        clause += (
            f" {join.predicate}"
            if join.predicate.upper().startswith("USING (")
            else f" ON {join.predicate}"
        )
    return clause


def render_cardinality_answer(facts: Sequence[CardinalityFact]) -> str:
    """Render the structured fact without a benchmark-specific template."""

    blocks: List[str] = []
    for fact in list(facts)[:_MAX_FACTS]:
        scope = f"`{fact.directed_scope}`"
        if fact.conclusion == "conditional_duplicate_risk":
            clauses = "; ".join(f"`{_render_join(join)}`" for join in fact.joins)
            keys = list(
                dict.fromkeys(
                    key for join in fact.joins for key in join.join_key_equalities
                )
            )
            if fact.condition == "full_join_key_uniqueness_unknown" and keys:
                condition = (
                    "Уникальность полного JOIN-ключа ("
                    + ", ".join(f"`{key}`" for key in keys)
                    + ") не установлена: в прочитанном mapping нет "
                    "подтверждающих данных"
                )
            else:
                condition = (
                    "Уникальность и кратность совпадений JOIN не установлены: "
                    "в прочитанном mapping нет подтверждающих данных"
                )
            blocks.append(
                f"Для {scope} сохранённое правило содержит {clauses}. "
                "Такой JOIN может размножить строки, "
                "если одному входному ряду соответствует несколько строк "
                f"другой стороны. {condition}. Поэтому возможный fan-out "
                "условен, а наличие фактических дубликатов не установлено."
            )
        elif fact.conclusion == "join_mechanism_not_detected":
            blocks.append(
                f"Для {scope} во внешнем SELECT соединение не обнаружено. "
                "Поэтому JOIN-based fan-out этим mapping не установлен, а "
                "проверка уникальности здесь не применима. Это не доказывает "
                "отсутствие дубликатов по другим причинам."
            )
        elif fact.conclusion == "conflicting":
            blocks.append(
                f"Для {scope} нельзя определить один JOIN-механизм: полные "
                "сохранённые S2T-правила расходятся по структуре соединений. "
                "Уникальность ключей и риск изменения кардинальности поэтому "
                "не оценены."
            )
        else:
            blocks.append(
                f"Для {scope} evidence не позволяет определить JOIN-механизм "
                f"(`{fact.mechanism}`). Уникальность ключей и риск изменения "
                "кардинальности не оценены; неполные либо неподдерживаемые "
                "данные не подтверждают и отсутствие дубликатов."
            )
    return "\n\n".join(blocks)


__all__ = [
    "CardinalityConclusion",
    "CardinalityCondition",
    "CardinalityFact",
    "CardinalityJoinFact",
    "CardinalityMechanism",
    "CardinalityRuleAnalysis",
    "MAX_CARDINALITY_PAYLOAD_CHARS",
    "analyze_cardinality_rule",
    "cardinality_payload",
    "derive_cardinality_facts",
    "render_cardinality_answer",
]
