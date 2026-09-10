"""Deterministic field-scoped value-change facts for exact S2T evidence."""

from __future__ import annotations

import json
import re
from typing import Dict, List, Literal, Sequence

from pydantic import BaseModel, ConfigDict, Field

from .contracts import EvidenceArtifact
from .tools.saved_results import SavedResultStore
from .transformation_ast import (
    FieldValueChangeAnalysis,
    analyze_field_value_change,
    normalize_transformation,
)


FieldValueChangeConclusion = Literal[
    "not_detected",
    "may_change",
    "not_assessed",
    "conflicting",
]
FieldValueChangeMechanism = Literal[
    "direct_column",
    "value_expression",
    "different_source_column",
    "direct_mapping_rule",
    "unparseable_rule",
    "missing_target_projection",
    "unresolved_source_provenance",
    "ambiguous_target_projection",
    "set_operation",
    "no_exact_mapping",
    "incomplete_evidence",
    "conflicting_rules",
]

_ENDPOINT_RE = re.compile(
    r"[`\"']?(?P<source>[A-Za-zА-Яа-яЁё0-9_$.-]+)[`\"']?\s*"
    r"(?:→|->|=>)\s*"
    r"[`\"']?(?P<target>[A-Za-zА-Яа-яЁё0-9_$.-]+)[`\"']?"
)
_EXACT_DIRECTED_TOOLS = frozenset(
    {"read_s2t_mapping", "read_s2t_source_to_target"}
)
_DIRECT_RULE_MARKERS = frozenset({"", "-"})
_MAX_FIELD_PAIRS = 8
_MAX_EXPRESSIONS = 4
_MAX_EXPRESSION_CHARS = 300
_MAX_EVIDENCE_IDS = 8
_MAX_TABLE_CHARS = 200
_MAX_FIELD_CHARS = 128
_MAX_DISTINCT_RULES = 100
MAX_VALUE_CHANGE_PAYLOAD_CHARS = 12_000


class ExactFieldPair(BaseModel):
    """One literal directed field pair found in the original task."""

    model_config = ConfigDict(extra="forbid")

    source_table: str = Field(max_length=_MAX_TABLE_CHARS)
    source_field: str = Field(max_length=_MAX_FIELD_CHARS)
    target_table: str = Field(max_length=_MAX_TABLE_CHARS)
    target_field: str = Field(max_length=_MAX_FIELD_CHARS)

    @property
    def source_reference(self) -> str:
        return f"{self.source_table}.{self.source_field}"

    @property
    def target_reference(self) -> str:
        return f"{self.target_table}.{self.target_field}"


class FieldValueChangeFact(BaseModel):
    """Authoritative code-derived conclusion for one exact field pair."""

    model_config = ConfigDict(extra="forbid")

    source_table: str = Field(max_length=_MAX_TABLE_CHARS)
    source_field: str = Field(max_length=_MAX_FIELD_CHARS)
    target_table: str = Field(max_length=_MAX_TABLE_CHARS)
    target_field: str = Field(max_length=_MAX_FIELD_CHARS)
    conclusion: FieldValueChangeConclusion
    mechanism: FieldValueChangeMechanism
    matching_rows: int = Field(ge=0)
    target_expressions: List[str] = Field(default_factory=list)
    evidence_ids: List[str] = Field(default_factory=list)

    @property
    def source_reference(self) -> str:
        return f"{self.source_table}.{self.source_field}"

    @property
    def target_reference(self) -> str:
        return f"{self.target_table}.{self.target_field}"


def extract_exact_field_pairs(task: str) -> List[ExactFieldPair]:
    """Extract every literal ``table.field → table.field`` pair, bounded."""

    pairs: List[ExactFieldPair] = []
    seen: set[tuple[str, str, str, str]] = set()
    for match in _ENDPOINT_RE.finditer(str(task or "")):
        source = match.group("source").strip(".`\"'")
        target = match.group("target").strip(".`\"'")
        if "." not in source or "." not in target:
            continue
        source_table, source_field = source.rsplit(".", 1)
        target_table, target_field = target.rsplit(".", 1)
        values = (source_table, source_field, target_table, target_field)
        if not all(values):
            continue
        if (
            len(source_table) > _MAX_TABLE_CHARS
            or len(target_table) > _MAX_TABLE_CHARS
            or len(source_field) > _MAX_FIELD_CHARS
            or len(target_field) > _MAX_FIELD_CHARS
        ):
            continue
        identity = tuple(value.casefold() for value in values)
        if identity in seen:
            continue
        seen.add(identity)
        pairs.append(
            ExactFieldPair(
                source_table=source_table,
                source_field=source_field,
                target_table=target_table,
                target_field=target_field,
            )
        )
        if len(pairs) >= _MAX_FIELD_PAIRS:
            break
    return pairs


def _sql_literal(value: str) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _artifact_matches_pair(
    artifact: EvidenceArtifact,
    pair: ExactFieldPair,
) -> bool:
    if artifact.tool_name not in _EXACT_DIRECTED_TOOLS:
        return False
    args = artifact.compact_args
    return (
        str(args.get("source_table") or "").casefold()
        == pair.source_table.casefold()
        and str(args.get("target_table") or "").casefold()
        == pair.target_table.casefold()
    )


def _analysis_mechanism(
    analysis: FieldValueChangeAnalysis,
) -> FieldValueChangeMechanism:
    return {
        "direct_projection": "direct_column",
        "value_expression": "value_expression",
        "different_source_column": "different_source_column",
        "source_provenance_unknown": "unresolved_source_provenance",
        "target_projection_missing": "missing_target_projection",
        "ambiguous_target_projection": "ambiguous_target_projection",
        "set_operation_unsupported": "set_operation",
        "parse_error": "unparseable_rule",
    }[analysis.status]


def _bounded_expression(value: str) -> str:
    clean = str(value or "").strip()
    if len(clean) <= _MAX_EXPRESSION_CHARS:
        return clean
    return clean[: _MAX_EXPRESSION_CHARS - 1].rstrip() + "…"


def _fact_for_pair(
    pair: ExactFieldPair,
    artifacts: Sequence[EvidenceArtifact],
    store: SavedResultStore,
) -> FieldValueChangeFact:
    matching_artifacts = [
        artifact
        for artifact in artifacts
        if _artifact_matches_pair(artifact, pair)
    ]
    evidence_ids = [
        str(value)[:120]
        for value in list(
            dict.fromkeys(
                artifact.evidence_id for artifact in matching_artifacts
            )
        )[:_MAX_EVIDENCE_IDS]
    ]
    if not matching_artifacts:
        return FieldValueChangeFact(
            **pair.model_dump(),
            conclusion="not_assessed",
            mechanism="incomplete_evidence",
            matching_rows=0,
        )
    dataset_refs: List[str] = []
    invalid_dataset_ref = False
    for artifact in matching_artifacts:
        if artifact.dataset_ref is None:
            invalid_dataset_ref = True
            continue
        descriptor = store.descriptor(artifact.dataset_ref)
        if (
            descriptor is None
            or descriptor.source_tool != artifact.tool_name
            or descriptor.truncated
        ):
            invalid_dataset_ref = True
            continue
        if artifact.dataset_ref not in dataset_refs:
            dataset_refs.append(artifact.dataset_ref)
    if invalid_dataset_ref or not dataset_refs:
        return FieldValueChangeFact(
            **pair.model_dump(),
            conclusion="not_assessed",
            mechanism="incomplete_evidence",
            matching_rows=0,
            evidence_ids=evidence_ids,
        )

    predicates = " AND ".join(
        (
            f"LOWER(source_table) = LOWER({_sql_literal(pair.source_table)})",
            f"LOWER(source_field) = LOWER({_sql_literal(pair.source_field)})",
            f"LOWER(target_table) = LOWER({_sql_literal(pair.target_table)})",
            f"LOWER(target_field) = LOWER({_sql_literal(pair.target_field)})",
        )
    )
    grouped_rules: Dict[object, int] = {}
    matching_row_count = 0
    incomplete = False
    for dataset_ref in dataset_refs:
        count_result = store.query(
            result_ref=dataset_ref,
            query=(
                "SELECT COUNT(*) AS matching_rows FROM result WHERE "
                + predicates
            ),
            preview_limit=1,
        )
        count_rows = count_result.get("rows", [])
        if (
            count_result.get("error")
            or count_result.get("input_truncated")
            or not count_rows
            or not isinstance(count_rows[0], dict)
        ):
            incomplete = True
            continue
        matching_row_count += int(count_rows[0].get("matching_rows") or 0)

        rules_result = store.query(
            result_ref=dataset_ref,
            query=(
                "SELECT transformation_rule, COUNT(*) AS occurrence_count "
                "FROM result WHERE "
                + predicates
                + " GROUP BY transformation_rule ORDER BY transformation_rule"
            ),
            preview_limit=_MAX_DISTINCT_RULES,
        )
        if (
            rules_result.get("error")
            or rules_result.get("input_truncated")
            or rules_result.get("truncated")
        ):
            incomplete = True
            continue
        for row in rules_result.get("rows", []):
            if not isinstance(row, dict):
                incomplete = True
                continue
            rule = row.get("transformation_rule")
            grouped_rules[rule] = grouped_rules.get(rule, 0) + int(
                row.get("occurrence_count") or 0
            )
        incomplete = incomplete or (
            len(grouped_rules) > _MAX_DISTINCT_RULES
        )
    if incomplete:
        return FieldValueChangeFact(
            **pair.model_dump(),
            conclusion="not_assessed",
            mechanism="incomplete_evidence",
            matching_rows=matching_row_count,
            evidence_ids=evidence_ids,
        )
    if matching_row_count == 0:
        return FieldValueChangeFact(
            **pair.model_dump(),
            conclusion="not_assessed",
            mechanism="no_exact_mapping",
            matching_rows=0,
            evidence_ids=evidence_ids,
        )

    analyses: List[tuple[FieldValueChangeAnalysis, FieldValueChangeMechanism]] = []
    cache: Dict[str, FieldValueChangeAnalysis] = {}
    direct_rule_marker = False
    for raw_rule in grouped_rules:
        rule = str(raw_rule or "").strip()
        if rule in _DIRECT_RULE_MARKERS:
            direct_rule_marker = True
            continue
        analysis = cache.get(rule)
        if analysis is None:
            analysis = analyze_field_value_change(
                normalize_transformation(rule),
                source_table=pair.source_table,
                source_field=pair.source_field,
                target_field=pair.target_field,
            )
            cache[rule] = analysis
        analyses.append((analysis, _analysis_mechanism(analysis)))

    expressions = list(
        dict.fromkeys(
            _bounded_expression(analysis.target_expression)
            for analysis, _ in analyses
            if analysis.target_expression
        )
    )[:_MAX_EXPRESSIONS]
    statuses = {analysis.status for analysis, _ in analyses}
    known = {
        (
            "not_detected"
            if status == "direct_projection"
            else "may_change"
        )
        for status in statuses
        if status
        in {"direct_projection", "value_expression", "different_source_column"}
    }
    if direct_rule_marker:
        known.add("not_detected")
    unknown = any(
        status
        in {
            "ambiguous_target_projection",
            "parse_error",
            "set_operation_unsupported",
            "source_provenance_unknown",
            "target_projection_missing",
        }
        for status in statuses
    )
    if len(known) > 1:
        conclusion: FieldValueChangeConclusion = "conflicting"
        mechanism: FieldValueChangeMechanism = "conflicting_rules"
    elif unknown or (not analyses and not direct_rule_marker):
        conclusion = "not_assessed"
        mechanisms = {item for _, item in analyses}
        mechanism = (
            next(iter(mechanisms))
            if len(mechanisms) == 1
            else "unparseable_rule"
        )
    elif known == {"not_detected"}:
        conclusion = "not_detected"
        mechanism = (
            "direct_mapping_rule"
            if direct_rule_marker
            else "direct_column"
        )
    else:
        conclusion = "may_change"
        mechanisms = {item for _, item in analyses}
        mechanism = (
            next(iter(mechanisms))
            if len(mechanisms) == 1
            else "value_expression"
        )

    return FieldValueChangeFact(
        **pair.model_dump(),
        conclusion=conclusion,
        mechanism=mechanism,
        matching_rows=matching_row_count,
        target_expressions=expressions,
        evidence_ids=evidence_ids,
    )


def derive_field_value_change_facts(
    task: str,
    artifacts: Sequence[EvidenceArtifact],
    store: SavedResultStore,
) -> List[FieldValueChangeFact]:
    """Derive bounded authoritative facts from full accepted saved results."""

    return [
        _fact_for_pair(pair, artifacts, store)
        for pair in extract_exact_field_pairs(task)
    ]


def field_value_change_payload(
    facts: Sequence[FieldValueChangeFact],
) -> Dict[str, object]:
    """Return a bounded model/metrics payload without copying raw S2T rows."""

    payload: Dict[str, object] = {
        "authority": "deterministic_sqlglot_full_saved_result",
        "scope_rule": (
            "Each conclusion applies only to its exact target projection; "
            "expressions in neighbouring output aliases are excluded."
        ),
        "facts": [fact.model_dump(mode="json") for fact in facts],
    }
    if (
        len(json.dumps(payload, ensure_ascii=False))
        > MAX_VALUE_CHANGE_PAYLOAD_CHARS
    ):
        compact_facts = []
        for fact in facts:
            item = fact.model_dump(mode="json")
            item["target_expressions"] = [
                _bounded_expression(value)[:120]
                for value in item["target_expressions"][:1]
            ]
            item["evidence_ids"] = item["evidence_ids"][:4]
            compact_facts.append(item)
        payload["facts"] = compact_facts
        payload["payload_truncated"] = True
    if (
        len(json.dumps(payload, ensure_ascii=False))
        > MAX_VALUE_CHANGE_PAYLOAD_CHARS
    ):
        raw_facts = payload.get("facts")
        if isinstance(raw_facts, list):
            for item in raw_facts:
                if isinstance(item, dict):
                    item["target_expressions"] = []
                    item["evidence_ids"] = item["evidence_ids"][:1]
    return payload


def render_field_value_change_answer(
    facts: Sequence[FieldValueChangeFact],
) -> str:
    """Render the final single-aspect verdict without another LLM call."""

    lines: List[str] = []
    for fact in facts:
        pair = f"`{fact.source_reference} → {fact.target_reference}`"
        if fact.conclusion == "not_detected":
            lines.append(
                f"Для {pair} механизм изменения значения не обнаружен: "
                "target field получает прямую колонку source field. "
                "Выражения соседних output aliases к этой паре не относятся."
            )
        elif fact.conclusion == "may_change":
            lines.append(
                f"Для {pair} значение может измениться: обнаружен механизм "
                f"`{fact.mechanism}` в точной target-проекции."
            )
        elif fact.conclusion == "conflicting":
            lines.append(
                f"Для {pair} сохранённые S2T-правила противоречат друг другу; "
                "однозначный вывод об изменении значения невозможен."
            )
        else:
            lines.append(
                f"Для {pair} изменение значения не оценено: "
                f"`{fact.mechanism}`."
            )
    return "\n".join(lines)


__all__ = [
    "ExactFieldPair",
    "FieldValueChangeConclusion",
    "FieldValueChangeFact",
    "FieldValueChangeMechanism",
    "MAX_VALUE_CHANGE_PAYLOAD_CHARS",
    "derive_field_value_change_facts",
    "extract_exact_field_pairs",
    "field_value_change_payload",
    "render_field_value_change_answer",
]
