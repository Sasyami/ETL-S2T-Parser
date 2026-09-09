"""Deterministic write-semantics facts for exact directed S2T evidence.

This module intentionally answers a narrower question than transformation
analysis: does the *saved mapping itself* contain an explicit write statement?
It never infers append, overwrite or conflict handling from keys, catalog
metadata, a clipped preview, or write-looking words nested in a SELECT.
"""

from __future__ import annotations

import json
import re
from typing import Dict, List, Literal, Sequence

import sqlglot
from pydantic import BaseModel, ConfigDict, Field
from sqlglot import exp
from sqlglot.errors import SqlglotError

from services.sql_dialects import GREENPLUM_DIALECT

from .contracts import EvidenceArtifact
from .tools.saved_results import SavedResultStore
from .transformation_ast import quote_dollar_schemas


WriteSemanticsConclusion = Literal[
    "not_assessed",
    "explicit_write",
    "conflicting",
    "incomplete",
]
WriteSemanticsMechanism = Literal[
    "write_statement_absent",
    "insert",
    "insert_on_conflict",
    "insert_overwrite",
    "update",
    "merge",
    "delete",
    "ctas",
    "multiple",
    "unparseable_rule",
    "unsupported_statement",
    "multiple_statements",
    "no_exact_mapping",
    "incomplete_evidence",
]

_EXPLICIT_WRITE_MECHANISMS = frozenset(
    {
        "insert",
        "insert_on_conflict",
        "insert_overwrite",
        "update",
        "merge",
        "delete",
        "ctas",
    }
)
_DIRECTED_TOOLS = frozenset(
    {"read_s2t_mapping", "read_s2t_source_to_target"}
)
_REQUIRED_DATASET_COLUMNS = frozenset(
    {"source_table", "target_table", "transformation_rule"}
)
_EMPTY_RULE_MARKERS = frozenset({"", "-"})
_MAX_TABLE_PAIRS = 8
_MAX_TABLE_CHARS = 240
_MAX_DISTINCT_RULES = 100
_MAX_EVIDENCE_IDS = 8
_MAX_DETECTED_MECHANISMS = 8
MAX_WRITE_SEMANTICS_PAYLOAD_CHARS = 8_000

_BARE_IDENTIFIER_PART = (
    r"[A-Za-z_$\u0410-\u042f\u0430-\u044f\u0401\u0451]"
    r"[A-Za-z0-9_$\u0410-\u042f\u0430-\u044f\u0401\u0451-]*"
)
_QUOTED_IDENTIFIER_PART = r"(?:`[^`\r\n.]+`|\"[^\"\r\n.]+\")"
_IDENTIFIER_PART = rf"(?:{_BARE_IDENTIFIER_PART}|{_QUOTED_IDENTIFIER_PART})"
_QUOTED_ENDPOINT = r"(?:`[^`\r\n]+`|\"[^\"\r\n]+\")"
_TABLE_ENDPOINT = (
    rf"(?:{_QUOTED_ENDPOINT}|{_IDENTIFIER_PART}"
    rf"(?:\s*\.\s*{_IDENTIFIER_PART}){{0,3}})"
)
_TABLE_PAIR_RE = re.compile(
    rf"(?<![\w$`\"])(?P<source>{_TABLE_ENDPOINT})\s*"
    rf"(?:→|->|=>)\s*(?P<target>{_TABLE_ENDPOINT})(?![\w$`\"])",
    re.IGNORECASE,
)

_WRITE_SEMANTICS_PHRASE = (
    r"(?:write[ _-]*semantics|"
    r"семантик\w*\s+(?:запис|загруз)\w*|"
    r"режим\w*\s+(?:запис|загруз)\w*)"
)
_SAME_CLAUSE_GAP = r"[^.!?\r\n]{0,96}"
_EXCLUSIVE_WRITE_SEMANTICS_RE = re.compile(
    rf"(?:\b(?:только|only)\b{_SAME_CLAUSE_GAP}{_WRITE_SEMANTICS_PHRASE}"
    rf"|{_WRITE_SEMANTICS_PHRASE}{_SAME_CLAUSE_GAP}\b(?:только|only)\b)",
    re.IGNORECASE,
)
_ADDITIONAL_OR_PRESENTATION_INTENT_RE = re.compile(
    r"\b(?:также|покажи|выведи|список|объясни|сравни|"
    r"построй|"
    r"also|show|list|display|explain|compare)\b"
    r"|\bа\s+ещ[её]\b"
    r"|\bполн\w*\s+результат\w*\b",
    re.IGNORECASE,
)
_OTHER_SQL_RISK_ASPECT_RE = re.compile(
    r"\b(?:row[ _-]*filtering|cardinality|constraint[ _-]*rejection|"
    r"value[ _-]*changes?)\b"
    r"|изменен\w*\s+значен\w*"
    r"|кардинал\w*"
    r"|фильтр\w*\s+строк\w*",
    re.IGNORECASE,
)


class ExactTablePair(BaseModel):
    """One literal directed table pair found in the original task."""

    model_config = ConfigDict(extra="forbid")

    source_table: str = Field(max_length=_MAX_TABLE_CHARS)
    target_table: str = Field(max_length=_MAX_TABLE_CHARS)


class WriteStatementClassification(BaseModel):
    """Root-statement classification for one distinct transformation rule."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["absent", "explicit", "conflicting", "unknown"]
    mechanism: WriteSemanticsMechanism
    detected_mechanisms: List[WriteSemanticsMechanism] = Field(
        default_factory=list
    )


class WriteSemanticsFact(BaseModel):
    """Bounded code-derived write-semantics conclusion for one table pair."""

    model_config = ConfigDict(extra="forbid")

    source_table: str = Field(max_length=_MAX_TABLE_CHARS)
    target_table: str = Field(max_length=_MAX_TABLE_CHARS)
    conclusion: WriteSemanticsConclusion
    mechanism: WriteSemanticsMechanism
    matching_rows: int = Field(ge=0)
    detected_mechanisms: List[WriteSemanticsMechanism] = Field(
        default_factory=list
    )
    evidence_ids: List[str] = Field(default_factory=list)


def _clean_endpoint(value: str) -> str:
    clean = str(value or "").strip()
    if (
        len(clean) >= 2
        and clean[0] in {'`', '"'}
        and clean[-1] == clean[0]
    ):
        clean = clean[1:-1].strip()
    parts = [
        part.strip().strip('`"').strip()
        for part in re.split(r"\s*\.\s*", clean)
    ]
    return ".".join(part for part in parts if part)


def extract_exact_table_pairs(task: str) -> List[ExactTablePair]:
    """Extract bounded literal ``source_table → target_table`` pairs."""

    pairs: List[ExactTablePair] = []
    seen: set[tuple[str, str]] = set()
    for match in _TABLE_PAIR_RE.finditer(str(task or "")):
        source_table = _clean_endpoint(match.group("source"))
        target_table = _clean_endpoint(match.group("target"))
        if (
            not source_table
            or not target_table
            or len(source_table) > _MAX_TABLE_CHARS
            or len(target_table) > _MAX_TABLE_CHARS
        ):
            continue
        identity = (source_table.casefold(), target_table.casefold())
        if identity in seen:
            continue
        seen.add(identity)
        pairs.append(
            ExactTablePair(
                source_table=source_table,
                target_table=target_table,
            )
        )
        if len(pairs) >= _MAX_TABLE_PAIRS:
            break
    return pairs


def is_exclusive_write_semantics_request(task: str) -> bool:
    """Return whether code may answer only the requested write-semantics facet.

    The predicate deliberately requires a literal directed table pair and an
    explicit ``только``/``only`` qualifier close to the aspect name.  A
    presentation request or another SQL-risk facet disables this narrow path.
    """

    if not isinstance(task, str) or not task.strip():
        return False
    if _ADDITIONAL_OR_PRESENTATION_INTENT_RE.search(task):
        return False
    if _OTHER_SQL_RISK_ASPECT_RE.search(task):
        return False
    return bool(
        extract_exact_table_pairs(task)
        and _EXCLUSIVE_WRITE_SEMANTICS_RE.search(task)
    )


def _node_write_mechanism(
    statement: exp.Expression,
) -> WriteSemanticsMechanism | None:
    """Return an explicit write mechanism for one AST node, if any."""

    if isinstance(statement, exp.Insert):
        if bool(statement.args.get("overwrite")):
            return "insert_overwrite"
        if statement.args.get("conflict") is not None:
            return "insert_on_conflict"
        return "insert"
    if isinstance(statement, exp.Update):
        return "update"
    if isinstance(statement, exp.Merge):
        return "merge"
    if isinstance(statement, exp.Delete):
        return "delete"
    if (
        isinstance(statement, exp.Create)
        and str(statement.args.get("kind") or "").casefold() == "table"
        and isinstance(statement.args.get("expression"), exp.Query)
    ):
        return "ctas"
    if (
        isinstance(statement, exp.Query)
        and statement.args.get("into") is not None
    ):
        return "ctas"
    return None


def _has_write_ancestor(
    node: exp.Expression,
    root: exp.Expression,
) -> bool:
    """Avoid counting MERGE action nodes as separate statements."""

    current = node.parent
    while current is not None and current is not root:
        if _node_write_mechanism(current) is not None:
            return True
        current = current.parent
    return False


def classify_write_semantics_rule(rule: object) -> WriteStatementClassification:
    """Classify parsed write statements, never SQL-looking substrings.

    A plain SELECT/WITH has no write semantics.  Data-modifying CTEs are
    detected structurally, while DML nodes nested inside a MERGE action are
    not double-counted as independent write modes.
    """

    if rule is not None and not isinstance(rule, str):
        return WriteStatementClassification(
            status="unknown",
            mechanism="unparseable_rule",
        )
    text = str(rule or "").strip()
    if text in _EMPTY_RULE_MARKERS:
        return WriteStatementClassification(
            status="absent",
            mechanism="write_statement_absent",
        )
    try:
        statements = sqlglot.parse(
            quote_dollar_schemas(text),
            read=GREENPLUM_DIALECT,
        )
    except (SqlglotError, TypeError, ValueError):
        return WriteStatementClassification(
            status="unknown",
            mechanism="unparseable_rule",
        )
    if len(statements) != 1 or statements[0] is None:
        return WriteStatementClassification(
            status="unknown",
            mechanism="multiple_statements",
        )

    statement = statements[0]
    root_mechanism = _node_write_mechanism(statement)
    if root_mechanism is not None and not isinstance(statement, exp.Query):
        # MERGE contains Update/Insert AST action nodes.  Its root is the
        # authoritative statement kind, not a bundle of three write modes.
        return WriteStatementClassification(
            status="explicit",
            mechanism=root_mechanism,
            detected_mechanisms=[root_mechanism],
        )

    if isinstance(statement, exp.Query):
        mechanisms = {
            mechanism
            for node in statement.walk()
            if not _has_write_ancestor(node, statement)
            if (mechanism := _node_write_mechanism(node)) is not None
        }
        ordered = sorted(mechanisms, key=str)
        if len(mechanisms) > 1:
            return WriteStatementClassification(
                status="conflicting",
                mechanism="multiple",
                detected_mechanisms=ordered,
            )
        if len(mechanisms) == 1:
            mechanism = next(iter(mechanisms))
            return WriteStatementClassification(
                status="explicit",
                mechanism=mechanism,
                detected_mechanisms=[mechanism],
            )
        return WriteStatementClassification(
            status="absent",
            mechanism="write_statement_absent",
        )

    return WriteStatementClassification(
        status="unknown",
        mechanism="unsupported_statement",
    )


def _sql_literal(value: str) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _artifact_matches_pair(
    artifact: EvidenceArtifact,
    pair: ExactTablePair,
) -> bool:
    if artifact.tool_name not in _DIRECTED_TOOLS:
        return False
    args = artifact.compact_args
    return (
        str(args.get("source_table") or "").casefold()
        == pair.source_table.casefold()
        and str(args.get("target_table") or "").casefold()
        == pair.target_table.casefold()
    )


def _incomplete_fact(
    pair: ExactTablePair,
    *,
    matching_rows: int = 0,
    evidence_ids: Sequence[str] = (),
    mechanism: WriteSemanticsMechanism = "incomplete_evidence",
    detected_mechanisms: Sequence[WriteSemanticsMechanism] = (),
) -> WriteSemanticsFact:
    return WriteSemanticsFact(
        **pair.model_dump(),
        conclusion="incomplete",
        mechanism=mechanism,
        matching_rows=matching_rows,
        detected_mechanisms=list(detected_mechanisms)[
            :_MAX_DETECTED_MECHANISMS
        ],
        evidence_ids=list(evidence_ids)[:_MAX_EVIDENCE_IDS],
    )


def _fact_for_pair(
    pair: ExactTablePair,
    artifacts: Sequence[EvidenceArtifact],
    store: SavedResultStore,
) -> WriteSemanticsFact:
    matching_artifacts = [
        artifact
        for artifact in artifacts
        if _artifact_matches_pair(artifact, pair)
    ]
    evidence_ids = [
        str(evidence_id)[:120]
        for evidence_id in list(
            dict.fromkeys(
                artifact.evidence_id for artifact in matching_artifacts
            )
        )[:_MAX_EVIDENCE_IDS]
    ]
    if not matching_artifacts:
        return _incomplete_fact(pair)

    dataset_refs: List[str] = []
    for artifact in matching_artifacts:
        if artifact.dataset_ref is None:
            return _incomplete_fact(pair, evidence_ids=evidence_ids)
        descriptor = store.descriptor(artifact.dataset_ref)
        descriptor_columns = (
            {column.name.casefold() for column in descriptor.columns}
            if descriptor is not None
            else set()
        )
        if (
            descriptor is None
            or descriptor.source_tool != artifact.tool_name
            or descriptor.truncated
            or (
                descriptor.source_total is not None
                and descriptor.source_total > descriptor.row_count
            )
            or not _REQUIRED_DATASET_COLUMNS.issubset(descriptor_columns)
        ):
            return _incomplete_fact(pair, evidence_ids=evidence_ids)
        if artifact.dataset_ref not in dataset_refs:
            dataset_refs.append(artifact.dataset_ref)

    predicates = " AND ".join(
        (
            f"LOWER(source_table) = LOWER({_sql_literal(pair.source_table)})",
            f"LOWER(target_table) = LOWER({_sql_literal(pair.target_table)})",
        )
    )
    grouped_rules: Dict[object, int] = {}
    matching_rows = 0
    for dataset_ref in dataset_refs:
        count_result = store.query(
            result_ref=dataset_ref,
            query="SELECT COUNT(*) AS matching_rows FROM result WHERE "
            + predicates,
            preview_limit=1,
        )
        count_rows = count_result.get("rows", [])
        if (
            count_result.get("error")
            or count_result.get("input_truncated")
            or count_result.get("truncated")
            or len(count_rows) != 1
            or not isinstance(count_rows[0], dict)
        ):
            return _incomplete_fact(
                pair,
                matching_rows=matching_rows,
                evidence_ids=evidence_ids,
            )
        try:
            dataset_matching_rows = int(
                count_rows[0].get("matching_rows") or 0
            )
        except (TypeError, ValueError):
            return _incomplete_fact(
                pair,
                matching_rows=matching_rows,
                evidence_ids=evidence_ids,
            )
        if dataset_matching_rows < 0:
            return _incomplete_fact(
                pair,
                matching_rows=matching_rows,
                evidence_ids=evidence_ids,
            )
        matching_rows += dataset_matching_rows

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
            return _incomplete_fact(
                pair,
                matching_rows=matching_rows,
                evidence_ids=evidence_ids,
            )
        grouped_count = 0
        for row in rules_result.get("rows", []):
            if not isinstance(row, dict):
                return _incomplete_fact(
                    pair,
                    matching_rows=matching_rows,
                    evidence_ids=evidence_ids,
                )
            try:
                occurrence_count = int(row.get("occurrence_count") or 0)
            except (TypeError, ValueError):
                return _incomplete_fact(
                    pair,
                    matching_rows=matching_rows,
                    evidence_ids=evidence_ids,
                )
            if occurrence_count <= 0:
                return _incomplete_fact(
                    pair,
                    matching_rows=matching_rows,
                    evidence_ids=evidence_ids,
                )
            rule = row.get("transformation_rule")
            grouped_rules[rule] = grouped_rules.get(rule, 0) + occurrence_count
            grouped_count += occurrence_count
        if grouped_count != dataset_matching_rows:
            return _incomplete_fact(
                pair,
                matching_rows=matching_rows,
                evidence_ids=evidence_ids,
            )

    if matching_rows == 0:
        return WriteSemanticsFact(
            **pair.model_dump(),
            conclusion="not_assessed",
            mechanism="no_exact_mapping",
            matching_rows=0,
            evidence_ids=evidence_ids,
        )

    classifications = [
        classify_write_semantics_rule(rule) for rule in grouped_rules
    ]
    explicit_mechanisms = {
        mechanism
        for item in classifications
        for mechanism in item.detected_mechanisms
        if mechanism in _EXPLICIT_WRITE_MECHANISMS
    }
    per_rule_conflict = any(
        item.status == "conflicting" for item in classifications
    )
    unknown_mechanisms = {
        item.mechanism
        for item in classifications
        if item.status == "unknown"
    }
    detected_mechanisms = sorted(
        explicit_mechanisms,
        key=str,
    )[:_MAX_DETECTED_MECHANISMS]

    if per_rule_conflict or len(explicit_mechanisms) > 1:
        return WriteSemanticsFact(
            **pair.model_dump(),
            conclusion="conflicting",
            mechanism="multiple",
            matching_rows=matching_rows,
            detected_mechanisms=detected_mechanisms,
            evidence_ids=evidence_ids,
        )
    if unknown_mechanisms:
        mechanism = (
            next(iter(unknown_mechanisms))
            if len(unknown_mechanisms) == 1
            else "incomplete_evidence"
        )
        return _incomplete_fact(
            pair,
            matching_rows=matching_rows,
            evidence_ids=evidence_ids,
            mechanism=mechanism,
            detected_mechanisms=detected_mechanisms,
        )
    if len(explicit_mechanisms) == 1:
        mechanism = next(iter(explicit_mechanisms))
        if mechanism not in _EXPLICIT_WRITE_MECHANISMS:
            return _incomplete_fact(
                pair,
                matching_rows=matching_rows,
                evidence_ids=evidence_ids,
            )
        return WriteSemanticsFact(
            **pair.model_dump(),
            conclusion="explicit_write",
            mechanism=mechanism,
            matching_rows=matching_rows,
            detected_mechanisms=[mechanism],
            evidence_ids=evidence_ids,
        )
    return WriteSemanticsFact(
        **pair.model_dump(),
        conclusion="not_assessed",
        mechanism="write_statement_absent",
        matching_rows=matching_rows,
        evidence_ids=evidence_ids,
    )


def derive_write_semantics_facts(
    task: str,
    artifacts: Sequence[EvidenceArtifact],
    store: SavedResultStore,
) -> List[WriteSemanticsFact]:
    """Derive bounded facts from accepted full saved-result relations."""

    return [
        _fact_for_pair(pair, artifacts, store)
        for pair in extract_exact_table_pairs(task)
    ]


def write_semantics_payload(
    facts: Sequence[WriteSemanticsFact],
) -> Dict[str, object]:
    """Return a bounded payload without copying transformation-rule rows."""

    payload: Dict[str, object] = {
        "authority": "deterministic_sqlglot_full_saved_result",
        "scope_rule": (
            "Write mode is reported only for a parsed root write statement; "
            "SELECT/WITH and key metadata do not imply write semantics."
        ),
        "facts": [fact.model_dump(mode="json") for fact in facts],
    }
    if len(json.dumps(payload, ensure_ascii=False)) > (
        MAX_WRITE_SEMANTICS_PAYLOAD_CHARS
    ):
        compact_facts: List[Dict[str, object]] = []
        for fact in facts[:_MAX_TABLE_PAIRS]:
            item = fact.model_dump(mode="json")
            item["detected_mechanisms"] = item[
                "detected_mechanisms"
            ][:_MAX_DETECTED_MECHANISMS]
            item["evidence_ids"] = item["evidence_ids"][:4]
            compact_facts.append(item)
        payload["facts"] = compact_facts
        payload["payload_truncated"] = True
    if len(json.dumps(payload, ensure_ascii=False)) > (
        MAX_WRITE_SEMANTICS_PAYLOAD_CHARS
    ):
        raw_facts = payload.get("facts")
        if isinstance(raw_facts, list):
            for item in raw_facts:
                if isinstance(item, dict):
                    item["evidence_ids"] = []
                    item["detected_mechanisms"] = item[
                        "detected_mechanisms"
                    ][:2]
            while raw_facts and len(
                json.dumps(payload, ensure_ascii=False)
            ) > MAX_WRITE_SEMANTICS_PAYLOAD_CHARS:
                raw_facts.pop()
    return payload


def render_write_semantics_answer(
    facts: Sequence[WriteSemanticsFact],
) -> str:
    """Render exact-pair write-semantics conclusions without an LLM."""

    lines: List[str] = []
    for fact in facts:
        pair = f"`{fact.source_table} → {fact.target_table}`"
        if (
            fact.conclusion == "not_assessed"
            and fact.mechanism == "write_statement_absent"
        ):
            lines.append(
                f"Для {pair} write semantics не оценено: в полном "
                "сохранённом S2T-mapping нет явного write statement "
                "(INSERT, UPDATE, MERGE, DELETE или CTAS). Режим "
                "append/overwrite/upsert и conflict handling нельзя "
                "выводить из PK/UNIQUE."
            )
        elif fact.conclusion == "explicit_write":
            lines.append(
                f"Для {pair} в точном S2T-mapping обнаружен явный "
                f"write statement: `{fact.mechanism}`."
            )
        elif fact.conclusion == "conflicting":
            mechanisms = ", ".join(
                f"`{item}`" for item in fact.detected_mechanisms
            ) or "несколько режимов"
            lines.append(
                f"Для {pair} write semantics противоречива: "
                f"сохранены {mechanisms}."
            )
        elif fact.mechanism == "no_exact_mapping":
            lines.append(
                f"Для {pair} write semantics не оценено: в полном "
                "evidence нет строк точной направленной пары."
            )
        else:
            lines.append(
                f"Для {pair} write semantics не оценено: evidence неполно "
                f"или неоднозначно (`{fact.mechanism}`)."
            )
    return "\n".join(lines)


def render_terminal_write_semantics_negative(
    facts: Sequence[WriteSemanticsFact],
) -> str:
    """Render only a proven terminal negative; otherwise return an empty string."""

    if not facts or any(
        fact.conclusion != "not_assessed"
        or fact.mechanism != "write_statement_absent"
        or fact.matching_rows <= 0
        or not fact.evidence_ids
        for fact in facts
    ):
        return ""
    return render_write_semantics_answer(facts)


__all__ = [
    "ExactTablePair",
    "MAX_WRITE_SEMANTICS_PAYLOAD_CHARS",
    "WriteSemanticsConclusion",
    "WriteSemanticsFact",
    "WriteSemanticsMechanism",
    "WriteStatementClassification",
    "classify_write_semantics_rule",
    "derive_write_semantics_facts",
    "extract_exact_table_pairs",
    "is_exclusive_write_semantics_request",
    "render_terminal_write_semantics_negative",
    "render_write_semantics_answer",
    "write_semantics_payload",
]
