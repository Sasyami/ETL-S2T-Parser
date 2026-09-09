"""Fail-closed sufficiency check for conditional cardinality analysis.

The coordinator may ignore an upstream request for redundant metadata only
when the original task asks for a conditional duplicate/cardinality risk and
one accepted exact directed reader produced a demonstrably complete saved
mapping.  This module never claims factual counts or uniqueness.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Sequence

from .contracts import EvidenceArtifact
from .tools.saved_results import SavedResultStore


_EXACT_DIRECTED_TOOLS = frozenset(
    {"read_s2t_mapping", "read_s2t_source_to_target"}
)
_REQUIRED_COLUMNS = frozenset(
    {"source_table", "target_table", "transformation_rule"}
)

_IDENTIFIER_PART = (
    r"(?:[A-Za-z_\u0410-\u042f\u0430-\u044f\u0401\u0451$]"
    r"[A-Za-z0-9_\u0410-\u042f\u0430-\u044f\u0401\u0451$-]*|"
    r"`[^`\r\n.]+`|\"[^\"\r\n.]+\")"
)
# Whitespace around dots is deliberately unsupported.  Apart from being an
# unusual spelling for a physical name, accepting it makes the full stop in
# ``tgt_table. Следующее предложение`` look like another identifier segment.
_ENDPOINT = rf"{_IDENTIFIER_PART}(?:\.{_IDENTIFIER_PART}){{0,3}}"
_PAIR_RE = re.compile(
    rf"(?<![\w$`\"])(?P<source>{_ENDPOINT})\s*"
    rf"(?:→|->|=>)\s*(?P<target>{_ENDPOINT})(?![\w$`\"])",
    re.IGNORECASE,
)

_CARDINALITY_INTENT_RE = re.compile(
    r"(?:\bcardinality\b|\bduplicate(?:s|d)?\b|\bmany[- ]to[- ]many\b|"
    r"\bone[- ]to[- ]many\b|\brow\w*\s+multipli\w*|"
    r"\b1\s*:\s*n\b|\bn\s*:\s*m\b|"
    r"\bкардинальн\w*|дубликат\w*|размнож\w*\s+строк\w*)",
    re.IGNORECASE,
)
_FACTUAL_COUNT_RE = re.compile(
    r"\b(?:сколько|посчитай|подсчитай|сосчитай|вычисли|count)\b"
    r"|\bcount\s*\("
    r"|\b(?:фактическ|реальн|точн)\w*\s+"
    r"(?:количеств|числ|count)\w*"
    r"|\b(?:количеств|числ)\w*\s+(?:фактическ|реальн|точн)\w*",
    re.IGNORECASE,
)
_UNIQUENESS_FACT_RE = re.compile(
    r"\b(?:проверь|проверить|подтверди|подтвердить|определи|определить|"
    r"узнай|установи|установить|прочитай|получи|найди|верни|покажи|"
    r"перечисли|check|verify|fetch|read|show|list)\b"
    r"[^.!?\r\n]{0,96}"
    r"(?:уникальн\w*|первичн\w*\s+ключ\w*|\bpk\b|"
    r"primary[ _-]*key|unique[ _-]*(?:key|constraint|index)|метаданн\w*)"
    r"|\b(?:уникален|уникальна|уникальны|unique)\s+ли\b"
    r"|\b(?:есть|имеется|существует|has|have)\s+ли\b"
    r"[^.!?\r\n]{0,64}"
    r"(?:\bpk\b|первичн\w*\s+ключ\w*|unique[ _-]*(?:key|constraint|index))"
    r"|\b(?:фактическ|реальн|сохран[её]нн)\w*\s+уникальн\w*",
    re.IGNORECASE,
)
_ADDITIONAL_INTENT_RE = re.compile(
    r"\b(?:также|дополнительно|also|additionally)\b"
    r"|\b(?:а|и)\s+ещ[её]\b|\bкроме\s+того\b"
    r"|\b(?:покажи|выведи|перечисли|верни|show|display|list)\b"
    r"[^.!?\r\n]{0,64}\b(?:все\s+)?"
    r"(?:строк\w*|пол\w*|колон\w*|каталог\w*|metadata|ddl)\b",
    re.IGNORECASE,
)
_OTHER_SQL_RISK_RE = re.compile(
    r"\b(?:row[ _-]*filtering|constraint[ _-]*rejection|"
    r"value[ _-]*changes?|write[ _-]*semantics)\b"
    r"|потер\w*\s+строк\w*|отсеч\w*\s+строк\w*"
    r"|изменен\w*\s+значен\w*|режим\w*\s+(?:запис|загруз)\w*"
    r"|(?:not[ _-]*null|тип\w*|constraint\w*)\s+"
    r"(?:ошиб|отклон|риск)\w*",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ExactTablePair:
    """One conservative literal technical table pair from the task."""

    source_table: str
    target_table: str


def _clean_endpoint(value: str) -> str:
    clean = str(value or "").strip().strip("`\"")
    return ".".join(part.strip("`\"") for part in clean.split("."))


def _is_technical_identifier(value: str) -> bool:
    return bool(value) and any(marker in value for marker in ("_", ".", "$"))


def extract_literal_table_pairs(task: str) -> List[ExactTablePair]:
    """Extract unique literal technical source→target pairs, bounded."""

    pairs: List[ExactTablePair] = []
    seen: set[tuple[str, str]] = set()
    for match in _PAIR_RE.finditer(str(task or "")):
        source_table = _clean_endpoint(match.group("source"))
        target_table = _clean_endpoint(match.group("target"))
        if not (
            _is_technical_identifier(source_table)
            and _is_technical_identifier(target_table)
        ):
            continue
        identity = (source_table.casefold(), target_table.casefold())
        if identity in seen:
            continue
        seen.add(identity)
        pairs.append(ExactTablePair(source_table, target_table))
        if len(pairs) >= 8:
            break
    return pairs


def is_conditional_cardinality_request(task: str) -> bool:
    """Return whether exact mapping alone can support a conditional answer.

    Factual row counts, factual uniqueness/constraint checks, other SQL-risk
    aspects and explicit secondary data intents all fail closed.
    """

    text = str(task or "").strip()
    if not text or len(extract_literal_table_pairs(text)) != 1:
        return False
    if not _CARDINALITY_INTENT_RE.search(text):
        return False
    return not any(
        pattern.search(text)
        for pattern in (
            _FACTUAL_COUNT_RE,
            _UNIQUENESS_FACT_RE,
            _ADDITIONAL_INTENT_RE,
            _OTHER_SQL_RISK_RE,
        )
    )


def _sql_literal(value: str) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def complete_cardinality_mapping_evidence_ids(
    task: str,
    artifacts: Sequence[EvidenceArtifact],
    store: SavedResultStore,
) -> List[str]:
    """Return evidence IDs that prove one exact mapping is fully materialized.

    The descriptor must expose a trustworthy source total equal to its saved
    row count, be untruncated, contain the required columns, and every saved
    row must match the literal directed pair and carry a non-empty rule.
    """

    if not is_conditional_cardinality_request(task):
        return []
    pair = extract_literal_table_pairs(task)[0]
    evidence_ids: List[str] = []
    for artifact in artifacts:
        if artifact.tool_name not in _EXACT_DIRECTED_TOOLS:
            continue
        args = artifact.compact_args
        if (
            str(args.get("source_table") or "").casefold()
            != pair.source_table.casefold()
            or str(args.get("target_table") or "").casefold()
            != pair.target_table.casefold()
            or artifact.dataset_ref is None
        ):
            continue
        descriptor = store.descriptor(artifact.dataset_ref)
        if (
            descriptor is None
            or descriptor.source_tool != artifact.tool_name
            or descriptor.truncated
            or descriptor.row_count <= 0
            or descriptor.source_total is None
            or descriptor.source_total != descriptor.row_count
            or not _REQUIRED_COLUMNS.issubset(
                {column.name for column in descriptor.columns}
            )
        ):
            continue

        source = _sql_literal(pair.source_table)
        target = _sql_literal(pair.target_table)
        result = store.query(
            result_ref=descriptor.result_ref,
            query=(
                "SELECT COUNT(*) AS saved_rows, "
                "SUM(CASE WHEN LOWER(TRIM(source_table)) = "
                f"LOWER(TRIM({source})) AND LOWER(TRIM(target_table)) = "
                f"LOWER(TRIM({target})) THEN 1 ELSE 0 END) AS exact_rows, "
                "SUM(CASE WHEN NULLIF(TRIM(transformation_rule), '') "
                "IS NULL OR TRIM(transformation_rule) = '-' "
                "THEN 1 ELSE 0 END) AS missing_rule_rows FROM result"
            ),
            preview_limit=1,
        )
        rows = result.get("rows") or []
        if (
            result.get("error")
            or result.get("input_truncated")
            or result.get("truncated")
            or len(rows) != 1
            or not isinstance(rows[0], dict)
            or int(rows[0].get("saved_rows") or 0) != descriptor.row_count
            or int(rows[0].get("exact_rows") or 0) != descriptor.row_count
            or int(rows[0].get("missing_rule_rows") or 0) != 0
        ):
            continue
        if artifact.evidence_id not in evidence_ids:
            evidence_ids.append(artifact.evidence_id)
    return evidence_ids


__all__ = [
    "ExactTablePair",
    "complete_cardinality_mapping_evidence_ids",
    "extract_literal_table_pairs",
    "is_conditional_cardinality_request",
]
