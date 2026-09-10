"""Deterministic nullable-constraint facts for exact accepted evidence.

This module answers one deliberately narrow question: whether the catalogued
nullable contract of an exact source field is compatible with the catalogued
``NOT NULL`` contract of its exact target field.  It does not infer runtime
values, database constraints outside the stored catalog, or guarantees from a
transformation preview.
"""

from __future__ import annotations

import json
import re
from typing import Dict, List, Literal, Sequence

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .contracts import EvidenceArtifact
from .sql_risk_scope_contract import (
    SqlRiskScopeContract,
    build_sql_risk_scope_contract,
)
from .tools.saved_results import SavedResultStore


ConstraintRejectionConclusion = Literal[
    "conditional_rejection_risk",
    "nullable_mismatch_not_detected",
    "not_assessed",
    "conflicting",
]
ConstraintRejectionMechanism = Literal[
    "nullable_source_to_not_null_target",
    "target_allows_null",
    "both_not_null",
    "unknown_source_not_null",
    "unknown_target_not_null",
    "missing_source_metadata",
    "missing_target_metadata",
    "missing_both_metadata",
    "conflicting_source_metadata",
    "conflicting_target_metadata",
    "no_exact_field_mapping",
    "incomplete_mapping_evidence",
    "incomplete_catalog_evidence",
    "scope_mismatch",
]

_MAPPING_TOOL = "read_s2t_source_to_target"
_METADATA_TOOL = "get_source_target_column_pair"
_MAPPING_COLUMNS = frozenset(
    {
        "source_table",
        "source_field",
        "target_table",
        "target_field",
        "transformation_rule",
    }
)
_METADATA_COLUMNS = frozenset(
    {
        "column_role",
        "file_id",
        "table_name",
        "column_name",
        "not_null",
    }
)
_MAX_EVIDENCE_IDS = 8
_MAX_IDENTIFIER_CHARS = 200
MAX_CONSTRAINT_REJECTION_PAYLOAD_CHARS = 8_000

_IDENTIFIER_ATOM = r"[A-Za-z_][A-Za-z0-9_$]*"
_FIELD_ENDPOINT = rf"{_IDENTIFIER_ATOM}(?:\.{_IDENTIFIER_ATOM}){{1,2}}"
_OUTPUT_CONTRACT = (
    r"(?:верни\s+`?source_not_null`?\s*=\s*<\s*0\s*\|\s*1\s*>"
    r"\s*,?\s*`?target_not_null`?\s*=\s*<\s*0\s*\|\s*1\s*>"
    r"\s*(?:,?\s*и\s+вывод)?|"
    r"return\s+`?source_not_null`?\s*=\s*<\s*0\s*\|\s*1\s*>"
    r"\s*,?\s*`?target_not_null`?\s*=\s*<\s*0\s*\|\s*1\s*>"
    r"\s*(?:,?\s*and\s+(?:a\s+)?conclusion)?)"
)
_SUPPORTED_NULLABLE_REQUEST_RE = re.compile(
    rf"^\s*(?:"
    rf"для\s+file_id\s*=\s*[1-9][0-9]*\s+оцени\s+только\s+"
    rf"(?:sql[-‑–— ]?риск\s+)?constraint[ _-]*rejection\s+из-за\s+"
    rf"nullable[-‑–— ]?ограничен\w*\s+{_FIELD_ENDPOINT}\s*→\s*"
    rf"{_FIELD_ENDPOINT}\s*[.!?]?\s*{_OUTPUT_CONTRACT}"
    rf"|for\s+file_id\s*=\s*[1-9][0-9]*\s+assess\s+only\s+"
    rf"(?:sql[ _-]*risk\s+)?constraint[ _-]*rejection\s+"
    rf"(?:caused\s+by|from)\s+nullable\s+constraints?\s+for\s+"
    rf"{_FIELD_ENDPOINT}\s*→\s*{_FIELD_ENDPOINT}\s*[.!?]?\s*"
    rf"{_OUTPUT_CONTRACT}"
    rf")\s*[.!?]?\s*$",
    re.IGNORECASE,
)

_CONSTRAINT_ASPECT = (
    r"(?:constraint[ _-]*rejection|"
    r"nullable[^.!?\r\n]{0,48}(?:огранич|constraint)|"
    r"(?:огранич|constraint)[^.!?\r\n]{0,48}nullable)"
)
_EXCLUSIVE_CONSTRAINT_RE = re.compile(
    rf"(?:\b(?:только|only)\b[^.!?\r\n]{{0,96}}{_CONSTRAINT_ASPECT}"
    rf"|{_CONSTRAINT_ASPECT}[^.!?\r\n]{{0,96}}\b(?:только|only)\b)",
    re.IGNORECASE,
)
_OTHER_SQL_RISK_RE = re.compile(
    r"\b(?:row[ _-]*filtering|cardinality|value[ _-]*changes?|"
    r"write[ _-]*semantics)\b"
    r"|потер\w*\s+строк\w*|дубликат\w*|кардинальн\w*|"
    r"изменен\w*\s+(?:значен|данн)\w*|"
    r"режим\w*\s+(?:запис|загруз)\w*",
    re.IGNORECASE,
)
_NULLABLE_INTENT_RE = re.compile(
    r"\bnullable\b|\bnot[ _-]*null\b|\bnullability\b|"
    r"(?:допуска\w*|запрещ\w*|огранич\w*)[^.!?\r\n]{0,32}\bnull\b",
    re.IGNORECASE,
)
_NULLABLE_OUTPUT_CONTRACT_RE = re.compile(
    r"(?:верни|return)\s+`?source_not_null`?\s*=\s*<\s*0\s*\|\s*1\s*>"
    r"\s*,?\s*`?target_not_null`?\s*=\s*<\s*0\s*\|\s*1\s*>"
    r"\s*(?:,?\s*(?:и\s+вывод|and\s+(?:a\s+)?conclusion))?\s*[.!?]?\s*$",
    re.IGNORECASE,
)
_OTHER_CONSTRAINT_KIND_RE = re.compile(
    r"\b(?:data[ _-]*type|primary[ _-]*key|foreign[ _-]*key|"
    r"unique(?:ness)?|check|default|referential[ _-]*integrity)\b"
    r"|\b(?:pk|fk)\b|(?:совместим\w*|несовместим\w*)\s+тип\w*|"
    r"первичн\w*\s+ключ\w*|внешн\w*\s+ключ\w*|уникальн\w*|"
    r"ссылочн\w*\s+целостн\w*|внешн\w*\s+ссыл\w*|дефолт\w*|"
    r"переполн\w*[^.!?\r\n]{0,24}(?:длин\w*|length)|"
    r"(?:length|длин\w*)[^.!?\r\n]{0,24}переполн\w*|"
    r"фактич\w*[^.!?\r\n]{0,32}\bnull\b|"
    r"\bnull\b[^.!?\r\n]{0,32}(?:в\s+)?данн\w*",
    re.IGNORECASE,
)
_UNSUPPORTED_DATA_OUTPUT_RE = re.compile(
    r"\b(?:data[ _-]*types?|description|mapping|transformation[ _-]*rule)\b|"
    r"\b(?:тип\w*\s+данн\w*|описан\w*|маппинг\w*|"
    r"правил\w*\s+трансформац\w*)\b",
    re.IGNORECASE,
)
_ADDITIONAL_RESULT_RE = re.compile(
    r"\b(?:также|дополнительно|also|additionally)\b|"
    r"\b(?:покажи|выведи|перечисли|верни|show|display|list|return)\b"
    r"[^.!?\r\n]{0,64}\b(?:все\s+|весь\s+|полный\s+)?"
    r"(?:строк\w*|маппинг\w*|правил\w*|sql|ddl|metadata|метаданн\w*|"
    r"каталог\w*|колон\w*)\b",
    re.IGNORECASE,
)


class ConstraintRejectionFact(BaseModel):
    """One code-derived nullable conclusion for a literal field pair."""

    model_config = ConfigDict(extra="forbid")

    file_id: int = Field(gt=0)
    source_table: str = Field(max_length=_MAX_IDENTIFIER_CHARS)
    source_field: str = Field(max_length=_MAX_IDENTIFIER_CHARS)
    target_table: str = Field(max_length=_MAX_IDENTIFIER_CHARS)
    target_field: str = Field(max_length=_MAX_IDENTIFIER_CHARS)
    source_not_null: Literal[0, 1] | None = None
    target_not_null: Literal[0, 1] | None = None
    conclusion: ConstraintRejectionConclusion
    mechanism: ConstraintRejectionMechanism
    mapping_rows: int = Field(default=0, ge=0)
    exact_field_rows: int = Field(default=0, ge=0)
    source_metadata_rows: int = Field(default=0, ge=0)
    target_metadata_rows: int = Field(default=0, ge=0)
    evidence_ids: List[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_conclusion_flags(self) -> "ConstraintRejectionFact":
        """Keep public flags consistent with the rendered conclusion."""

        if self.conclusion == "conditional_rejection_risk" and (
            self.mechanism != "nullable_source_to_not_null_target"
            or (self.source_not_null, self.target_not_null) != (0, 1)
        ):
            raise ValueError(
                "conditional rejection requires source_not_null=0 and "
                "target_not_null=1"
            )
        if self.conclusion == "nullable_mismatch_not_detected":
            valid = (
                self.mechanism == "target_allows_null"
                and self.source_not_null in {0, 1}
                and self.target_not_null == 0
            ) or (
                self.mechanism == "both_not_null"
                and (self.source_not_null, self.target_not_null) == (1, 1)
            )
            if not valid:
                raise ValueError(
                    "not-detected nullable conclusion contradicts flags"
                )
        if self.conclusion == "conflicting" and (
            self.mechanism
            not in {
                "conflicting_source_metadata",
                "conflicting_target_metadata",
            }
            or self.source_not_null is not None
            or self.target_not_null is not None
        ):
            raise ValueError(
                "conflicting metadata must not expose a chosen not_null flag"
            )
        return self

    @property
    def source_reference(self) -> str:
        return f"{self.source_table}.{self.source_field}"

    @property
    def target_reference(self) -> str:
        return f"{self.target_table}.{self.target_field}"


def _exact_contract(task: str) -> SqlRiskScopeContract | None:
    contract = build_sql_risk_scope_contract(
        task,
        ["constraint_rejection"],
        enabled=True,
    )
    if (
        contract is None
        or not contract.scope.is_field_pair
        or contract.scope.file_id is None
        or contract.scope.source_field is None
        or contract.scope.target_field is None
        or any(
            len(value) > _MAX_IDENTIFIER_CHARS
            for value in (
                contract.scope.source_table,
                contract.scope.source_field,
                contract.scope.target_table,
                contract.scope.target_field,
            )
        )
        or tuple(item.tool_name for item in contract.requirements)
        != (_MAPPING_TOOL, _METADATA_TOOL)
    ):
        return None
    return contract


def is_exclusive_constraint_rejection_request(task: str) -> bool:
    """Return whether a deterministic nullable-only answer is safe."""

    if not isinstance(task, str) or not task.strip():
        return False
    return bool(
        _exact_contract(task) is not None
        and _SUPPORTED_NULLABLE_REQUEST_RE.fullmatch(task)
        and _EXCLUSIVE_CONSTRAINT_RE.search(task)
        and _NULLABLE_INTENT_RE.search(task)
        and _NULLABLE_OUTPUT_CONTRACT_RE.search(task)
        and not _OTHER_SQL_RISK_RE.search(task)
        and not _OTHER_CONSTRAINT_KIND_RE.search(task)
        and not _UNSUPPORTED_DATA_OUTPUT_RE.search(task)
        and not _ADDITIONAL_RESULT_RE.search(task)
    )


def _sql_literal(value: str) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _exact_arguments(
    contract: SqlRiskScopeContract,
    tool_name: str,
) -> Dict[str, object]:
    for requirement in contract.requirements:
        if requirement.tool_name == tool_name:
            return dict(requirement.arguments)
    return {}


def _matching_artifacts(
    artifacts: Sequence[EvidenceArtifact],
    *,
    tool_name: str,
    arguments: Dict[str, object],
) -> List[EvidenceArtifact]:
    return [
        artifact
        for artifact in artifacts
        if artifact.tool_name == tool_name
        and artifact.compact_args == arguments
    ]


def _evidence_ids(
    *artifact_groups: Sequence[EvidenceArtifact],
) -> List[str]:
    return [
        str(value)[:120]
        for value in list(
            dict.fromkeys(
                artifact.evidence_id
                for group in artifact_groups
                for artifact in group
            )
        )[:_MAX_EVIDENCE_IDS]
    ]


def _base_fact(
    contract: SqlRiskScopeContract,
    *,
    conclusion: ConstraintRejectionConclusion,
    mechanism: ConstraintRejectionMechanism,
    evidence_ids: Sequence[str] = (),
    mapping_rows: int = 0,
    exact_field_rows: int = 0,
    source_metadata_rows: int = 0,
    target_metadata_rows: int = 0,
    source_not_null: Literal[0, 1] | None = None,
    target_not_null: Literal[0, 1] | None = None,
) -> ConstraintRejectionFact:
    scope = contract.scope
    assert scope.file_id is not None
    assert scope.source_field is not None
    assert scope.target_field is not None
    return ConstraintRejectionFact(
        file_id=scope.file_id,
        source_table=scope.source_table,
        source_field=scope.source_field,
        target_table=scope.target_table,
        target_field=scope.target_field,
        source_not_null=source_not_null,
        target_not_null=target_not_null,
        conclusion=conclusion,
        mechanism=mechanism,
        mapping_rows=mapping_rows,
        exact_field_rows=exact_field_rows,
        source_metadata_rows=source_metadata_rows,
        target_metadata_rows=target_metadata_rows,
        evidence_ids=list(evidence_ids)[:_MAX_EVIDENCE_IDS],
    )


def _validated_dataset_refs(
    artifacts: Sequence[EvidenceArtifact],
    store: SavedResultStore,
    *,
    required_columns: frozenset[str],
) -> List[str] | None:
    if not artifacts:
        return None
    refs: List[str] = []
    for artifact in artifacts:
        if artifact.dataset_ref is None:
            return None
        descriptor = store.descriptor(artifact.dataset_ref)
        columns = (
            {column.name.casefold() for column in descriptor.columns}
            if descriptor is not None
            else set()
        )
        if (
            descriptor is None
            or descriptor.source_tool != artifact.tool_name
            or descriptor.truncated
            or descriptor.source_total is None
            or descriptor.source_total != descriptor.row_count
            or not required_columns.issubset(columns)
        ):
            return None
        if artifact.dataset_ref not in refs:
            refs.append(artifact.dataset_ref)
    return refs


def _query_one(
    store: SavedResultStore,
    *,
    result_ref: str,
    query: str,
) -> Dict[str, object] | None:
    result = store.query(
        result_ref=result_ref,
        query=query,
        preview_limit=1,
    )
    rows = result.get("rows") or []
    if (
        result.get("error")
        or result.get("input_truncated")
        or result.get("truncated")
        or len(rows) != 1
        or not isinstance(rows[0], dict)
    ):
        return None
    return dict(rows[0])


def _safe_count(row: Dict[str, object], name: str) -> int | None:
    try:
        value = int(row.get(name) or 0)
    except (TypeError, ValueError):
        return None
    return value if value >= 0 else None


def _mapping_counts(
    contract: SqlRiskScopeContract,
    refs: Sequence[str],
    store: SavedResultStore,
) -> tuple[int, int, bool] | None:
    scope = contract.scope
    assert scope.source_field is not None
    assert scope.target_field is not None
    table_predicate = (
        "LOWER(TRIM(source_table)) = "
        f"LOWER(TRIM({_sql_literal(scope.source_table)})) AND "
        "LOWER(TRIM(target_table)) = "
        f"LOWER(TRIM({_sql_literal(scope.target_table)}))"
    )
    field_predicate = (
        table_predicate
        + " AND LOWER(TRIM(source_field)) = "
        f"LOWER(TRIM({_sql_literal(scope.source_field)}))"
        + " AND LOWER(TRIM(target_field)) = "
        f"LOWER(TRIM({_sql_literal(scope.target_field)}))"
    )
    total_rows = 0
    exact_field_rows = 0
    scope_matches = True
    for result_ref in refs:
        row = _query_one(
            store,
            result_ref=result_ref,
            query=(
                "SELECT COUNT(*) AS total_rows, "
                "SUM(CASE WHEN "
                + table_predicate
                + " THEN 1 ELSE 0 END) AS exact_table_rows, "
                "SUM(CASE WHEN "
                + field_predicate
                + " THEN 1 ELSE 0 END) AS exact_field_rows FROM result"
            ),
        )
        if row is None:
            return None
        dataset_total = _safe_count(row, "total_rows")
        exact_tables = _safe_count(row, "exact_table_rows")
        exact_fields = _safe_count(row, "exact_field_rows")
        if None in {dataset_total, exact_tables, exact_fields}:
            return None
        assert dataset_total is not None
        assert exact_tables is not None
        assert exact_fields is not None
        total_rows += dataset_total
        exact_field_rows += exact_fields
        scope_matches = scope_matches and exact_tables == dataset_total
    return total_rows, exact_field_rows, scope_matches


def _metadata_summary(
    contract: SqlRiskScopeContract,
    refs: Sequence[str],
    store: SavedResultStore,
) -> Dict[str, object] | None:
    scope = contract.scope
    assert scope.file_id is not None
    assert scope.source_field is not None
    assert scope.target_field is not None
    source_predicate = (
        "LOWER(TRIM(column_role)) = 'source' AND file_id = "
        f"{scope.file_id} AND LOWER(TRIM(table_name)) = "
        f"LOWER(TRIM({_sql_literal(scope.source_table)})) AND "
        "LOWER(TRIM(column_name)) = "
        f"LOWER(TRIM({_sql_literal(scope.source_field)}))"
    )
    target_predicate = (
        "LOWER(TRIM(column_role)) = 'target' AND file_id = "
        f"{scope.file_id} AND LOWER(TRIM(table_name)) = "
        f"LOWER(TRIM({_sql_literal(scope.target_table)})) AND "
        "LOWER(TRIM(column_name)) = "
        f"LOWER(TRIM({_sql_literal(scope.target_field)}))"
    )
    summary: Dict[str, object] = {
        "total_rows": 0,
        "source_rows": 0,
        "target_rows": 0,
        "invalid_rows": 0,
        "invalid_flags": 0,
        "source_unknown": 0,
        "target_unknown": 0,
        "source_values": set(),
        "target_values": set(),
    }
    for result_ref in refs:
        row = _query_one(
            store,
            result_ref=result_ref,
            query=(
                "SELECT COUNT(*) AS total_rows, "
                "SUM(CASE WHEN "
                + source_predicate
                + " THEN 1 ELSE 0 END) AS source_rows, "
                "SUM(CASE WHEN "
                + target_predicate
                + " THEN 1 ELSE 0 END) AS target_rows, "
                "SUM(CASE WHEN NOT (("
                + source_predicate
                + ") OR ("
                + target_predicate
                + ")) THEN 1 ELSE 0 END) AS invalid_rows, "
                "SUM(CASE WHEN ("
                + source_predicate
                + ") AND not_null IS NULL THEN 1 ELSE 0 END) "
                "AS source_unknown, "
                "SUM(CASE WHEN ("
                + target_predicate
                + ") AND not_null IS NULL THEN 1 ELSE 0 END) "
                "AS target_unknown, "
                "SUM(CASE WHEN not_null IS NOT NULL AND "
                "(TYPEOF(not_null) <> 'integer' OR not_null NOT IN (0, 1)) "
                "THEN 1 ELSE 0 END) AS invalid_flags, "
                "MIN(CASE WHEN "
                + source_predicate
                + " THEN not_null END) AS source_min, "
                "MAX(CASE WHEN "
                + source_predicate
                + " THEN not_null END) AS source_max, "
                "MIN(CASE WHEN "
                + target_predicate
                + " THEN not_null END) AS target_min, "
                "MAX(CASE WHEN "
                + target_predicate
                + " THEN not_null END) AS target_max FROM result"
            ),
        )
        if row is None:
            return None
        for name in (
            "total_rows",
            "source_rows",
            "target_rows",
            "invalid_rows",
            "source_unknown",
            "target_unknown",
            "invalid_flags",
        ):
            value = _safe_count(row, name)
            if value is None:
                return None
            summary[name] = int(summary[name]) + value
        for role in ("source", "target"):
            values = summary[f"{role}_values"]
            assert isinstance(values, set)
            for name in (f"{role}_min", f"{role}_max"):
                value = row.get(name)
                if value is not None:
                    if isinstance(value, bool):
                        value = int(value)
                    if not isinstance(value, int) or value not in {0, 1}:
                        return None
                    values.add(value)
    return summary


def _fact_for_contract(
    contract: SqlRiskScopeContract,
    artifacts: Sequence[EvidenceArtifact],
    store: SavedResultStore,
) -> ConstraintRejectionFact:
    mapping_artifacts = _matching_artifacts(
        artifacts,
        tool_name=_MAPPING_TOOL,
        arguments=_exact_arguments(contract, _MAPPING_TOOL),
    )
    metadata_artifacts = _matching_artifacts(
        artifacts,
        tool_name=_METADATA_TOOL,
        arguments=_exact_arguments(contract, _METADATA_TOOL),
    )
    evidence_ids = _evidence_ids(mapping_artifacts)

    mapping_refs = _validated_dataset_refs(
        mapping_artifacts,
        store,
        required_columns=_MAPPING_COLUMNS,
    )
    if mapping_refs is None:
        return _base_fact(
            contract,
            conclusion="not_assessed",
            mechanism="incomplete_mapping_evidence",
            evidence_ids=evidence_ids,
        )
    mapping_counts = _mapping_counts(contract, mapping_refs, store)
    if mapping_counts is None:
        return _base_fact(
            contract,
            conclusion="not_assessed",
            mechanism="incomplete_mapping_evidence",
            evidence_ids=evidence_ids,
        )
    mapping_rows, exact_field_rows, scope_matches = mapping_counts
    if not scope_matches:
        return _base_fact(
            contract,
            conclusion="not_assessed",
            mechanism="scope_mismatch",
            evidence_ids=evidence_ids,
            mapping_rows=mapping_rows,
            exact_field_rows=exact_field_rows,
        )
    if exact_field_rows == 0:
        return _base_fact(
            contract,
            conclusion="not_assessed",
            mechanism="no_exact_field_mapping",
            evidence_ids=evidence_ids,
            mapping_rows=mapping_rows,
        )

    evidence_ids = _evidence_ids(mapping_artifacts, metadata_artifacts)
    metadata_refs = _validated_dataset_refs(
        metadata_artifacts,
        store,
        required_columns=_METADATA_COLUMNS,
    )
    if metadata_refs is None:
        return _base_fact(
            contract,
            conclusion="not_assessed",
            mechanism="incomplete_catalog_evidence",
            evidence_ids=evidence_ids,
            mapping_rows=mapping_rows,
            exact_field_rows=exact_field_rows,
        )
    metadata = _metadata_summary(contract, metadata_refs, store)
    if metadata is None:
        return _base_fact(
            contract,
            conclusion="not_assessed",
            mechanism="incomplete_catalog_evidence",
            evidence_ids=evidence_ids,
            mapping_rows=mapping_rows,
            exact_field_rows=exact_field_rows,
        )
    source_rows = int(metadata["source_rows"])
    target_rows = int(metadata["target_rows"])
    common = {
        "evidence_ids": evidence_ids,
        "mapping_rows": mapping_rows,
        "exact_field_rows": exact_field_rows,
        "source_metadata_rows": source_rows,
        "target_metadata_rows": target_rows,
    }
    if int(metadata["invalid_flags"]) > 0:
        return _base_fact(
            contract,
            conclusion="not_assessed",
            mechanism="incomplete_catalog_evidence",
            **common,
        )
    if (
        int(metadata["invalid_rows"]) > 0
        or source_rows + target_rows != int(metadata["total_rows"])
    ):
        return _base_fact(
            contract,
            conclusion="not_assessed",
            mechanism="scope_mismatch",
            **common,
        )
    if source_rows == 0 and target_rows == 0:
        return _base_fact(
            contract,
            conclusion="not_assessed",
            mechanism="missing_both_metadata",
            **common,
        )
    if source_rows == 0:
        return _base_fact(
            contract,
            conclusion="not_assessed",
            mechanism="missing_source_metadata",
            **common,
        )
    if target_rows == 0:
        return _base_fact(
            contract,
            conclusion="not_assessed",
            mechanism="missing_target_metadata",
            **common,
        )

    source_values = metadata["source_values"]
    target_values = metadata["target_values"]
    assert isinstance(source_values, set)
    assert isinstance(target_values, set)
    if len(source_values) > 1:
        return _base_fact(
            contract,
            conclusion="conflicting",
            mechanism="conflicting_source_metadata",
            **common,
        )
    if len(target_values) > 1:
        return _base_fact(
            contract,
            conclusion="conflicting",
            mechanism="conflicting_target_metadata",
            **common,
        )
    source_not_null = next(iter(source_values), None)
    target_not_null = next(iter(target_values), None)
    if int(metadata["source_unknown"]) > 0 or source_not_null is None:
        return _base_fact(
            contract,
            conclusion="not_assessed",
            mechanism="unknown_source_not_null",
            source_not_null=None,
            target_not_null=target_not_null,
            **common,
        )
    if int(metadata["target_unknown"]) > 0 or target_not_null is None:
        return _base_fact(
            contract,
            conclusion="not_assessed",
            mechanism="unknown_target_not_null",
            source_not_null=source_not_null,
            target_not_null=None,
            **common,
        )
    if source_not_null == 0 and target_not_null == 1:
        conclusion: ConstraintRejectionConclusion = (
            "conditional_rejection_risk"
        )
        mechanism: ConstraintRejectionMechanism = (
            "nullable_source_to_not_null_target"
        )
    elif target_not_null == 0:
        conclusion = "nullable_mismatch_not_detected"
        mechanism = "target_allows_null"
    else:
        conclusion = "nullable_mismatch_not_detected"
        mechanism = "both_not_null"
    return _base_fact(
        contract,
        conclusion=conclusion,
        mechanism=mechanism,
        source_not_null=source_not_null,
        target_not_null=target_not_null,
        **common,
    )


def derive_constraint_rejection_facts(
    task: str,
    artifacts: Sequence[EvidenceArtifact],
    store: SavedResultStore,
) -> List[ConstraintRejectionFact]:
    """Derive one fact from full exact mapping and metadata relations."""

    contract = _exact_contract(task)
    if contract is None:
        return []
    return [_fact_for_contract(contract, artifacts, store)]


def constraint_rejection_payload(
    facts: Sequence[ConstraintRejectionFact],
) -> Dict[str, object]:
    """Return bounded facts without copying catalog or transformation rows."""

    bounded_facts = []
    for fact in facts:
        dumped = fact.model_dump(mode="json")
        dumped["evidence_ids"] = [
            str(value)[:120]
            for value in fact.evidence_ids[:_MAX_EVIDENCE_IDS]
        ]
        bounded_facts.append(dumped)
    payload: Dict[str, object] = {
        "authority": (
            "deterministic_exact_s2t_and_catalog_full_saved_results"
        ),
        "scope_rule": (
            "The conclusion covers only the exact field pair's catalogued "
            "nullable/NOT NULL mismatch. It does not prove actual NULL values "
            "or the absence of other database constraints."
        ),
        "facts": bounded_facts,
    }
    if (
        len(json.dumps(payload, ensure_ascii=False))
        > MAX_CONSTRAINT_REJECTION_PAYLOAD_CHARS
    ):
        payload["facts"] = bounded_facts[:1]
        payload["payload_truncated"] = True
    return payload


def _flag_text(value: Literal[0, 1] | None) -> str:
    return "unknown" if value is None else str(value)


def render_constraint_rejection_answer(
    facts: Sequence[ConstraintRejectionFact],
) -> str:
    """Render nullable facts without another model call."""

    blocks: List[str] = []
    for fact in facts:
        pair = f"`{fact.source_reference} → {fact.target_reference}`"
        prefix = (
            f"source_not_null={_flag_text(fact.source_not_null)}\n"
            f"target_not_null={_flag_text(fact.target_not_null)}\n"
        )
        if fact.conclusion == "conditional_rejection_risk":
            conclusion = (
                f"Вывод: для {pair} подтверждён условный риск constraint "
                "rejection: source допускает NULL, а target запрещает NULL. "
                "Фактическое отклонение возможно только если точная target-"
                "проекция действительно выдаст NULL."
            )
        elif fact.conclusion == "nullable_mismatch_not_detected":
            detail = (
                "target допускает NULL"
                if fact.mechanism == "target_allows_null"
                else "обе catalog-записи имеют not_null=1"
            )
            conclusion = (
                f"Вывод: для {pair} catalogued nullable mismatch не "
                f"обнаружен: {detail}. Это не доказывает отсутствие rejection "
                "из-за преобразования значений или других constraints."
            )
        elif fact.conclusion == "conflicting":
            conclusion = (
                f"Вывод: для {pair} риск constraint rejection не оценён: "
                f"catalog metadata противоречива (`{fact.mechanism}`)."
            )
        else:
            conclusion = (
                f"Вывод: для {pair} риск constraint rejection не оценён: "
                f"`{fact.mechanism}`. Пустое или неполное exact evidence не "
                "доказывает отсутствие constraints."
            )
        blocks.append(prefix + conclusion)
    return "\n\n".join(blocks)


__all__ = [
    "ConstraintRejectionConclusion",
    "ConstraintRejectionFact",
    "ConstraintRejectionMechanism",
    "MAX_CONSTRAINT_REJECTION_PAYLOAD_CHARS",
    "constraint_rejection_payload",
    "derive_constraint_rejection_facts",
    "is_exclusive_constraint_rejection_request",
    "render_constraint_rejection_answer",
]
