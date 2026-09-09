"""Deterministic completeness checks for clean downstream reroute plans."""

from __future__ import annotations

import re
from typing import Iterable

from .cardinality_sufficiency import is_conditional_cardinality_request
from .contracts import SqlRiskAspect, WorkerPlan


_ARROW_PAIR_RE = re.compile(
    r"(?P<source>[A-Za-zА-Яа-яЁё0-9_$]"
    r"[A-Za-zА-Яа-яЁё0-9_$.-]*)"
    r"[\s`\"')\]}]*"
    r"(?:→|->|=>)"
    r"[\s`\"'([{]*"
    r"(?P<target>[A-Za-zА-Яа-яЁё0-9_$]"
    r"[A-Za-zА-Яа-яЁё0-9_$.-]*)"
)
_MAPPING_READ_RE = re.compile(
    r"(?:\bs2t\b|\bmapping\b|маппинг|правил)",
    re.IGNORECASE,
)
_TRANSFORMATION_READ_RE = re.compile(
    r"(?:прочит\w*|получ\w*|извлеч\w*|найт\w*|покаж\w*)\s+"
    r"(?:(?:полн\w*|точн\w*|сохран\w*|directed)\s+)*"
    r"трансформац\w*"
    r"|\bread\s+(?:(?:full|exact|saved|directed)\s+)*transformation\b",
    re.IGNORECASE,
)
_COLUMN_METADATA_READ_RE = re.compile(
    r"(?:\bcolumn\s+metadata\b|\bmetadata\b|метаданн\w*|"
    r"каталог\w*\s+колон\w*|nullable|not[_ ]?null|constraint)",
    re.IGNORECASE,
)
_EVIDENCE_READ_RE = re.compile(
    r"(?:прочит\w*|получ\w*|извлеч\w*|найт\w*|покаж\w*|"
    r"\bread\b|\bget\b|\bfetch\b)",
    re.IGNORECASE,
)
MAPPING_DEPENDENT_SQL_RISK_ASPECTS: frozenset[SqlRiskAspect] = frozenset(
    {
        "row_filtering",
        "cardinality",
        "constraint_rejection",
        "value_changes",
        "write_semantics",
    }
)


class ReroutePlanRequirementError(ValueError):
    """Raised when a clean reroute omits required base evidence."""


class SqlRiskPlanRequirementError(ValueError):
    """Raised when a typed SQL-risk plan omits required evidence shape."""


def _clean_identifier(value: str) -> str:
    return value.strip(". `\"'()[]{}<>,:;").casefold()


def _technical_forms(value: str) -> tuple[str, ...]:
    """Return compatible full/table forms for a technical endpoint."""

    clean = _clean_identifier(value)
    if not clean or not any(marker in clean for marker in ("_", ".", "$")):
        return ()
    if "." not in clean:
        return (clean,)
    table = clean.rsplit(".", 1)[0]
    return tuple(dict.fromkeys((clean, table)))


def _directed_pairs(
    value: str,
) -> Iterable[tuple[tuple[str, ...], tuple[str, ...]]]:
    """Yield only conservative technical pairs written around an arrow."""

    for match in _ARROW_PAIR_RE.finditer(str(value or "")):
        source_forms = _technical_forms(match.group("source"))
        target_forms = _technical_forms(match.group("target"))
        if source_forms and target_forms:
            yield source_forms, target_forms


def _same_endpoint(
    candidate_forms: tuple[str, ...],
    expected_forms: tuple[str, ...],
) -> bool:
    return bool(set(candidate_forms) & set(expected_forms))


def _step_reads_directed_mapping(
    task: str,
    source_forms: tuple[str, ...],
    target_forms: tuple[str, ...],
) -> bool:
    """Return whether one self-contained task re-reads the directed mapping."""

    if not (
        _MAPPING_READ_RE.search(task)
        or _TRANSFORMATION_READ_RE.search(task)
    ):
        return False
    return any(
        _same_endpoint(candidate_source, source_forms)
        and _same_endpoint(candidate_target, target_forms)
        for candidate_source, candidate_target in _directed_pairs(task)
    )


def validate_sql_risk_reroute_plan(
    plan: WorkerPlan,
    original_task: str,
    *,
    sql_risk_aspects: Iterable[SqlRiskAspect] = (),
) -> None:
    """Require base mapping reads again after a clean SQL-risk reroute.

    Coordinator intentionally discards all evidence when upstream starts a new
    cycle.  A downstream plan that contains only the newly requested delta
    (for example key metadata) would therefore leave upstream without the SQL
    mapping needed to assess the risk.  For explicit technical source→target
    pairs, require one task per pair that both preserves direction and clearly
    asks to read S2T/mapping/rule/transformation data.

    Natural-language arrows and requests without a recognizable directed pair
    stay outside this narrow guard; they cannot be validated reliably without
    another model call.
    """

    selected_aspects = set(sql_risk_aspects)
    if selected_aspects and not (
        selected_aspects & MAPPING_DEPENDENT_SQL_RISK_ASPECTS
    ):
        return

    missing: list[str] = []
    for source_forms, target_forms in _directed_pairs(original_task):
        if any(
            _step_reads_directed_mapping(
                step.task,
                source_forms,
                target_forms,
            )
            for step in plan.steps
        ):
            continue
        missing.append(f"{source_forms[0]} → {target_forms[0]}")

    if missing:
        raise ReroutePlanRequirementError(
            "чистый reroute потерял обязательное повторное чтение "
            "точного directed S2T mapping: "
            + ", ".join(missing)
            + "; добавь самодостаточную task с этой source → target парой"
        )


def validate_sql_risk_plan_requirements(
    plan: WorkerPlan,
    original_task: str,
    *,
    sql_risk_aspects: Iterable[SqlRiskAspect] = (),
) -> None:
    """Validate high-confidence evidence shape for a typed SQL-risk plan.

    For a typed, conditional ``cardinality`` question, one complete exact
    mapping is the whole evidence plan: catalog metadata can only remove a
    stated uncertainty and must not become a second worker.  For
    ``constraint_rejection`` on an explicit directed endpoint pair, both
    endpoint metadata records must be requested by one self-contained worker
    task. Splitting the roles lets a model accidentally pass ``table.field``
    as a table-only batch argument and makes completeness impossible to check
    atomically. The guard validates only literal technical arrow pairs and
    never invents or rewrites an identifier.
    """

    selected_aspects = set(sql_risk_aspects)
    if (
        selected_aspects == {"cardinality"}
        and is_conditional_cardinality_request(original_task)
    ):
        pairs = list(_directed_pairs(original_task))
        mapping_steps = [
            step
            for step in plan.steps
            if any(
                _step_reads_directed_mapping(
                    step.task,
                    source_forms,
                    target_forms,
                )
                for source_forms, target_forms in pairs
            )
        ]
        if (
            len(plan.steps) != 1
            or len(mapping_steps) != 1
            or _COLUMN_METADATA_READ_RE.search(mapping_steps[0].task)
        ):
            raise SqlRiskPlanRequirementError(
                "conditional cardinality требует ровно одну worker task "
                "для чтения полного exact directed S2T mapping; metadata, "
                "проверка ключей и дополнительные steps не нужны для "
                "правдивого условного вывода"
            )

    if "constraint_rejection" not in selected_aspects:
        return

    missing: list[str] = []
    for source_forms, target_forms in _directed_pairs(original_task):
        exact_source = source_forms[0]
        exact_target = target_forms[0]
        has_pair_evidence_step = any(
            _COLUMN_METADATA_READ_RE.search(step.task)
            and _MAPPING_READ_RE.search(step.task)
            and _EVIDENCE_READ_RE.search(step.task)
            and exact_source in step.task.casefold()
            and exact_target in step.task.casefold()
            for step in plan.steps
        )
        if not has_pair_evidence_step:
            missing.append(f"{exact_source} → {exact_target}")

    if missing:
        raise SqlRiskPlanRequirementError(
            "constraint_rejection требует одну самодостаточную worker task "
            "для совместного чтения exact directed S2T mapping и column "
            "metadata обеих точных endpoint-колонок (не отдельные tasks): "
            + ", ".join(missing)
        )


__all__ = [
    "MAPPING_DEPENDENT_SQL_RISK_ASPECTS",
    "ReroutePlanRequirementError",
    "SqlRiskPlanRequirementError",
    "validate_sql_risk_plan_requirements",
    "validate_sql_risk_reroute_plan",
]
