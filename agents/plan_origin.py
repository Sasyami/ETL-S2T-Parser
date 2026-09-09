"""Fail-closed origin checks for LLM-produced downstream worker plans."""

from __future__ import annotations

import json
import re
from typing import Any, Iterable, Mapping

from .contracts import WorkerPlan


_ARROW_RE = re.compile(r"(?:→|->|=>)")
_IDENTIFIER_RE = re.compile(
    r"[A-Za-zА-Яа-яЁё0-9_$][A-Za-zА-Яа-яЁё0-9_$.-]*"
)
_FILE_ID_RE = re.compile(r"\bfile_id\s*(?:=|:)\s*(\d+)\b", re.IGNORECASE)
_IDENTIFIER_BOUNDARY_CHARS = "A-Za-zА-Яа-яЁё0-9_$"
_DEPENDENCY_REFERENCE_KEYS = {
    "dependency",
    "dependency_step",
    "from_dependency",
    "from_step",
    "source_step",
    "step",
}
_DEPENDENCY_TEMPLATE_RE = re.compile(
    r"^\s*(?:\{\{\s*|\$)(?:dependency|step|worker|шаг)"
    r"[\s_.:#-]*(\d+)\b",
    re.IGNORECASE,
)
_DEPENDENCY_RESULT_RE = re.compile(
    r"(?:результат\w*|result|output)\s+(?:из\s+|of\s+)?"
    r"(?:dependency|step|worker|шаг\w*)[\s_:#-]*(\d+)\b",
    re.IGNORECASE,
)
_FILTER_IDENTIFIER_KINDS = {
    "column": "identifier",
    "column_name": "identifier",
    "column_names": "identifier",
    "field": "identifier",
    "field_name": "identifier",
    "field_names": "identifier",
    "file_id": "file_id",
    "file_ids": "file_id",
    "file_name": "identifier",
    "filename": "identifier",
    "sheet": "identifier",
    "sheet_name": "identifier",
    "source_column": "identifier",
    "source_field": "identifier",
    "source_table": "identifier",
    "table": "identifier",
    "table_name": "identifier",
    "table_names": "identifier",
    "target_column": "identifier",
    "target_field": "identifier",
    "target_table": "identifier",
}
_GENERIC_FILTER_SENTINELS = {"*", "all", "any", "null"}


class PlanOriginError(ValueError):
    """Raised when a plan materializes a literal with no allowed origin."""


def _adjacent_identifier(value: str, *, before: bool) -> str | None:
    """Return the identifier immediately adjacent to a direction arrow."""

    if before:
        matches = list(_IDENTIFIER_RE.finditer(value.rstrip(" `\"'([{,:;")))
        return matches[-1].group(0) if matches else None
    match = _IDENTIFIER_RE.search(value.lstrip(" `\"')]},:;"))
    return match.group(0) if match else None


def _technical_anchor(identifier: str) -> tuple[tuple[str, ...], ...]:
    """Return conservative alternative literal forms a plan may preserve.

    A dotted endpoint can denote either ``schema.table`` or ``table.field``.
    The planner may correctly split the latter into structured table/field
    values, so either the complete endpoint or all of its structured parts
    must remain present.  Keeping only the table prefix is insufficient for a
    field-level request.  Bare endpoints are checked only when they look
    technical; natural-language arrows are deliberately outside this narrow
    guard.
    """

    clean = identifier.strip(". `\"'()[]{}<>,:;").casefold()
    if not clean or not any(marker in clean for marker in ("_", ".", "$")):
        return ()
    if "." not in clean:
        return ((clean,),)
    prefix, field = clean.rsplit(".", 1)
    alternatives = [(clean,), (prefix, field)]
    parts = tuple(part for part in clean.split(".") if part)
    if len(parts) > 2:
        alternatives.append(parts)
    return tuple(dict.fromkeys(alternatives))


def _directed_endpoint_anchors(
    value: str,
) -> Iterable[tuple[tuple[str, ...], ...]]:
    for arrow in _ARROW_RE.finditer(value):
        left = _adjacent_identifier(value[: arrow.start()], before=True)
        right = _adjacent_identifier(value[arrow.end() :], before=False)
        for identifier in (left, right):
            if identifier:
                anchors = _technical_anchor(identifier)
                if anchors:
                    yield anchors


def _plan_identifiers(value: str) -> set[str]:
    return {
        item.strip(". `\"'()[]{}<>,:;").casefold()
        for item in _IDENTIFIER_RE.findall(value)
        if item.strip(". `\"'()[]{}<>,:;")
    }


def _has_file_id(value: str, file_id: int) -> bool:
    return bool(
        re.search(
            rf"\bfile_id\s*(?:=|:)\s*{re.escape(str(file_id))}\b",
            value,
            re.IGNORECASE,
        )
    )


def _normalized_literal(value: Any) -> str:
    return re.sub(
        r"\s+",
        " ",
        str(value or "").strip(" `\"'()[]{}<>,;"),
    ).casefold()


def _literal_has_origin(value: Any, source: str) -> bool:
    """Match one complete literal or qualified-name component in source text."""

    literal = _normalized_literal(value)
    if not literal:
        return True
    normalized_source = re.sub(r"\s+", " ", str(source or "")).casefold()
    return bool(
        re.search(
            rf"(?<![{_IDENTIFIER_BOUNDARY_CHARS}])"
            rf"{re.escape(literal)}"
            rf"(?![{_IDENTIFIER_BOUNDARY_CHARS}])",
            normalized_source,
        )
    )


def _dependency_reference_steps(value: Any) -> set[int]:
    """Extract only explicit references to outputs of numbered plan steps."""

    if isinstance(value, Mapping):
        references: set[int] = set()
        for key, raw_step in value.items():
            if str(key or "").strip().casefold() not in _DEPENDENCY_REFERENCE_KEYS:
                continue
            if isinstance(raw_step, bool):
                continue
            if isinstance(raw_step, int) or str(raw_step or "").strip().isdigit():
                references.add(int(raw_step))
        return references
    if not isinstance(value, str):
        return set()
    references = {
        int(match)
        for pattern in (_DEPENDENCY_TEMPLATE_RE, _DEPENDENCY_RESULT_RE)
        for match in pattern.findall(value)
    }
    return references


def _validate_structured_literal(
    value: Any,
    *,
    path: str,
    kind: str,
    source: str,
    dependency_steps: set[int],
    dependency_literals: set[str],
    issues: list[str],
) -> None:
    """Validate an entity/filter literal without rejecting generic filter data."""

    references = _dependency_reference_steps(value)
    if references:
        undeclared = sorted(references - dependency_steps)
        if undeclared:
            issues.append(
                f"{path} ссылается на step {undeclared[0]}, но этот "
                "step не указан в dependencies"
            )
        return
    if isinstance(value, Mapping):
        for key, child in value.items():
            _validate_structured_literal(
                child,
                path=f"{path}.{key}",
                kind=kind,
                source=source,
                dependency_steps=dependency_steps,
                dependency_literals=dependency_literals,
                issues=issues,
            )
        return
    if isinstance(value, (list, tuple, set)):
        for index, child in enumerate(value):
            _validate_structured_literal(
                child,
                path=f"{path}[{index}]",
                kind=kind,
                source=source,
                dependency_steps=dependency_steps,
                dependency_literals=dependency_literals,
                issues=issues,
            )
        return
    if value is None:
        return
    if kind == "file_id":
        if isinstance(value, bool) or not str(value).strip().isdigit():
            issues.append(
                f"{path}={value!r} не является допустимым file_id"
            )
            return
        file_id = int(value)
        if (
            not _has_file_id(source, file_id)
            and str(file_id) not in dependency_literals
        ):
            issues.append(
                f"{path}={file_id} отсутствует в original_task/context и "
                "не получен из declared dependency"
            )
        return
    if isinstance(value, bool) or isinstance(value, (int, float)):
        issues.append(
            f"{path}={value!r} не является допустимым "
            "строковым идентификатором"
        )
        return
    literal = _normalized_literal(value)
    if literal in _GENERIC_FILTER_SENTINELS:
        return
    if literal in dependency_literals or _literal_has_origin(value, source):
        return
    issues.append(
        f"{path}={value!r} отсутствует в original_task/context и не "
        "получен из declared dependency"
    )


def _structured_filter_values(
    filters: Mapping[str, Any],
    *,
    prefix: str,
) -> Iterable[tuple[str, str, Any]]:
    """Yield only filter values whose key denotes a physical identifier."""

    for key, value in filters.items():
        clean_key = str(key or "").strip().casefold().replace("-", "_")
        path = f"{prefix}.{key}"
        kind = _FILTER_IDENTIFIER_KINDS.get(clean_key)
        if kind is not None:
            yield path, kind, value
            continue
        if isinstance(value, Mapping):
            yield from _structured_filter_values(value, prefix=path)
        elif isinstance(value, (list, tuple)):
            for index, child in enumerate(value):
                if isinstance(child, Mapping):
                    yield from _structured_filter_values(
                        child,
                        prefix=f"{path}[{index}]",
                    )


def _plain_structured_literals(value: Any) -> Iterable[str]:
    """Return concrete literals a validated dependency can expose downstream."""

    if _dependency_reference_steps(value):
        return
    if isinstance(value, Mapping):
        for child in value.values():
            yield from _plain_structured_literals(child)
        return
    if isinstance(value, (list, tuple, set)):
        for child in value:
            yield from _plain_structured_literals(child)
        return
    if value is None or isinstance(value, bool):
        return
    literal = _normalized_literal(value)
    if literal:
        yield literal


def validate_worker_plan_origin(
    plan: WorkerPlan,
    original_task: str,
    *,
    context: str = "",
) -> None:
    """Reject high-confidence identifier invention before any worker runs.

    Free-form Russian worker prose remains outside this narrow guard.  The
    structured boundary is fail-closed for file scope, entity table/field and
    identifier-bearing filter values.  Generic data filters stay unconstrained.
    A future value may be represented only as an explicit reference to a
    declared dependency; it cannot be materialized ahead of that evidence.
    """

    source = "\n".join(
        part for part in (str(original_task or ""), str(context or "")) if part
    )
    source_folded = source.casefold()
    serialized = json.dumps(
        plan.model_dump(mode="json", exclude_none=True),
        ensure_ascii=False,
        sort_keys=True,
    )
    serialized_folded = serialized.casefold()
    plan_identifiers = _plan_identifiers(serialized_folded)
    issues: list[str] = []
    produced_literals: dict[int, set[str]] = {}

    for raw_file_id in dict.fromkeys(_FILE_ID_RE.findall(serialized)):
        file_id = int(raw_file_id)
        if not _has_file_id(source, file_id):
            issues.append(
                f"file_id={file_id} отсутствует в original_task/context"
            )

    for step_number, step in enumerate(plan.steps, start=1):
        dependency_steps = set(step.dependencies or [])
        dependency_literals = {
            literal
            for dependency in dependency_steps
            for literal in produced_literals.get(dependency, set())
        }
        entity = step.entity
        if entity is not None:
            for field_name in ("table", "field"):
                literal = getattr(entity, field_name)
                if literal:
                    _validate_structured_literal(
                        literal,
                        path=f"step {step_number} entity.{field_name}",
                        kind="identifier",
                        source=source,
                        dependency_steps=dependency_steps,
                        dependency_literals=dependency_literals,
                        issues=issues,
                    )

        scope = step.scope
        if scope is not None:
            if (
                scope.file_id is not None
                and not _has_file_id(source, scope.file_id)
            ):
                issue = (
                    f"file_id={scope.file_id} отсутствует в "
                    "original_task/context"
                )
                if issue not in issues:
                    issues.append(issue)
            for field_name in ("filename", "sheet_name"):
                literal = getattr(scope, field_name)
                if literal and literal.casefold() not in source_folded:
                    issues.append(
                        f"step {step_number} scope.{field_name}={literal!r} "
                        "отсутствует в original_task/context"
                    )
            for path, kind, value in _structured_filter_values(
                scope.filters,
                prefix=f"step {step_number} scope.filters",
            ):
                _validate_structured_literal(
                    value,
                    path=path,
                    kind=kind,
                    source=source,
                    dependency_steps=dependency_steps,
                    dependency_literals=dependency_literals,
                    issues=issues,
                )

        step_literals: set[str] = set()
        if entity is not None:
            for value in (entity.table, entity.field):
                step_literals.update(_plain_structured_literals(value))
        if scope is not None:
            for _, _, value in _structured_filter_values(
                scope.filters,
                prefix=f"step {step_number} scope.filters",
            ):
                step_literals.update(_plain_structured_literals(value))
        produced_literals[step_number] = step_literals

    # Stable context can contain reusable examples unrelated to this request.
    # Reference resolutions, when needed, are already appended to task by the
    # supervisor, so only the current task defines mandatory pair endpoints.
    for accepted_alternatives in _directed_endpoint_anchors(original_task):
        if not any(
            all(form in plan_identifiers for form in alternative)
            for alternative in accepted_alternatives
        ):
            issues.append(
                "план потерял literal endpoint "
                + repr(accepted_alternatives[0][0])
                + " из направленной пары original_task"
            )

    if issues:
        raise PlanOriginError("; ".join(issues))


__all__ = ["PlanOriginError", "validate_worker_plan_origin"]
