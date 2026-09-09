"""Conservative, deterministic predicates for narrow operation intents."""

from __future__ import annotations

import re


_IDENTIFIER_PART = (
    r"(?:[A-Za-z_А-Яа-яЁё][A-Za-z0-9_$А-Яа-яЁё]*|"
    r"`[^`\r\n.]+`|\"[^\"\r\n.]+\")"
)
_EXACT_FIELD_REFERENCE = (
    rf"{_IDENTIFIER_PART}(?:\s*\.\s*{_IDENTIFIER_PART}){{1,3}}"
)
_EXACT_FIELD_PAIR_RE = re.compile(
    rf"(?<![\w$`\"]){_EXACT_FIELD_REFERENCE}\s*"
    rf"(?:→|->|=>)\s*{_EXACT_FIELD_REFERENCE}(?![\w$`\"])",
    re.IGNORECASE,
)

_VALUE_CHANGE_PHRASE = (
    r"(?:value[ _-]*changes?|"
    r"изменени(?:е|я|ем|ю|й)\s+"
    r"значени(?:е|я|ем|ю|й))"
)
# Keep exclusivity and the requested aspect in one short clause.  Apart from
# being deterministic, the bound prevents a distant "только" from turning a
# broader request into an eligible deterministic-answer path.
_SAME_CLAUSE_GAP = r"[^.!?\r\n]{0,96}"
_EXCLUSIVE_VALUE_CHANGE_RE = re.compile(
    rf"(?:\b(?:только|only)\b{_SAME_CLAUSE_GAP}{_VALUE_CHANGE_PHRASE}"
    rf"|{_VALUE_CHANGE_PHRASE}{_SAME_CLAUSE_GAP}\b(?:только|only)\b)",
    re.IGNORECASE,
)

_ADDITIONAL_OR_PRESENTATION_INTENT_RE = re.compile(
    r"\b(?:также|покажи|выведи|список|also|show|list|display)\b"
    r"|\bа\s+ещ[её]\b"
    r"|\bполный\s+результат\b",
    re.IGNORECASE,
)


def is_exclusive_value_change_request(task: str) -> bool:
    """Return whether ``task`` is safe for value-change-only handling.

    The predicate is deliberately narrower than general intent detection.  It
    accepts only a literal directed ``table.field → table.field`` pair and an
    explicit ``только``/``only`` qualifier close to the value-change phrase.
    Any explicit secondary or presentation intent makes the result false.
    """

    if not isinstance(task, str) or not task.strip():
        return False
    if _ADDITIONAL_OR_PRESENTATION_INTENT_RE.search(task):
        return False
    return bool(
        _EXACT_FIELD_PAIR_RE.search(task)
        and _EXCLUSIVE_VALUE_CHANGE_RE.search(task)
    )


__all__ = ["is_exclusive_value_change_request"]
