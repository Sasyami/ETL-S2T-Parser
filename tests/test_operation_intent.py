"""Tests for conservative deterministic-operation intent predicates."""

from __future__ import annotations

import pytest

from agents.operation_intent import is_exclusive_value_change_request


def test_live_value_change_wording_is_exclusive() -> None:
    task = (
        "Оцени только SQL-аспект value changes: может ли "
        "сохранённая S2T-трансформация src_np.id → tgt_np.id "
        "изменить значение? Остальные SQL-риски не анализируй."
    )

    assert is_exclusive_value_change_request(task) is True


@pytest.mark.parametrize(
    "task",
    [
        "Only assess value changes for source_table.id -> target_table.id.",
        "Assess source_table.id => target_table.id value changes only.",
        "Оцени только изменение значения `src_table`.`id` → `tgt_table`.`id`.",
        "Только риск изменением значения: schema.src.id → schema.tgt.id.",
    ],
)
def test_accepts_explicit_exclusive_exact_field_pair(task: str) -> None:
    assert is_exclusive_value_change_request(task) is True


@pytest.mark.parametrize(
    "task",
    [
        "Оцени только SQL-аспект value changes.",
        "Оцени value changes: src.id → tgt.id.",
        "Оцени только value changes: src → tgt.",
        "Оцени только row filtering: src.id → tgt.id.",
        "Только это. Оцени value changes: src.id → tgt.id.",
        "Только " + ("x" * 97) + " value changes: src.id → tgt.id.",
        "",
    ],
)
def test_rejects_missing_required_high_confidence_signals(task: str) -> None:
    assert is_exclusive_value_change_request(task) is False


@pytest.mark.parametrize(
    "additional_intent",
    [
        "также оцени row filtering",
        "а ещё проверь агрегацию",
        "покажи evidence",
        "выведи строки",
        "список правил",
        "полный результат",
        "also assess filtering",
        "show the evidence",
        "list all rules",
        "display the result",
    ],
)
def test_rejects_additional_or_presentation_intents(
    additional_intent: str,
) -> None:
    task = (
        "Оцени только SQL-аспект value changes для "
        f"src_table.id → tgt_table.id; {additional_intent}."
    )

    assert is_exclusive_value_change_request(task) is False


@pytest.mark.parametrize("task", [None, 42, object()])
def test_rejects_non_string_input(task: object) -> None:
    assert is_exclusive_value_change_request(task) is False  # type: ignore[arg-type]
