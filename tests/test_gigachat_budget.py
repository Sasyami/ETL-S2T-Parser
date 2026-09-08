from types import SimpleNamespace

import pytest

from scripts.gigachat_budget import (
    ULTRA_JUDGE_MAX_CALLS_PER_EXCHANGE,
    UltraBalanceUnavailable,
    UltraBudgetTooLow,
    configured_gigachat_judge_model,
    guard_ultra_budget,
    is_ultra_model,
    ultra_budget_reservations,
)


def _entries(*pairs):
    return [SimpleNamespace(usage=name, value=value) for name, value in pairs]


def test_non_ultra_model_does_not_request_balance():
    called = False

    def loader():
        nonlocal called
        called = True
        return []

    assert guard_ultra_budget(model="GigaChat-2-Pro", balance_loader=loader) is None
    assert called is False


def test_ultra_budget_reserves_run_cost_above_floor():
    decision = guard_ultra_budget(
        model="GigaChat-3-Ultra",
        floor_tokens=15_000_000,
        reserved_tokens=1_000_000,
        balance_loader=lambda: _entries(
            ("GigaChat-3-Ultra", 16_500_000),
            ("GigaChat-2-Pro", 9_000_000),
        ),
    )

    assert decision is not None
    assert decision.projected_remaining_tokens == 15_500_000
    assert is_ultra_model(decision.model)


def test_ultra_budget_blocks_run_that_can_cross_floor():
    with pytest.raises(UltraBudgetTooLow, match="remaining=15500000"):
        guard_ultra_budget(
            model="GigaChat-3-Ultra",
            floor_tokens=15_000_000,
            reserved_tokens=600_000,
            balance_loader=lambda: _entries(("GigaChat-3-Ultra", 15_500_000)),
        )


def test_ultra_budget_floor_cannot_be_lowered_below_fifteen_million():
    with pytest.raises(UltraBudgetTooLow, match="floor=15000000"):
        guard_ultra_budget(
            model="GigaChat-3-Ultra",
            floor_tokens=1,
            reserved_tokens=0,
            balance_loader=lambda: _entries(
                ("GigaChat-3-Ultra", 14_999_999)
            ),
        )


def test_ultra_per_exchange_reserve_cannot_be_lowered():
    reservations = ultra_budget_reservations(
        chat_model="GigaChat-3-Ultra",
        exchange_count=2,
        reserve_per_exchange=0,
    )

    assert len(reservations) == 1
    assert reservations[0].reserved_tokens == 2 * 250_000


def test_ultra_budget_fails_closed_for_missing_or_ambiguous_entry():
    with pytest.raises(UltraBalanceUnavailable, match="нельзя определить"):
        guard_ultra_budget(
            model="GigaChat-3-Ultra",
            balance_loader=lambda: _entries(
                ("Ultra input", 20_000_000),
                ("Ultra output", 20_000_000),
            ),
        )


def test_gigachat_judge_model_uses_factory_precedence():
    assert configured_gigachat_judge_model(
        {
            "GIGACHAT_JUDGE_MODEL": "GigaChat-3-Ultra",
            "LLM_JUDGE_MODEL": "fallback-ultra",
        }
    ) == "GigaChat-3-Ultra"
    assert configured_gigachat_judge_model(
        {"LLM_JUDGE_MODEL": "fallback-ultra"}
    ) == "fallback-ultra"
    assert configured_gigachat_judge_model({}) == "GigaChat-2-Pro"


def test_ultra_judge_reserve_accounts_for_all_structured_retries():
    reservations = ultra_budget_reservations(
        chat_model="GigaChat-2-Pro",
        exchange_count=2,
        reserve_per_exchange=250_000,
        judge_enabled=True,
        judge_model="GigaChat-3-Ultra",
    )

    assert ULTRA_JUDGE_MAX_CALLS_PER_EXCHANGE == 6
    assert [(item.model, item.reserved_tokens) for item in reservations] == [
        ("GigaChat-3-Ultra", 2 * 6 * 250_000)
    ]


def test_same_ultra_chat_and_judge_share_combined_reserve():
    reservations = ultra_budget_reservations(
        chat_model="GigaChat-3-Ultra",
        exchange_count=3,
        reserve_per_exchange=250_000,
        judge_enabled=True,
        judge_model="gigachat 3 ultra",
    )

    assert [(item.model, item.reserved_tokens) for item in reservations] == [
        ("GigaChat-3-Ultra", 3 * 7 * 250_000)
    ]
