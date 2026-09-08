"""Fail-closed token budget guard for GigaChat Ultra experiments."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Callable, Iterable, Mapping, Protocol


DEFAULT_ULTRA_TOKEN_FLOOR = 15_000_000
DEFAULT_ULTRA_RESERVE_PER_SCENARIO = 250_000
DEFAULT_GIGACHAT_JUDGE_MODEL = "GigaChat-2-Pro"
# ``judge_agent_response`` can run an identifier audit and the final verdict;
# each structured call is configured for at most three attempts.
ULTRA_JUDGE_MAX_CALLS_PER_EXCHANGE = 2 * 3


class BalanceEntryLike(Protocol):
    usage: str
    value: float


class UltraBudgetError(RuntimeError):
    """Base class for an intentionally blocked Ultra experiment."""


class UltraBalanceUnavailable(UltraBudgetError):
    """Raised when the provider cannot prove a usable model balance."""


class UltraBudgetTooLow(UltraBudgetError):
    """Raised when the run could cross the configured remaining-token floor."""


@dataclass(frozen=True)
class UltraBudgetDecision:
    model: str
    usage: str
    remaining_tokens: int
    floor_tokens: int
    reserved_tokens: int

    @property
    def projected_remaining_tokens(self) -> int:
        return self.remaining_tokens - self.reserved_tokens


@dataclass(frozen=True)
class UltraBudgetReservation:
    """Conservative token reservation for one configured Ultra balance."""

    model: str
    reserved_tokens: int


def is_ultra_model(model: str) -> bool:
    """Return whether a configured model explicitly targets the Ultra tier."""
    return "ultra" in str(model or "").casefold()


def configured_gigachat_judge_model(
    environ: Mapping[str, str] | None = None,
) -> str:
    """Resolve the judge model with the same precedence as ``llm_factory``."""

    values = os.environ if environ is None else environ
    return (
        str(values.get("GIGACHAT_JUDGE_MODEL") or "").strip()
        or str(values.get("LLM_JUDGE_MODEL") or "").strip()
        or DEFAULT_GIGACHAT_JUDGE_MODEL
    )


def ultra_budget_reservations(
    *,
    chat_model: str,
    exchange_count: int,
    reserve_per_exchange: int = DEFAULT_ULTRA_RESERVE_PER_SCENARIO,
    judge_enabled: bool = False,
    judge_model: str | None = None,
) -> tuple[UltraBudgetReservation, ...]:
    """Build fail-closed reserves for every Ultra model used by a live run.

    The chat-model reserve is charged once per real HTTP ``/chat`` exchange.
    An Ultra judge is charged for the conservative worst case of two structured
    stages with three attempts each.  Identical chat/judge models share one
    combined reservation so the same balance cannot be budgeted twice.
    """

    exchanges = max(0, int(exchange_count))
    per_exchange = max(
        DEFAULT_ULTRA_RESERVE_PER_SCENARIO,
        int(reserve_per_exchange),
    )
    requested: list[tuple[str, int]] = []
    clean_chat_model = str(chat_model or "").strip()
    if is_ultra_model(clean_chat_model):
        requested.append((clean_chat_model, exchanges * per_exchange))
    if judge_enabled:
        clean_judge_model = str(
            judge_model or configured_gigachat_judge_model()
        ).strip()
        if is_ultra_model(clean_judge_model):
            requested.append(
                (
                    clean_judge_model,
                    exchanges
                    * per_exchange
                    * ULTRA_JUDGE_MAX_CALLS_PER_EXCHANGE,
                )
            )

    combined: dict[str, UltraBudgetReservation] = {}
    for model, reserved_tokens in requested:
        identity = _identity(model)
        previous = combined.get(identity)
        combined[identity] = UltraBudgetReservation(
            model=previous.model if previous is not None else model,
            reserved_tokens=(
                reserved_tokens
                + (previous.reserved_tokens if previous is not None else 0)
            ),
        )
    return tuple(combined.values())


def _identity(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").casefold())


def _select_balance_entry(
    model: str,
    entries: Iterable[BalanceEntryLike],
) -> BalanceEntryLike:
    values = list(entries)
    model_identity = _identity(model)
    exact = [entry for entry in values if _identity(entry.usage) == model_identity]
    if len(exact) == 1:
        return exact[0]
    ultra = [entry for entry in values if "ultra" in _identity(entry.usage)]
    if is_ultra_model(model) and len(ultra) == 1:
        return ultra[0]
    available = ", ".join(sorted(str(item.usage) for item in values)) or "нет"
    raise UltraBalanceUnavailable(
        f"Баланс модели {model!r} нельзя определить однозначно; entries: {available}."
    )


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return value.strip().casefold() in {"1", "true", "yes", "on"}


def fetch_gigachat_balance() -> list[BalanceEntryLike]:
    """Read prepaid balances through the official SDK without model calls."""
    from dotenv import load_dotenv
    from gigachat import GigaChat

    load_dotenv()
    credentials = (
        os.getenv("GIGACHAT_API_KEY")
        or os.getenv("GIGACHAT_CREDENTIALS")
        or os.getenv("GIGACHAT_EMBEDDINGS_CREDENTIALS")
    )
    if not credentials:
        raise UltraBalanceUnavailable("Не настроены GigaChat credentials.")
    configured_url = os.getenv("GIGACHAT_API_URL", "https://api.giga.chat/v1")
    balance_urls = list(
        dict.fromkeys(
            value
            for value in (
                os.getenv("GIGACHAT_BALANCE_API_URL", "").strip(),
                configured_url,
                # The provider keeps this official endpoint for existing
                # connections; it is useful when the new edge has a transient
                # TLS failure. Both calls are read-only and consume no model
                # tokens.
                "https://gigachat.devices.sberbank.ru/api/v1",
            )
            if value
        )
    )
    last_error: Exception | None = None
    for base_url in balance_urls:
        try:
            with GigaChat(
                credentials=credentials,
                base_url=base_url,
                scope=os.getenv("GIGACHAT_SCOPE", "GIGACHAT_API_PERS"),
                verify_ssl_certs=_env_bool("GIGACHAT_VERIFY_SSL", False),
            ) as client:
                return list(client.get_balance().balance)
        except Exception as exc:  # try only the documented official fallback
            last_error = exc
    assert last_error is not None
    raise UltraBalanceUnavailable(
        "GigaChat не подтвердил остаток токенов: "
        f"{type(last_error).__name__}: {last_error}"
    ) from last_error


def guard_ultra_budget(
    *,
    model: str,
    floor_tokens: int = DEFAULT_ULTRA_TOKEN_FLOOR,
    reserved_tokens: int = 0,
    balance_loader: Callable[[], list[BalanceEntryLike]] = fetch_gigachat_balance,
) -> UltraBudgetDecision | None:
    """Allow non-Ultra models; require proven headroom for every Ultra run."""
    if not is_ultra_model(model):
        return None
    # The user-authorized safety boundary is a hard minimum.  CLI/env values
    # may raise it for a run, but must never weaken the 15M guarantee.
    floor = max(DEFAULT_ULTRA_TOKEN_FLOOR, int(floor_tokens))
    reserve = max(0, int(reserved_tokens))
    entry = _select_balance_entry(model, balance_loader())
    remaining = int(entry.value)
    if remaining - reserve < floor:
        raise UltraBudgetTooLow(
            "Ultra experiment blocked: "
            f"remaining={remaining}, reserved={reserve}, floor={floor}."
        )
    return UltraBudgetDecision(
        model=model,
        usage=str(entry.usage),
        remaining_tokens=remaining,
        floor_tokens=floor,
        reserved_tokens=reserve,
    )


__all__ = [
    "DEFAULT_GIGACHAT_JUDGE_MODEL",
    "DEFAULT_ULTRA_RESERVE_PER_SCENARIO",
    "DEFAULT_ULTRA_TOKEN_FLOOR",
    "ULTRA_JUDGE_MAX_CALLS_PER_EXCHANGE",
    "UltraBalanceUnavailable",
    "UltraBudgetDecision",
    "UltraBudgetError",
    "UltraBudgetReservation",
    "UltraBudgetTooLow",
    "configured_gigachat_judge_model",
    "fetch_gigachat_balance",
    "guard_ultra_budget",
    "is_ultra_model",
    "ultra_budget_reservations",
]
