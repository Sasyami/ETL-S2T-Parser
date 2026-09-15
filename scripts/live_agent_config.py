"""Configuration shared by the opt-in live-agent test harness."""

from __future__ import annotations

import math
import os
from collections.abc import Mapping

from agents.env_flags import parse_binary_flag


DEFAULT_LIVE_AGENT_HTTP_TIMEOUT = 300.0
RUN_LIVE_AGENT_SCENARIOS_ENV = "RUN_LIVE_AGENT_SCENARIOS"
LIVE_AGENT_LLM_JUDGE_ENV = "LIVE_AGENT_LLM_JUDGE"


def read_live_agent_scenarios_enabled(
    environ: Mapping[str, str] | None = None,
) -> bool:
    """Return whether real live-agent scenarios are explicitly enabled."""

    source = os.environ if environ is None else environ
    return parse_binary_flag(
        RUN_LIVE_AGENT_SCENARIOS_ENV,
        source.get(RUN_LIVE_AGENT_SCENARIOS_ENV),
        default=False,
    )


def read_live_agent_llm_judge_enabled(
    environ: Mapping[str, str] | None = None,
) -> bool:
    """Return whether the semantic live-agent judge is explicitly enabled."""

    source = os.environ if environ is None else environ
    return parse_binary_flag(
        LIVE_AGENT_LLM_JUDGE_ENV,
        source.get(LIVE_AGENT_LLM_JUDGE_ENV),
        default=False,
    )


def read_live_agent_http_timeout(
    environ: Mapping[str, str] | None = None,
) -> float:
    """Return the positive finite HTTP timeout configured for live scenarios."""
    source = os.environ if environ is None else environ
    raw = source.get("LIVE_AGENT_HTTP_TIMEOUT", "").strip()
    if not raw:
        return DEFAULT_LIVE_AGENT_HTTP_TIMEOUT

    try:
        timeout = float(raw)
    except ValueError as exc:
        raise ValueError(
            "LIVE_AGENT_HTTP_TIMEOUT must be a finite positive number of "
            f"seconds, got {raw!r}"
        ) from exc

    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError(
            "LIVE_AGENT_HTTP_TIMEOUT must be a finite positive number of "
            f"seconds, got {raw!r}"
        )
    return timeout
