"""Configuration shared by the opt-in live-agent test harness."""

from __future__ import annotations

import math
import os
from collections.abc import Mapping


DEFAULT_LIVE_AGENT_HTTP_TIMEOUT = 300.0


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
