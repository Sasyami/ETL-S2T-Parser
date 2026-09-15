"""Strict binary configuration for experimental feature flags.

Experimental toggles accept only the literal strings ``0`` and ``1`` through
the same strict parser as the other project-owned boolean settings. A typo
must fail visibly instead of selecting an experiment arm.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Mapping

from .env_flags import parse_binary_flag, read_binary_env_flag


WORKER_CAPABILITY_REROUTE_EXPERIMENT_ENV = (
    "WORKER_CAPABILITY_REROUTE_EXPERIMENT"
)
WORKER_SPLIT_TOOL_CALL_EXPERIMENT_ENV = "WORKER_SPLIT_TOOL_CALL_EXPERIMENT"
OPERATION_SQL_RISK_ASPECTS_EXPERIMENT_ENV = (
    "OPERATION_SQL_RISK_ASPECTS_EXPERIMENT"
)
OPERATION_SQL_RISK_SCOPE_EVIDENCE_EXPERIMENT_ENV = (
    "OPERATION_SQL_RISK_SCOPE_EVIDENCE_EXPERIMENT"
)
S2T_NARROW_TOOLS_EXPERIMENT_ENV = "S2T_NARROW_TOOLS_EXPERIMENT"


BINARY_EXPERIMENT_DEFAULTS: Mapping[str, bool] = MappingProxyType(
    {
        WORKER_CAPABILITY_REROUTE_EXPERIMENT_ENV: False,
        WORKER_SPLIT_TOOL_CALL_EXPERIMENT_ENV: False,
        OPERATION_SQL_RISK_ASPECTS_EXPERIMENT_ENV: False,
        OPERATION_SQL_RISK_SCOPE_EVIDENCE_EXPERIMENT_ENV: False,
        S2T_NARROW_TOOLS_EXPERIMENT_ENV: False,
    }
)


def parse_binary_experiment_flag(
    name: str,
    value: str | None,
) -> bool:
    """Parse one experiment flag using the strict ``0``/``1`` contract."""

    try:
        default = BINARY_EXPERIMENT_DEFAULTS[name]
    except KeyError as exc:
        raise ValueError(f"Unregistered binary experiment flag: {name!r}") from exc
    return parse_binary_flag(name, value, default=default)


def experiment_flag_enabled(name: str) -> bool:
    """Read and parse one binary experiment flag from the environment."""

    try:
        default = BINARY_EXPERIMENT_DEFAULTS[name]
    except KeyError as exc:
        raise ValueError(f"Unregistered binary experiment flag: {name!r}") from exc
    return read_binary_env_flag(name, default=default)


__all__ = [
    "BINARY_EXPERIMENT_DEFAULTS",
    "OPERATION_SQL_RISK_ASPECTS_EXPERIMENT_ENV",
    "OPERATION_SQL_RISK_SCOPE_EVIDENCE_EXPERIMENT_ENV",
    "S2T_NARROW_TOOLS_EXPERIMENT_ENV",
    "WORKER_CAPABILITY_REROUTE_EXPERIMENT_ENV",
    "WORKER_SPLIT_TOOL_CALL_EXPERIMENT_ENV",
    "experiment_flag_enabled",
    "parse_binary_experiment_flag",
]
