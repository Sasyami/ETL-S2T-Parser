"""Strict parsing helpers for binary environment flags."""

from __future__ import annotations

import os


def parse_binary_flag(
    name: str,
    value: str | None,
    *,
    default: bool,
) -> bool:
    """Parse an optional environment value using only literal ``0``/``1``."""

    if value is None:
        return default
    if value == "0":
        return False
    if value == "1":
        return True
    raise ValueError(f"{name} must be 0 or 1, got {value!r}")


def read_binary_env_flag(name: str, *, default: bool) -> bool:
    """Read and strictly parse one binary flag from the environment."""

    return parse_binary_flag(name, os.getenv(name), default=default)


__all__ = ["parse_binary_flag", "read_binary_env_flag"]
