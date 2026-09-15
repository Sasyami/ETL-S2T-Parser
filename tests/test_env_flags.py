"""Shared strict contract for project-owned boolean environment flags."""

from __future__ import annotations

import pytest

from agents.env_flags import parse_binary_flag, read_binary_env_flag


def test_binary_flag_uses_declared_default_only_when_unset():
    assert parse_binary_flag("FEATURE", None, default=False) is False
    assert parse_binary_flag("FEATURE", None, default=True) is True


@pytest.mark.parametrize(("value", "expected"), [("0", False), ("1", True)])
def test_binary_flag_accepts_exact_zero_or_one(value, expected):
    assert parse_binary_flag("FEATURE", value, default=not expected) is expected


@pytest.mark.parametrize(
    "value",
    ["", " ", " 1 ", "true", "false", "yes", "no", "on", "off", "2"],
)
def test_binary_flag_rejects_aliases_whitespace_and_unknown_values(value):
    with pytest.raises(ValueError, match="FEATURE must be 0 or 1"):
        parse_binary_flag("FEATURE", value, default=False)


def test_binary_env_reader_uses_the_same_contract(monkeypatch):
    monkeypatch.setenv("FEATURE", "1")
    assert read_binary_env_flag("FEATURE", default=False) is True

    monkeypatch.setenv("FEATURE", "true")
    with pytest.raises(ValueError, match="FEATURE must be 0 or 1"):
        read_binary_env_flag("FEATURE", default=False)
