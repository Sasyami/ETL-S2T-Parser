from __future__ import annotations

import pytest

from agents.observability import is_langfuse_configured as _is_langfuse_configured


@pytest.mark.parametrize(
    ("value", "expected"),
    [(None, False), ("0", False), ("1", True)],
)
def test_langfuse_enabled_uses_strict_binary_flag(
    monkeypatch,
    value,
    expected,
):
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "public")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "secret")
    if value is None:
        monkeypatch.delenv("LANGFUSE_ENABLED", raising=False)
    else:
        monkeypatch.setenv("LANGFUSE_ENABLED", value)

    assert _is_langfuse_configured() is expected


def test_langfuse_still_requires_both_credentials(monkeypatch):
    monkeypatch.setenv("LANGFUSE_ENABLED", "1")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "public")
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)

    assert _is_langfuse_configured() is False


@pytest.mark.parametrize("value", ["", "true", "false", " 1 ", "2"])
def test_langfuse_enabled_rejects_non_binary_values(monkeypatch, value):
    monkeypatch.setenv("LANGFUSE_ENABLED", value)

    with pytest.raises(ValueError, match="LANGFUSE_ENABLED must be 0 or 1"):
        _is_langfuse_configured()
