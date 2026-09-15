"""Strict contracts shared by binary experiment environment flags."""

from __future__ import annotations

import pytest

from agents.experiment_flags import (
    BINARY_EXPERIMENT_DEFAULTS,
    WORKER_CAPABILITY_REROUTE_EXPERIMENT_ENV,
    experiment_flag_enabled,
    parse_binary_experiment_flag,
)


def test_worker_capability_reroute_experiment_is_opt_in():
    assert (
        BINARY_EXPERIMENT_DEFAULTS[
            WORKER_CAPABILITY_REROUTE_EXPERIMENT_ENV
        ]
        is False
    )


@pytest.mark.parametrize(
    ("name", "default"),
    tuple(BINARY_EXPERIMENT_DEFAULTS.items()),
)
def test_binary_experiment_flag_uses_registered_default_when_unset(
    name,
    default,
):
    assert parse_binary_experiment_flag(
        name,
        None,
    ) is default


@pytest.mark.parametrize(("value", "expected"), [("0", False), ("1", True)])
def test_binary_experiment_flag_accepts_only_zero_and_one(
    value,
    expected,
):
    name = next(iter(BINARY_EXPERIMENT_DEFAULTS))
    assert parse_binary_experiment_flag(
        name,
        value,
    ) is expected


@pytest.mark.parametrize(
    "value",
    [
        "",
        " ",
        " 1 ",
        "true",
        "false",
        "yes",
        "no",
        "on",
        "off",
        "enabled",
        "disabled",
        "default",
        "current",
        "operation_scope",
        "typed_plan",
        "2",
    ],
)
def test_binary_experiment_flag_rejects_aliases_and_unknown_values(value):
    name = next(iter(BINARY_EXPERIMENT_DEFAULTS))
    with pytest.raises(ValueError, match="must be 0 or 1"):
        parse_binary_experiment_flag(
            name,
            value,
        )


def test_binary_experiment_flag_rejects_unregistered_names():
    with pytest.raises(ValueError, match="Unregistered binary experiment flag"):
        parse_binary_experiment_flag("UNKNOWN_EXPERIMENT", "1")


@pytest.mark.parametrize(
    ("name", "default"),
    tuple(BINARY_EXPERIMENT_DEFAULTS.items()),
)
def test_registered_binary_experiment_flags_share_one_parser(
    monkeypatch,
    name,
    default,
):
    monkeypatch.delenv(name, raising=False)
    assert experiment_flag_enabled(name) is default

    monkeypatch.setenv(name, "1")
    assert experiment_flag_enabled(name) is True

    monkeypatch.setenv(name, "0")
    assert experiment_flag_enabled(name) is False

    monkeypatch.setenv(name, "true")
    with pytest.raises(ValueError, match=rf"{name} must be 0 or 1"):
        experiment_flag_enabled(name)
