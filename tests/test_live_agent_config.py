import pytest

from scripts.live_agent_config import (
    DEFAULT_LIVE_AGENT_HTTP_TIMEOUT,
    LIVE_AGENT_LLM_JUDGE_ENV,
    RUN_LIVE_AGENT_SCENARIOS_ENV,
    read_live_agent_llm_judge_enabled,
    read_live_agent_scenarios_enabled,
    read_live_agent_http_timeout,
)


@pytest.mark.parametrize(
    ("environ", "expected"),
    [
        ({}, DEFAULT_LIVE_AGENT_HTTP_TIMEOUT),
        ({"LIVE_AGENT_HTTP_TIMEOUT": ""}, DEFAULT_LIVE_AGENT_HTTP_TIMEOUT),
        ({"LIVE_AGENT_HTTP_TIMEOUT": " 600 "}, 600.0),
        ({"LIVE_AGENT_HTTP_TIMEOUT": "0.25"}, 0.25),
    ],
)
def test_read_live_agent_http_timeout_accepts_default_and_positive_values(
    environ,
    expected,
):
    assert read_live_agent_http_timeout(environ) == expected


@pytest.mark.parametrize(
    "value",
    ["invalid", "0", "-1", "nan", "inf", "-inf"],
)
def test_read_live_agent_http_timeout_rejects_invalid_values(value):
    with pytest.raises(
        ValueError,
        match=(
            r"LIVE_AGENT_HTTP_TIMEOUT must be a finite positive number of "
            r"seconds"
        ),
    ):
        read_live_agent_http_timeout({"LIVE_AGENT_HTTP_TIMEOUT": value})


@pytest.mark.parametrize(
    "reader",
    [
        read_live_agent_scenarios_enabled,
        read_live_agent_llm_judge_enabled,
    ],
)
def test_live_agent_boolean_flags_are_disabled_when_unset(reader):
    assert reader({}) is False


@pytest.mark.parametrize(
    ("reader", "name"),
    [
        (read_live_agent_scenarios_enabled, RUN_LIVE_AGENT_SCENARIOS_ENV),
        (read_live_agent_llm_judge_enabled, LIVE_AGENT_LLM_JUDGE_ENV),
    ],
)
@pytest.mark.parametrize(("value", "expected"), [("0", False), ("1", True)])
def test_live_agent_boolean_flags_accept_only_binary_values(
    reader,
    name,
    value,
    expected,
):
    assert reader({name: value}) is expected


@pytest.mark.parametrize(
    ("reader", "name"),
    [
        (read_live_agent_scenarios_enabled, RUN_LIVE_AGENT_SCENARIOS_ENV),
        (read_live_agent_llm_judge_enabled, LIVE_AGENT_LLM_JUDGE_ENV),
    ],
)
@pytest.mark.parametrize(
    "value",
    ["", " ", " 1 ", "true", "false", "yes", "no", "on", "off", "2"],
)
def test_live_agent_boolean_flags_reject_aliases_and_invalid_values(
    reader,
    name,
    value,
):
    with pytest.raises(ValueError, match=rf"{name} must be 0 or 1"):
        reader({name: value})
