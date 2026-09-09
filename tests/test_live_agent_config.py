import pytest

from scripts.live_agent_config import (
    DEFAULT_LIVE_AGENT_HTTP_TIMEOUT,
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
