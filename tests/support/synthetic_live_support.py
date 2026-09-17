"""Versioned pytest guard for immutable synthetic live-agent fixtures."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Iterator

import pytest


PLUGIN_API_VERSION = 1
PLUGIN_INTERFACE = (
    "pytest_configure",
    "synthetic_live_database_guard",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pytest_configure(config) -> None:
    """Declare the immutable-fixture contract exposed by this plugin."""
    config.addinivalue_line(
        "markers",
        "synthetic_live_fixture: uses the immutable synthetic live database",
    )


@pytest.fixture(scope="session", autouse=True)
def synthetic_live_database_guard() -> Iterator[None]:
    """Fail the live run if its explicit synthetic SQLite fixture changes."""
    if os.getenv("RUN_LIVE_AGENT_SCENARIOS") != "1":
        yield
        return

    raw_path = os.getenv("LIVE_AGENT_DB_PATH", "")
    if not raw_path:
        pytest.fail("LIVE_AGENT_DB_PATH is required by synthetic live support")
    database = Path(raw_path).expanduser().resolve()
    if not database.is_file():
        pytest.fail(f"synthetic live database is missing: {database}")

    before = _sha256(database)
    yield
    if not database.is_file() or _sha256(database) != before:
        pytest.fail("synthetic live database changed during the test session")
