"""Read-only Neo4j query execution."""

from typing import Any, Dict, List, Mapping, Optional

from neo4j import READ_ACCESS

from .config import load_neo4j_settings
from .connection import close_neo4j_driver, create_neo4j_driver


def execute_neo4j_read(
    query: str,
    parameters: Optional[Mapping[str, Any]] = None,
    row_limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Execute a fixed read query in a Neo4j read-access transaction."""
    settings = load_neo4j_settings()
    driver = create_neo4j_driver(settings)
    try:
        with driver.session(
            database=settings.database,
            default_access_mode=READ_ACCESS,
        ) as session:
            return session.execute_read(
                lambda tx: _collect_rows(
                    tx.run(query, **dict(parameters or {})),
                    row_limit,
                )
            )
    finally:
        close_neo4j_driver(driver)


def _collect_rows(result: Any, row_limit: Optional[int]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for record in result:
        rows.append(record.data())
        if row_limit is not None and len(rows) >= row_limit:
            break
    return rows
