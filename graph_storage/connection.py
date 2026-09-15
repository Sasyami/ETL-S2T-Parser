"""Neo4j driver lifecycle without graph reads or writes."""

from typing import Optional

from neo4j import Driver, GraphDatabase

from .config import Neo4jSettings, load_neo4j_settings

NEO4J_CONNECTION_TIMEOUT_SECONDS = 3.0
NEO4J_CONNECTION_ACQUISITION_TIMEOUT_SECONDS = 3.0
NEO4J_MAX_TRANSACTION_RETRY_TIME_SECONDS = 0.0


def create_neo4j_driver(
    settings: Optional[Neo4jSettings] = None,
) -> Driver:
    """Create a Neo4j driver without opening a session or executing Cypher."""
    resolved_settings = settings or load_neo4j_settings()
    return GraphDatabase.driver(
        resolved_settings.uri,
        auth=(
            resolved_settings.username,
            resolved_settings.password,
        ),
        connection_timeout=NEO4J_CONNECTION_TIMEOUT_SECONDS,
        connection_acquisition_timeout=(
            NEO4J_CONNECTION_ACQUISITION_TIMEOUT_SECONDS
        ),
        max_transaction_retry_time=NEO4J_MAX_TRANSACTION_RETRY_TIME_SECONDS,
    )


def verify_neo4j_connectivity(driver: Driver) -> None:
    """Verify server connectivity without executing a Cypher query."""
    driver.verify_connectivity()


def close_neo4j_driver(driver: Driver) -> None:
    """Close the Neo4j driver and its connection pool."""
    driver.close()
