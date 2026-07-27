"""Neo4j driver lifecycle without graph reads or writes."""

from typing import Optional

from neo4j import Driver, GraphDatabase

from .config import Neo4jSettings, load_neo4j_settings


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
    )


def verify_neo4j_connectivity(driver: Driver) -> None:
    """Verify server connectivity without executing a Cypher query."""
    driver.verify_connectivity()


def close_neo4j_driver(driver: Driver) -> None:
    """Close the Neo4j driver and its connection pool."""
    driver.close()
