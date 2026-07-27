"""Neo4j configuration loaded independently from the application runtime."""

import os
from dataclasses import dataclass

from dotenv import load_dotenv


@dataclass(frozen=True)
class Neo4jSettings:
    """Connection settings for an external Neo4j database."""

    uri: str
    username: str
    password: str
    database: str


def load_neo4j_settings() -> Neo4jSettings:
    """Load Neo4j connection settings from the project environment."""
    load_dotenv()
    return Neo4jSettings(
        uri=os.environ["NEO4J_URI"],
        username=os.environ["NEO4J_USERNAME"],
        password=os.environ["NEO4J_PASSWORD"],
        database=os.environ["NEO4J_DATABASE"],
    )
