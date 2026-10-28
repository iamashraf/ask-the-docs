from functools import lru_cache

import neo4j

from ask_the_docs.config import Settings


@lru_cache
def _get_graph_client_cached(uri: str, user: str, password: str) -> neo4j.Driver:
    """Keyed on primitives (Settings isn't hashable); a driver owns a connection pool, so build once per config."""
    return neo4j.GraphDatabase.driver(uri, auth=(user, password))


def get_graph_client(settings: Settings) -> neo4j.Driver:
    """Factory: returns the shared Neo4j driver for the configured server."""
    return _get_graph_client_cached(
        settings.neo4j_uri,
        settings.neo4j_user,
        settings.neo4j_password.get_secret_value(),
    )
