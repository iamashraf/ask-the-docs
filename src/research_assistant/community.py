"""Runs Neo4j GDS Leiden community detection over the entity graph."""

import logging

import neo4j

logger = logging.getLogger(__name__)

_GRAPH_NAME = "entityGraph"


def run_community_detection(driver: neo4j.Driver) -> None:
    try:
        with driver.session() as session:
            # .consume() forces each query to fully execute now, so an error raised
            # during execution surfaces at this call rather than being deferred and
            # raised by a later session.run() (neo4j's buffer-on-next-call behavior).
            session.run(
                "CALL gds.graph.project($graph_name, 'Entity', 'RELATED_TO')",
                graph_name=_GRAPH_NAME,
            ).consume()
            try:
                session.run(
                    "CALL gds.leiden.write($graph_name, {writeProperty: 'communityId'})",
                    graph_name=_GRAPH_NAME,
                ).consume()
            finally:
                session.run("CALL gds.graph.drop($graph_name)", graph_name=_GRAPH_NAME).consume()
    except (neo4j.exceptions.ServiceUnavailable, neo4j.exceptions.Neo4jError):
        logger.warning(
            "Community detection failed; continuing without updating communities.", exc_info=True
        )
