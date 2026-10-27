"""Runs Neo4j GDS Leiden community detection and LLM-based community summarization."""

import logging

import neo4j

from research_assistant.config import Settings
from research_assistant.llm import get_llm_client

logger = logging.getLogger(__name__)

_GRAPH_NAME = "entityGraph"

_READ_COMMUNITIES_QUERY = (
    "MATCH (e:Entity) WHERE e.communityId IS NOT NULL "
    "RETURN e.communityId AS id, collect(e.name) AS names"
)
_WRITE_SUMMARY_QUERY = "MERGE (c:Community {id: $id}) SET c.summary = $summary"

_SUMMARY_PROMPT_TEMPLATE = (
    "These entity names belong to the same community in a knowledge graph: {names}.\n"
    "In 1-2 sentences, summarize the theme connecting them."
)


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


def summarize_communities(driver: neo4j.Driver, settings: Settings) -> None:
    try:
        with driver.session() as session:
            result = session.run(_READ_COMMUNITIES_QUERY)
            # Materialize rows before .consume() and before issuing any writes: the
            # real driver streams records lazily, so writes in the same session
            # while that stream is still open would buffer or discard results.
            communities = [(record["id"], record["names"]) for record in result]
            result.consume()

            client = get_llm_client(settings.generator)
            for community_id, names in communities:
                prompt = _SUMMARY_PROMPT_TEMPLATE.format(names=", ".join(names))
                summary = client.generate(prompt)
                # .consume() forces each query to fully execute now, so an error raised
                # during execution surfaces at this call rather than being deferred and
                # raised by a later session.run() (neo4j's buffer-on-next-call behavior).
                session.run(_WRITE_SUMMARY_QUERY, id=community_id, summary=summary).consume()
    except (neo4j.exceptions.ServiceUnavailable, neo4j.exceptions.Neo4jError):
        logger.warning(
            "Community summarization failed; continuing without updating summaries.", exc_info=True
        )
