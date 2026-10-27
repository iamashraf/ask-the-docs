"""Local-query retrieval: traverse the graph from named entities and resolve matched chunk_refs to Documents.
Global-query retrieval: map step over community summaries, scored for relevance to the query."""

import json
import logging
import re

import neo4j
from langchain_core.documents import Document

from research_assistant.config import Settings
from research_assistant.graph_client import get_graph_client
from research_assistant.index import get_all_chunks
from research_assistant.llm import get_llm_client

logger = logging.getLogger(__name__)

_LOCAL_SEARCH_QUERY_TEMPLATE = (
    "MATCH (start:Entity) WHERE start.name IN $entities "
    "MATCH (start)-[:RELATED_TO*1..{hop_depth}]-(connected:Entity) "
    "RETURN DISTINCT connected.chunk_refs AS chunk_refs"
)

_GLOBAL_SEARCH_QUERY = "MATCH (c:Community) RETURN c.id AS id, c.summary AS summary"

_RELEVANCE_PROMPT_TEMPLATE = (
    "On a scale from 0.0 to 1.0, rate how relevant the community summary below is to the question.\n\n"
    "Question: {query}\n\n"
    "Summary: {summary}\n\n"
    "Respond with ONLY a JSON object of this exact shape:\n"
    '{{"relevance": <float between 0.0 and 1.0>}}'
)

FENCED_BLOCK_PATTERN = re.compile(r"^```(?:json)?\s*\n?(.*?)\n?```$", re.DOTALL)


def _strip_code_fence(raw: str) -> str:
    """Small/local models often wrap JSON responses in markdown code fences; non-str input is left to json.loads to reject."""
    if not isinstance(raw, str):
        return raw
    match = FENCED_BLOCK_PATTERN.match(raw.strip())
    return match.group(1) if match else raw


def local_search(entities: list[str], settings: Settings) -> list[Document]:
    if not entities:
        return []

    query = _LOCAL_SEARCH_QUERY_TEMPLATE.format(hop_depth=settings.graph_traversal_hop_depth)
    try:
        driver = get_graph_client(settings)
        with driver.session() as session:
            result = session.run(query, entities=entities)
            # Materialize rows before .consume(): the real driver streams
            # records lazily, so reading them after would see nothing.
            chunk_refs_lists = [record["chunk_refs"] for record in result]
            result.consume()
    except (neo4j.exceptions.ServiceUnavailable, neo4j.exceptions.Neo4jError, neo4j.exceptions.DriverError):
        logger.warning("Local search graph traversal failed; returning no graph results.", exc_info=True)
        return []

    refs = {ref for chunk_refs in chunk_refs_lists for ref in chunk_refs}
    lookup = {
        f"{doc.metadata['source_doc_id']}:{doc.metadata['chunk_index']}": doc
        for doc in get_all_chunks(settings)
    }
    return [lookup[ref] for ref in refs if ref in lookup]


def global_search(query: str, settings: Settings) -> list[Document]:
    try:
        driver = get_graph_client(settings)
        with driver.session() as session:
            result = session.run(_GLOBAL_SEARCH_QUERY)
            # Materialize rows before .consume(): the real driver streams
            # records lazily, so reading them after would see nothing.
            communities = [(record["id"], record["summary"]) for record in result]
            result.consume()
    except (neo4j.exceptions.ServiceUnavailable, neo4j.exceptions.Neo4jError, neo4j.exceptions.DriverError):
        logger.warning("Global search failed to read communities; returning no graph results.", exc_info=True)
        return []

    if not communities:
        return []

    client = get_llm_client(settings.generator)
    scored = []
    for community_id, summary in communities:
        prompt = _RELEVANCE_PROMPT_TEMPLATE.format(query=query, summary=summary)
        try:
            raw = client.generate(prompt)
            score = float(json.loads(_strip_code_fence(raw))["relevance"])
        except Exception:
            # Isolated per community: a scoring failure for one community -- a
            # malformed/non-JSON response or the LLM backend being entirely
            # unreachable -- must not abort scoring for the rest.
            logger.warning("Failed to score community %s relevance; scoring 0.0.", community_id, exc_info=True)
            score = 0.0
        scored.append((score, community_id, summary))

    scored.sort(key=lambda item: item[0], reverse=True)
    top = scored[: settings.global_search_top_communities]
    return [
        Document(
            page_content=summary,
            metadata={
                "source": f"community-{community_id}",
                "source_doc_id": f"community-{community_id}",
                "chunk_index": 0,
            },
        )
        for _, community_id, summary in top
    ]
