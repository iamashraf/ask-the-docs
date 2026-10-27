"""Local-query retrieval: traverse the graph from named entities and resolve matched chunk_refs to Documents."""

import logging

import neo4j
from langchain_core.documents import Document

from research_assistant.config import Settings
from research_assistant.graph_client import get_graph_client
from research_assistant.index import get_all_chunks

logger = logging.getLogger(__name__)

_LOCAL_SEARCH_QUERY_TEMPLATE = (
    "MATCH (start:Entity) WHERE start.name IN $entities "
    "MATCH (start)-[:RELATED_TO*1..{hop_depth}]-(connected:Entity) "
    "RETURN DISTINCT connected.chunk_refs AS chunk_refs"
)


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
    except (neo4j.exceptions.ServiceUnavailable, neo4j.exceptions.Neo4jError):
        logger.warning("Local search graph traversal failed; returning no graph results.", exc_info=True)
        return []

    refs = {ref for chunk_refs in chunk_refs_lists for ref in chunk_refs}
    lookup = {
        f"{doc.metadata['source_doc_id']}:{doc.metadata['chunk_index']}": doc
        for doc in get_all_chunks(settings)
    }
    return [lookup[ref] for ref in refs if ref in lookup]
