"""Writes resolved triplets into Neo4j as the Entity/RELATED_TO graph schema."""

import logging

import neo4j

from ask_the_docs.graph_extract import Triplet

logger = logging.getLogger(__name__)

_MERGE_TRIPLET_QUERY = """
MERGE (s:Entity {name: $subject})
  ON CREATE SET s.chunk_refs = [$chunk_ref]
  ON MATCH SET s.chunk_refs = CASE WHEN $chunk_ref IN s.chunk_refs THEN s.chunk_refs ELSE s.chunk_refs + $chunk_ref END
MERGE (o:Entity {name: $object})
  ON CREATE SET o.chunk_refs = [$chunk_ref]
  ON MATCH SET o.chunk_refs = CASE WHEN $chunk_ref IN o.chunk_refs THEN o.chunk_refs ELSE o.chunk_refs + $chunk_ref END
MERGE (s)-[:RELATED_TO {relation: $relation}]->(o)
"""


def write_graph(driver: neo4j.Driver, chunk_triplets: list[tuple[Triplet, str]]) -> None:
    try:
        with driver.session() as session:
            for triplet, chunk_ref in chunk_triplets:
                # .consume() forces each query to fully execute now, so an error raised
                # during execution surfaces at this call rather than being deferred and
                # raised by a later session.run() (neo4j's buffer-on-next-call behavior).
                session.run(
                    _MERGE_TRIPLET_QUERY,
                    subject=triplet.subject,
                    relation=triplet.relation,
                    object=triplet.obj,
                    chunk_ref=chunk_ref,
                ).consume()
    except (neo4j.exceptions.ServiceUnavailable, neo4j.exceptions.Neo4jError):
        logger.warning("Graph write failed; continuing without updating the graph.", exc_info=True)
