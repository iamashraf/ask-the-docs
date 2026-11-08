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
MERGE (s)-[r:RELATED_TO {relation: $relation}]->(o)
  ON CREATE SET r.chunk_refs = [$chunk_ref]
  ON MATCH SET r.chunk_refs = CASE WHEN $chunk_ref IN r.chunk_refs THEN r.chunk_refs ELSE r.chunk_refs + $chunk_ref END
"""

# A ref is "<source_doc_id>:<chunk_index>"; matching the whole id (no further ':' after it) keeps "a" from claiming "a:b" refs.
_REF_IN_SOURCES = (
    "ANY(src IN $sources WHERE x STARTS WITH src + ':' AND NOT substring(x, size(src) + 1) CONTAINS ':')"
)
_REMOVE_SOURCES_FROM_EDGES_QUERY = f"""
MATCH ()-[r:RELATED_TO]->()
WHERE ANY(x IN r.chunk_refs WHERE {_REF_IN_SOURCES})
SET r.chunk_refs = [x IN r.chunk_refs WHERE NOT {_REF_IN_SOURCES}]
WITH r WHERE size(r.chunk_refs) = 0
DELETE r
"""
_REMOVE_SOURCES_FROM_ENTITIES_QUERY = f"""
MATCH (e:Entity)
WHERE ANY(x IN e.chunk_refs WHERE {_REF_IN_SOURCES})
SET e.chunk_refs = [x IN e.chunk_refs WHERE NOT {_REF_IN_SOURCES}]
WITH e WHERE size(e.chunk_refs) = 0
DETACH DELETE e
"""

_CLEAR_GRAPH_QUERY = "MATCH (n) WHERE n:Entity OR n:Community DETACH DELETE n"


def clear_graph(driver: neo4j.Driver) -> None:
    """Wipes entities and communities so a rebuild from all chunks leaves no stale facts; raises on Neo4j errors."""
    with driver.session() as session:
        session.run(_CLEAR_GRAPH_QUERY).consume()


def remove_sources(driver: neo4j.Driver, source_doc_ids: set[str]) -> None:
    """Strips these sources' chunk_refs and deletes edges/entities left with none; raises on Neo4j errors."""
    if not source_doc_ids:
        return
    with driver.session() as session:
        for query in (_REMOVE_SOURCES_FROM_EDGES_QUERY, _REMOVE_SOURCES_FROM_ENTITIES_QUERY):
            session.run(query, sources=sorted(source_doc_ids)).consume()


def graph_supports_incremental(driver: neo4j.Driver) -> bool:
    """True only for a non-empty graph whose edges all carry chunk_refs; an empty or pre-edge-ref graph needs a full rebuild."""
    with driver.session() as session:
        entities = session.run("MATCH (e:Entity) RETURN count(e) AS n").single()["n"]
        legacy_edges = session.run("MATCH ()-[r:RELATED_TO]->() WHERE r.chunk_refs IS NULL RETURN count(r) AS n").single()["n"]
    return entities > 0 and legacy_edges == 0


def get_entity_names(driver: neo4j.Driver) -> list[str]:
    with driver.session() as session:
        return sorted(record["name"] for record in session.run("MATCH (e:Entity) RETURN e.name AS name"))


def write_graph(driver: neo4j.Driver, chunk_triplets: list[tuple[Triplet, str]]) -> str | None:
    """Returns None on success, else the failure's exception class name."""
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
    except (neo4j.exceptions.ServiceUnavailable, neo4j.exceptions.Neo4jError) as exc:
        logger.warning("Graph write failed; continuing without updating the graph.", exc_info=True)
        return type(exc).__name__
    return None
