"""Chunking and the per-embeddings-backend Chroma vector index."""

import hashlib
import logging
from datetime import UTC, datetime

import neo4j
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ask_the_docs.community import run_community_detection, summarize_communities
from ask_the_docs.config import Settings
from ask_the_docs.embeddings import get_embeddings
from ask_the_docs.entity_resolution import resolve_entities
from ask_the_docs.graph_client import get_graph_client
from ask_the_docs.graph_extract import extract_triplets
from ask_the_docs.graph_store import write_graph

logger = logging.getLogger(__name__)

CHUNK_SIZE = 512
CHUNK_OVERLAP = 50


def _hash_sources(docs: list[Document]) -> dict[str, str]:
    """SHA-256 of each source_doc_id's full raw content (every Document
    sharing that id, concatenated in input order), computed before
    chunking -- lets a re-ingest detect an unchanged source without
    comparing chunk-by-chunk."""
    content_by_id: dict[str, str] = {}
    for doc in docs:
        doc_id = doc.metadata["source_doc_id"]
        content_by_id[doc_id] = content_by_id.get(doc_id, "") + doc.page_content
    return {doc_id: hashlib.sha256(content.encode()).hexdigest() for doc_id, content in content_by_id.items()}


def chunk_documents(docs: list[Document], embedding_model: str) -> list[Document]:
    """Splits docs into chunks and tags each with chunk_index/total_chunks
    (per source_doc_id), embedding_model, ingested_at, and content_hash."""
    source_hashes = _hash_sources(docs)
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_documents(docs)

    totals: dict[str, int] = {}
    for chunk in chunks:
        doc_id = chunk.metadata["source_doc_id"]
        totals[doc_id] = totals.get(doc_id, 0) + 1

    ingested_at = datetime.now(UTC).isoformat()
    running_index: dict[str, int] = {}
    for chunk in chunks:
        doc_id = chunk.metadata["source_doc_id"]
        chunk.metadata["chunk_index"] = running_index.get(doc_id, 0)
        chunk.metadata["total_chunks"] = totals[doc_id]
        chunk.metadata["embedding_model"] = embedding_model
        chunk.metadata["ingested_at"] = ingested_at
        chunk.metadata["content_hash"] = source_hashes[doc_id]
        running_index[doc_id] = chunk.metadata["chunk_index"] + 1

    return chunks


def get_collection_name(settings: Settings) -> str:
    """Keeps each embeddings backend's vectors in their own collection so
    switching backends never mixes incompatible vectors."""
    return f"chunks_{settings.embeddings_backend}"


def _embedding_model_name(settings: Settings) -> str:
    if settings.embeddings_backend == "ollama":
        return settings.ollama_embedding_model
    return settings.huggingface_embedding_model


def build_graph(chunks: list[Document], settings: Settings) -> None:
    """Extracts/resolves/writes a knowledge graph from chunks; no-op unless settings.graph_enabled."""
    if not settings.graph_enabled:
        return

    try:
        triplets = []
        chunk_refs = []
        for chunk in chunks:
            chunk_ref = f"{chunk.metadata['source_doc_id']}:{chunk.metadata['chunk_index']}"
            try:
                chunk_triplets = extract_triplets(chunk.page_content, settings)
            except Exception as exc:
                # Isolated per chunk: one transient LLM failure must not abort
                # graph building for every other chunk in the ingest.
                logger.warning("Graph triplet extraction failed for chunk %s: %s", chunk_ref, exc)
                continue
            for triplet in chunk_triplets:
                triplets.append(triplet)
                chunk_refs.append(chunk_ref)

        if not triplets:
            return

        # Resolved once over the whole corpus's triplets, not per chunk -- entity
        # clustering needs every entity name in play to merge well.
        resolved_triplets = resolve_entities(triplets, settings)

        driver = None
        try:
            driver = get_graph_client(settings)
            write_graph(driver, list(zip(resolved_triplets, chunk_refs, strict=True)))
        except (neo4j.exceptions.ServiceUnavailable, neo4j.exceptions.Neo4jError):
            logger.warning("Graph write failed; continuing without updating the graph.", exc_info=True)

        # Skip if get_graph_client itself failed above -- no driver to run
        # community detection/summarization with.
        if driver is not None:
            run_community_detection(driver)
            summarize_communities(driver, settings)
    except Exception:
        # Graph building is an optional enhancement, same reasoning as
        # retrieve.py's query-rewrite step -- it must never crash ingestion.
        logger.warning("Graph build failed; continuing without updating the graph.", exc_info=True)


def build_index(docs: list[Document], settings: Settings) -> Chroma:
    chunks = chunk_documents(docs, embedding_model=_embedding_model_name(settings))
    index = Chroma.from_documents(
        chunks,
        embedding=get_embeddings(settings),
        collection_name=get_collection_name(settings),
        persist_directory=str(settings.index_dir),
    )
    build_graph(chunks, settings)
    return index


def load_index(settings: Settings) -> Chroma:
    return Chroma(
        collection_name=get_collection_name(settings),
        embedding_function=get_embeddings(settings),
        persist_directory=str(settings.index_dir),
    )


def get_all_chunks(settings: Settings) -> list[Document]:
    """Full chunk fetch -- used to (re)build the BM25 index, not on every
    query. See get_chunk_count for the cheap per-query staleness check."""
    result = load_index(settings).get(include=["metadatas", "documents"])
    return [
        Document(page_content=content, metadata=metadata)
        for content, metadata in zip(result["documents"], result["metadatas"])
    ]


def get_chunks_by_refs(settings: Settings, refs: list[str]) -> list[Document]:
    """Targeted fetch of "source_doc_id:chunk_index" refs; Chroma rejects $or/$and with fewer than two clauses."""
    if not refs:
        return []
    clauses = []
    for ref in refs:
        source_doc_id, chunk_index = ref.rsplit(":", 1)
        clauses.append({"$and": [{"source_doc_id": source_doc_id}, {"chunk_index": int(chunk_index)}]})
    where = clauses[0] if len(clauses) == 1 else {"$or": clauses}
    result = load_index(settings).get(where=where, include=["metadatas", "documents"])
    return [
        Document(page_content=content, metadata=metadata)
        for content, metadata in zip(result["documents"], result["metadatas"])
    ]


def get_chunk_count(settings: Settings) -> int:
    """Cheap collection size, used as the cache-staleness signal -- avoids
    fetching every chunk's content just to check if anything changed."""
    return load_index(settings)._collection.count()
