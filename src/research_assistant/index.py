"""Chunking and the per-embeddings-backend Chroma vector index."""

import logging
from datetime import UTC, datetime

import neo4j
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from research_assistant.config import Settings
from research_assistant.embeddings import get_embeddings
from research_assistant.entity_resolution import resolve_entities
from research_assistant.graph_client import get_graph_client
from research_assistant.graph_extract import extract_triplets
from research_assistant.graph_store import write_graph

logger = logging.getLogger(__name__)

CHUNK_SIZE = 512
CHUNK_OVERLAP = 50


def chunk_documents(docs: list[Document], embedding_model: str) -> list[Document]:
    """Splits docs into chunks and tags each with chunk_index/total_chunks
    (per source_doc_id), embedding_model, and ingested_at."""
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

    triplets = []
    chunk_refs = []
    for chunk in chunks:
        chunk_ref = f"{chunk.metadata['source_doc_id']}:{chunk.metadata['chunk_index']}"
        for triplet in extract_triplets(chunk.page_content, settings):
            triplets.append(triplet)
            chunk_refs.append(chunk_ref)

    if not triplets:
        return

    # Resolved once over the whole corpus's triplets, not per chunk -- entity
    # clustering needs every entity name in play to merge well.
    resolved_triplets = resolve_entities(triplets, settings)

    try:
        driver = get_graph_client(settings)
        write_graph(driver, list(zip(resolved_triplets, chunk_refs, strict=True)))
    except (neo4j.exceptions.ServiceUnavailable, neo4j.exceptions.Neo4jError):
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


def get_chunk_count(settings: Settings) -> int:
    """Cheap collection size, used as the cache-staleness signal -- avoids
    fetching every chunk's content just to check if anything changed."""
    return load_index(settings)._collection.count()
