"""Chunking and the per-embeddings-backend Chroma vector index."""

import hashlib
import logging
from datetime import UTC, datetime
from pathlib import Path

import neo4j
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from ask_the_docs.community import run_community_detection, summarize_communities
from ask_the_docs.config import Settings
from ask_the_docs.embeddings import get_embeddings
from ask_the_docs.entity_resolution import resolve_entities
from ask_the_docs.graph_client import get_graph_client
from ask_the_docs.graph_extract import extract_triplets
from ask_the_docs.graph_store import write_graph
from ask_the_docs.llm import OllamaUnavailableError

logger = logging.getLogger(__name__)

CHUNK_SIZE = 512
CHUNK_OVERLAP = 50
MARKDOWN_HEADERS_TO_SPLIT_ON = [("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3")]
# Chroma.from_documents used to batch writes internally (create_batches);
# load_index(settings).add_documents() doesn't, so build_index batches it
# here instead -- keeps each Ollama embed request and each Chroma upsert
# bounded regardless of corpus size.
INDEX_WRITE_BATCH_SIZE = 64


class IndexWriteError(RuntimeError):
    pass


def _generation_path(settings: Settings) -> Path:
    return settings.index_dir / ".generation"


def get_index_generation(settings: Settings) -> int:
    """Monotonic counter bumped on every real write to the index (insert
    or delete-and-replace) -- unlike chunk count, it changes even when a
    replace produces the same number of chunks, so cache keys built from
    it (answer.py, bm25_index.py) can't be fooled by a same-length edit.
    Reads a local file, not Chroma, so computing a cache key never needs
    the index to be reachable."""
    path = _generation_path(settings)
    return int(path.read_text()) if path.exists() else 0


def _bump_index_generation(settings: Settings) -> None:
    path = _generation_path(settings)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(str(get_index_generation(settings) + 1))


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


def _markdown_heading_prefix(section_metadata: dict) -> str:
    titles = [section_metadata[key] for _, key in MARKDOWN_HEADERS_TO_SPLIT_ON if key in section_metadata]
    return " > ".join(titles)


def _split_markdown_document(doc: Document) -> list[Document]:
    """Splits on header boundaries first so every resulting chunk carries its
    section heading -- plain char-splitting only keeps the heading text in
    whichever single chunk it happens to land in, losing it for every other
    chunk in that section."""
    sections = MarkdownHeaderTextSplitter(MARKDOWN_HEADERS_TO_SPLIT_ON, strip_headers=True).split_text(
        doc.page_content
    )

    chunks = []
    for section in sections:
        heading = _markdown_heading_prefix(section.metadata)
        prefix = f"{heading}\n" if heading else ""
        body_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE - len(prefix), chunk_overlap=CHUNK_OVERLAP
        )
        for body_chunk in body_splitter.split_text(section.page_content):
            chunks.append(Document(page_content=prefix + body_chunk, metadata={**doc.metadata, **section.metadata}))
    return chunks


def chunk_documents(docs: list[Document], embedding_model: str) -> list[Document]:
    """Splits docs into chunks and tags each with chunk_index/total_chunks
    (per source_doc_id), embedding_model, ingested_at, and content_hash.
    Documents tagged content_format="markdown" (.md sources, and PDFs
    converted to markdown upstream) are split header-first (see
    _split_markdown_document); every other source type keeps today's plain
    char-based splitting."""
    source_hashes = _hash_sources(docs)
    generic_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks: list[Document] = []
    for doc in docs:
        if doc.metadata.get("content_format") == "markdown":
            chunks.extend(_split_markdown_document(doc))
        else:
            chunks.extend(generic_splitter.split_documents([doc]))

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


def _group_by_source(chunks: list[Document]) -> dict[str, list[Document]]:
    grouped: dict[str, list[Document]] = {}
    for chunk in chunks:
        grouped.setdefault(chunk.metadata["source_doc_id"], []).append(chunk)
    return grouped


def build_index(docs: list[Document], settings: Settings) -> Chroma:
    """Idempotent: a source whose existing chunks already match its fresh
    content hash and count is skipped entirely (no re-embed, no re-insert);
    any other source (new, changed, or a stale partial-write leftover) has
    its existing chunks deleted and replaced."""
    chunks = chunk_documents(docs, embedding_model=_embedding_model_name(settings))
    index = load_index(settings)
    chunks_by_source = _group_by_source(chunks)
    # Iterate every source_doc_id that was *attempted*, not just the ones
    # that produced a chunk: a source whose content now chunks to zero
    # (e.g. no extractable text) still needs its stale existing chunks
    # deleted, which a chunks-only loop would silently skip.
    attempted_source_ids = {doc.metadata["source_doc_id"] for doc in docs}

    to_insert: list[Document] = []
    for source_doc_id in attempted_source_ids:
        source_chunks = chunks_by_source.get(source_doc_id, [])
        existing = index.get(where={"source_doc_id": source_doc_id}, include=["metadatas"])
        if existing["ids"]:
            # .get(), not [...]: a chunk indexed before content-hash dedup
            # existed has no "content_hash" key at all, not a mismatched
            # one -- None here correctly reads as "changed" and triggers
            # a replace, same as any other mismatch.
            existing_hashes = {m.get("content_hash") for m in existing["metadatas"]}
            if (
                source_chunks
                and existing_hashes == {source_chunks[0].metadata["content_hash"]}
                and len(existing["ids"]) == len(source_chunks)
            ):
                continue
            index.delete(ids=existing["ids"])
        to_insert.extend(source_chunks)

    if to_insert:
        try:
            for i in range(0, len(to_insert), INDEX_WRITE_BATCH_SIZE):
                index.add_documents(to_insert[i : i + INDEX_WRITE_BATCH_SIZE])
        except OllamaUnavailableError:
            raise
        except Exception as exc:
            raise IndexWriteError(f"Failed to write {len(to_insert)} chunk(s) to the index") from exc
        _bump_index_generation(settings)
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
