"""Chunking and the per-embeddings-backend Chroma vector index."""

from datetime import UTC, datetime

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from research_assistant.config import Settings
from research_assistant.embeddings import get_embeddings

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


def build_index(docs: list[Document], settings: Settings) -> Chroma:
    chunks = chunk_documents(docs, embedding_model=_embedding_model_name(settings))
    return Chroma.from_documents(
        chunks,
        embedding=get_embeddings(settings),
        collection_name=get_collection_name(settings),
        persist_directory=str(settings.index_dir),
    )


def load_index(settings: Settings) -> Chroma:
    return Chroma(
        collection_name=get_collection_name(settings),
        embedding_function=get_embeddings(settings),
        persist_directory=str(settings.index_dir),
    )
