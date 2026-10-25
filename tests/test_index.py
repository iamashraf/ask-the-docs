import hashlib

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from research_assistant.config import ProviderConfig, Settings
from research_assistant.index import (
    CHUNK_SIZE,
    build_index,
    chunk_documents,
    get_all_chunks,
    get_chunk_count,
    get_collection_name,
    load_index,
)


def make_settings(tmp_path, embeddings_backend: str = "ollama") -> Settings:
    return Settings(
        generator=ProviderConfig(provider="ollama", model="llama3.1:8b"),
        judge=ProviderConfig(provider="anthropic", model="claude-sonnet-5-5", api_key="x"),
        embeddings_backend=embeddings_backend,
        index_dir=tmp_path / "index",
    )


def _vector(text: str) -> list[float]:
    digest = hashlib.md5(text.encode()).digest()
    return [b / 255 for b in digest[:8]]


class FakeEmbeddings(Embeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [_vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return _vector(text)


def test_chunk_documents_respects_chunk_size():
    doc = Document(page_content="word " * 400, metadata={"source": "paper.pdf", "source_doc_id": "doc1"})

    chunks = chunk_documents([doc], embedding_model="nomic-embed-text")

    assert len(chunks) > 1
    assert all(len(c.page_content) <= CHUNK_SIZE for c in chunks)


def test_chunk_documents_sets_metadata():
    doc = Document(page_content="word " * 400, metadata={"source": "paper.pdf", "source_doc_id": "doc1"})

    chunks = chunk_documents([doc], embedding_model="nomic-embed-text")

    assert [c.metadata["chunk_index"] for c in chunks] == list(range(len(chunks)))
    assert all(c.metadata["total_chunks"] == len(chunks) for c in chunks)
    assert all(c.metadata["embedding_model"] == "nomic-embed-text" for c in chunks)
    assert all("ingested_at" in c.metadata for c in chunks)


def test_chunk_documents_indexes_restart_per_source_doc():
    doc_a = Document(page_content="word " * 400, metadata={"source": "a.pdf", "source_doc_id": "docA"})
    doc_b = Document(page_content="word " * 400, metadata={"source": "b.pdf", "source_doc_id": "docB"})

    chunks = chunk_documents([doc_a, doc_b], embedding_model="m")

    a_chunks = [c for c in chunks if c.metadata["source_doc_id"] == "docA"]
    b_chunks = [c for c in chunks if c.metadata["source_doc_id"] == "docB"]
    assert [c.metadata["chunk_index"] for c in a_chunks] == list(range(len(a_chunks)))
    assert [c.metadata["chunk_index"] for c in b_chunks] == list(range(len(b_chunks)))


def test_get_collection_name_differs_by_backend(tmp_path):
    ollama_settings = make_settings(tmp_path, embeddings_backend="ollama")
    hf_settings = make_settings(tmp_path, embeddings_backend="huggingface")

    assert get_collection_name(ollama_settings) != get_collection_name(hf_settings)


def test_build_index_then_load_index_round_trips(tmp_path, monkeypatch):
    monkeypatch.setattr("research_assistant.index.get_embeddings", lambda settings: FakeEmbeddings())
    settings = make_settings(tmp_path)
    doc = Document(
        page_content="Attention is all you need for this transformer paper.",
        metadata={"source": "paper.pdf", "source_doc_id": "doc1"},
    )

    build_index([doc], settings)
    reloaded = load_index(settings)
    results = reloaded.similarity_search("Attention is all you need for this transformer paper.", k=1)

    assert len(results) == 1
    assert "Attention is all you need" in results[0].page_content


def test_get_all_chunks_returns_every_indexed_chunk(tmp_path, monkeypatch):
    monkeypatch.setattr("research_assistant.index.get_embeddings", lambda settings: FakeEmbeddings())
    settings = make_settings(tmp_path)
    docs = [
        Document(page_content="word " * 400, metadata={"source": "a.pdf", "source_doc_id": "docA"}),
        Document(page_content="word " * 400, metadata={"source": "b.pdf", "source_doc_id": "docB"}),
    ]
    build_index(docs, settings)

    chunks = get_all_chunks(settings)

    assert len(chunks) == get_chunk_count(settings)
    assert {c.metadata["source"] for c in chunks} == {"a.pdf", "b.pdf"}
    assert all(c.page_content for c in chunks)


def test_get_chunk_count_matches_number_of_chunks_built(tmp_path, monkeypatch):
    monkeypatch.setattr("research_assistant.index.get_embeddings", lambda settings: FakeEmbeddings())
    settings = make_settings(tmp_path)
    doc = Document(page_content="word " * 400, metadata={"source": "a.pdf", "source_doc_id": "docA"})

    expected = len(chunk_documents([doc], embedding_model="m"))
    build_index([doc], settings)

    assert get_chunk_count(settings) == expected
