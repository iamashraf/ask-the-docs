import hashlib
import logging

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from neo4j.exceptions import Neo4jError, ServiceUnavailable

from research_assistant.config import ProviderConfig, Settings
from research_assistant.graph_extract import Triplet
from research_assistant.index import (
    CHUNK_SIZE,
    build_graph,
    build_index,
    chunk_documents,
    get_all_chunks,
    get_chunk_count,
    get_collection_name,
    load_index,
)
from research_assistant.llm import OllamaUnavailableError


def make_settings(tmp_path, embeddings_backend: str = "ollama", graph_enabled: bool = False) -> Settings:
    return Settings(
        generator=ProviderConfig(provider="ollama", model="llama3.1:8b"),
        judge=ProviderConfig(provider="anthropic", model="claude-sonnet-5-5", api_key="x"),
        embeddings_backend=embeddings_backend,
        index_dir=tmp_path / "index",
        graph_enabled=graph_enabled,
    )


def make_chunk(source_doc_id: str, chunk_index: int, text: str) -> Document:
    return Document(page_content=text, metadata={"source_doc_id": source_doc_id, "chunk_index": chunk_index})


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


def test_build_graph_skipped_when_graph_disabled(tmp_path, monkeypatch):
    called = []
    monkeypatch.setattr("research_assistant.index.extract_triplets", lambda *a, **k: called.append(1) or [])
    settings = make_settings(tmp_path, graph_enabled=False)

    build_graph([make_chunk("doc1", 0, "text")], settings)

    assert called == []


def test_build_graph_degrades_when_graph_client_unreachable(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr("research_assistant.index.extract_triplets", lambda *a, **k: [Triplet("A", "uses", "B")])
    monkeypatch.setattr("research_assistant.index.resolve_entities", lambda triplets, s: triplets)

    def boom(settings):
        raise ServiceUnavailable("down")

    monkeypatch.setattr("research_assistant.index.get_graph_client", boom)
    detect_calls = []
    summarize_calls = []
    monkeypatch.setattr(
        "research_assistant.index.run_community_detection", lambda d: detect_calls.append(d)
    )
    monkeypatch.setattr(
        "research_assistant.index.summarize_communities", lambda d, s: summarize_calls.append(d)
    )
    settings = make_settings(tmp_path, graph_enabled=True)

    with caplog.at_level(logging.WARNING):
        build_graph([make_chunk("doc1", 0, "text")], settings)  # must not raise

    assert any(
        record.name == "research_assistant.index" and "graph" in record.message.lower()
        for record in caplog.records
    )
    # No driver was ever obtained, so there's nothing to run community
    # detection/summarization with -- and no second ("Graph build failed")
    # warning from trying to anyway.
    assert detect_calls == []
    assert summarize_calls == []
    assert len(caplog.records) == 1


def test_build_graph_degrades_when_write_graph_raises(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr("research_assistant.index.extract_triplets", lambda *a, **k: [Triplet("A", "uses", "B")])
    monkeypatch.setattr("research_assistant.index.resolve_entities", lambda triplets, s: triplets)
    monkeypatch.setattr("research_assistant.index.get_graph_client", lambda s: "fake-driver")

    def boom(driver, chunk_triplets):
        raise Neo4jError("boom")

    monkeypatch.setattr("research_assistant.index.write_graph", boom)
    settings = make_settings(tmp_path, graph_enabled=True)

    with caplog.at_level(logging.WARNING):
        build_graph([make_chunk("doc1", 0, "text")], settings)  # must not raise

    assert any(
        record.name == "research_assistant.index" and "graph" in record.message.lower()
        for record in caplog.records
    )


def test_build_graph_degrades_when_extract_triplets_raises(tmp_path, monkeypatch, caplog):
    # A hard failure (e.g. the LLM backend itself unreachable), not just the
    # malformed-output case extract_triplets already degrades internally --
    # must not crash build_graph, same as the narrower Neo4j-specific catch.
    def boom(chunk_text, settings):
        raise OllamaUnavailableError("ollama is down")

    monkeypatch.setattr("research_assistant.index.extract_triplets", boom)
    settings = make_settings(tmp_path, graph_enabled=True)

    with caplog.at_level(logging.WARNING):
        build_graph([make_chunk("doc1", 0, "text")], settings)  # must not raise

    assert any(
        record.name == "research_assistant.index" and "graph" in record.message.lower()
        for record in caplog.records
    )


def test_build_graph_pairs_resolved_triplets_with_correct_chunk_refs(tmp_path, monkeypatch):
    def fake_extract(chunk_text, settings):
        return {
            "doc1 text": [Triplet("A", "uses", "B")],
            "doc1 text 2": [],  # a chunk with zero triplets must not break the loop
            "doc2 text": [Triplet("C", "implements", "D"), Triplet("E", "relates-to", "F")],
        }[chunk_text]

    resolve_calls = []

    def fake_resolve(triplets, settings):
        resolve_calls.append(list(triplets))
        return triplets  # identity: preserves length/order, same as the real contract

    written = {}

    def fake_write_graph(driver, chunk_triplets):
        written["chunk_triplets"] = chunk_triplets
        written["driver"] = driver

    monkeypatch.setattr("research_assistant.index.extract_triplets", fake_extract)
    monkeypatch.setattr("research_assistant.index.resolve_entities", fake_resolve)
    monkeypatch.setattr("research_assistant.index.get_graph_client", lambda s: "fake-driver")
    monkeypatch.setattr("research_assistant.index.write_graph", fake_write_graph)

    chunks = [
        make_chunk("doc1", 0, "doc1 text"),
        make_chunk("doc1", 1, "doc1 text 2"),
        make_chunk("doc2", 0, "doc2 text"),
    ]
    settings = make_settings(tmp_path, graph_enabled=True)

    build_graph(chunks, settings)

    # resolve_entities must be called once on the whole corpus, not once per chunk.
    assert len(resolve_calls) == 1
    assert resolve_calls[0] == [
        Triplet("A", "uses", "B"),
        Triplet("C", "implements", "D"),
        Triplet("E", "relates-to", "F"),
    ]
    assert written["driver"] == "fake-driver"
    assert written["chunk_triplets"] == [
        (Triplet("A", "uses", "B"), "doc1:0"),
        (Triplet("C", "implements", "D"), "doc2:0"),
        (Triplet("E", "relates-to", "F"), "doc2:0"),
    ]


def test_build_graph_noop_safe_with_empty_chunks_list(tmp_path, monkeypatch):
    extract_calls = []
    resolve_calls = []
    get_client_calls = []
    write_calls = []
    monkeypatch.setattr("research_assistant.index.extract_triplets", lambda *a, **k: extract_calls.append(1) or [])
    monkeypatch.setattr(
        "research_assistant.index.resolve_entities", lambda triplets, s: resolve_calls.append(1) or triplets
    )
    monkeypatch.setattr(
        "research_assistant.index.get_graph_client", lambda s: get_client_calls.append(1) or "fake-driver"
    )
    monkeypatch.setattr(
        "research_assistant.index.write_graph", lambda driver, chunk_triplets: write_calls.append(chunk_triplets)
    )
    settings = make_settings(tmp_path, graph_enabled=True)

    build_graph([], settings)

    assert extract_calls == []
    # Nothing to resolve/write/connect for -- short-circuits before touching
    # entity resolution or Neo4j at all, not just before writing anything.
    assert resolve_calls == []
    assert get_client_calls == []
    assert write_calls == []


def test_build_graph_noop_safe_when_every_chunk_extracts_zero_triplets(tmp_path, monkeypatch):
    # Realistic all-empty case: e.g. the LLM wraps every chunk's JSON in code
    # fences, so extract_triplets degrades every chunk to [] (Task 3).
    resolve_calls = []
    get_client_calls = []
    monkeypatch.setattr("research_assistant.index.extract_triplets", lambda *a, **k: [])
    monkeypatch.setattr(
        "research_assistant.index.resolve_entities", lambda triplets, s: resolve_calls.append(1) or triplets
    )
    monkeypatch.setattr(
        "research_assistant.index.get_graph_client", lambda s: get_client_calls.append(1) or "fake-driver"
    )
    settings = make_settings(tmp_path, graph_enabled=True)

    build_graph([make_chunk("doc1", 0, "text")], settings)

    assert resolve_calls == []
    assert get_client_calls == []


def test_build_graph_runs_community_detection_and_summarization_after_write(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr("research_assistant.index.extract_triplets", lambda *a, **k: [Triplet("A", "uses", "B")])
    monkeypatch.setattr("research_assistant.index.resolve_entities", lambda triplets, s: triplets)
    monkeypatch.setattr("research_assistant.index.get_graph_client", lambda s: "fake-driver")
    monkeypatch.setattr(
        "research_assistant.index.write_graph", lambda *a, **k: calls.append(("write", a[0]))
    )
    monkeypatch.setattr(
        "research_assistant.index.run_community_detection", lambda d: calls.append(("detect", d))
    )
    monkeypatch.setattr(
        "research_assistant.index.summarize_communities", lambda d, s: calls.append(("summarize", d, s))
    )
    settings = make_settings(tmp_path, graph_enabled=True)

    build_graph([make_chunk("doc1", 0, "text")], settings)

    # Verifies ordering and that the same driver/settings flow through each
    # step, not just that each function was called.
    assert calls == [
        ("write", "fake-driver"),
        ("detect", "fake-driver"),
        ("summarize", "fake-driver", settings),
    ]


def test_build_index_calls_build_graph_with_its_chunks(tmp_path, monkeypatch):
    monkeypatch.setattr("research_assistant.index.get_embeddings", lambda settings: FakeEmbeddings())
    captured = {}
    monkeypatch.setattr(
        "research_assistant.index.build_graph",
        lambda chunks, settings: captured.update(chunks=chunks, settings=settings),
    )
    settings = make_settings(tmp_path, graph_enabled=True)
    doc = Document(page_content="word " * 400, metadata={"source": "a.pdf", "source_doc_id": "docA"})

    build_index([doc], settings)

    assert captured["settings"] is settings
    assert [c.metadata["source_doc_id"] for c in captured["chunks"]] == ["docA"] * len(captured["chunks"])


def test_build_index_completes_vector_indexing_when_graph_build_degrades(tmp_path, monkeypatch):
    monkeypatch.setattr("research_assistant.index.get_embeddings", lambda settings: FakeEmbeddings())
    monkeypatch.setattr("research_assistant.index.extract_triplets", lambda *a, **k: [Triplet("A", "uses", "B")])
    monkeypatch.setattr("research_assistant.index.resolve_entities", lambda triplets, s: triplets)

    def boom(settings):
        raise ServiceUnavailable("down")

    monkeypatch.setattr("research_assistant.index.get_graph_client", boom)
    settings = make_settings(tmp_path, graph_enabled=True)
    doc = Document(page_content="word " * 400, metadata={"source": "a.pdf", "source_doc_id": "docA"})

    build_index([doc], settings)  # must not raise

    assert get_chunk_count(settings) > 0


def test_build_index_completes_vector_indexing_when_graph_extraction_raises(tmp_path, monkeypatch):
    monkeypatch.setattr("research_assistant.index.get_embeddings", lambda settings: FakeEmbeddings())

    def boom(chunk_text, settings):
        raise OllamaUnavailableError("ollama is down")

    monkeypatch.setattr("research_assistant.index.extract_triplets", boom)
    settings = make_settings(tmp_path, graph_enabled=True)
    doc = Document(page_content="word " * 400, metadata={"source": "a.pdf", "source_doc_id": "docA"})

    build_index([doc], settings)  # must not raise

    assert get_chunk_count(settings) > 0
