import hashlib
import logging

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from neo4j.exceptions import Neo4jError, ServiceUnavailable

from ask_the_docs.config import ProviderConfig, Settings
from ask_the_docs.graph_extract import Triplet
from ask_the_docs.index import (
    CHUNK_SIZE,
    INDEX_WRITE_BATCH_SIZE,
    IndexWriteError,
    build_graph,
    build_index,
    chunk_documents,
    get_all_chunks,
    get_chunk_count,
    get_chunks_by_refs,
    get_collection_name,
    get_index_generation,
    load_index,
)
from ask_the_docs.llm import OllamaUnavailableError


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


class CountingEmbeddings(Embeddings):
    """Wraps FakeEmbeddings to prove a skipped source is never re-embedded, not just never re-inserted."""

    def __init__(self):
        self.embed_documents_call_count = 0

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.embed_documents_call_count += 1
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


def test_chunk_documents_sets_content_hash_metadata():
    doc = Document(page_content="hello world", metadata={"source": "a.txt", "source_doc_id": "a"})
    chunks = chunk_documents([doc], embedding_model="m")
    assert all("content_hash" in c.metadata for c in chunks)


def test_chunk_documents_content_hash_is_same_across_a_multi_page_source():
    page1 = Document(page_content="page one", metadata={"source": "a.pdf", "source_doc_id": "a"})
    page2 = Document(page_content="page two", metadata={"source": "a.pdf", "source_doc_id": "a"})
    chunks = chunk_documents([page1, page2], embedding_model="m")
    assert len({c.metadata["content_hash"] for c in chunks}) == 1


def test_chunk_documents_content_hash_differs_across_sources():
    doc_a = Document(page_content="hello world", metadata={"source": "a.txt", "source_doc_id": "a"})
    doc_b = Document(page_content="goodbye world", metadata={"source": "b.txt", "source_doc_id": "b"})
    chunks = chunk_documents([doc_a, doc_b], embedding_model="m")
    hash_a = next(c.metadata["content_hash"] for c in chunks if c.metadata["source_doc_id"] == "a")
    hash_b = next(c.metadata["content_hash"] for c in chunks if c.metadata["source_doc_id"] == "b")
    assert hash_a != hash_b


def test_chunk_documents_content_hash_is_stable_across_calls():
    doc = Document(page_content="hello world", metadata={"source": "a.txt", "source_doc_id": "a"})
    first = chunk_documents([doc], embedding_model="m")[0].metadata["content_hash"]
    second = chunk_documents([doc], embedding_model="m")[0].metadata["content_hash"]
    assert first == second


def test_chunk_documents_content_hash_changes_when_content_changes():
    doc_v1 = Document(page_content="hello world", metadata={"source": "a.txt", "source_doc_id": "a"})
    doc_v2 = Document(page_content="hello universe", metadata={"source": "a.txt", "source_doc_id": "a"})
    hash_v1 = chunk_documents([doc_v1], embedding_model="m")[0].metadata["content_hash"]
    hash_v2 = chunk_documents([doc_v2], embedding_model="m")[0].metadata["content_hash"]
    assert hash_v1 != hash_v2


def test_chunk_documents_markdown_heading_survives_in_every_chunk_of_its_section():
    # Body alone is long enough to span several 512-char chunks -- plain
    # char-splitting only keeps "Refund Policy" in whichever one chunk the
    # heading line happens to land in, losing it for every other chunk in
    # the same section. Header-aware splitting must keep it in all of them.
    text = "## Refund Policy\n" + "Customers may request a refund within 30 days of purchase, subject to conditions. " * 15
    doc = Document(
        page_content=text,
        metadata={"source": "policy.md", "source_doc_id": "doc1", "content_format": "markdown"},
    )

    chunks = chunk_documents([doc], embedding_model="nomic-embed-text")

    assert len(chunks) > 1
    assert all("Refund Policy" in c.page_content for c in chunks)
    assert all(c.metadata["embedding_model"] == "nomic-embed-text" for c in chunks)


def test_chunk_documents_markdown_long_section_still_respects_chunk_size():
    text = "## Terms\n" + "word " * 400
    doc = Document(
        page_content=text,
        metadata={"source": "terms.md", "source_doc_id": "doc1", "content_format": "markdown"},
    )

    chunks = chunk_documents([doc], embedding_model="m")

    assert len(chunks) > 1
    assert all(len(c.page_content) <= CHUNK_SIZE for c in chunks)


def test_chunk_documents_markdown_without_headers_falls_back_cleanly():
    doc = Document(
        page_content="word " * 400,
        metadata={"source": "plain.md", "source_doc_id": "doc1", "content_format": "markdown"},
    )

    chunks = chunk_documents([doc], embedding_model="m")

    assert len(chunks) > 1
    assert all(len(c.page_content) <= CHUNK_SIZE for c in chunks)


def test_chunk_documents_routes_by_content_format_not_file_extension():
    # PDF sources are converted to markdown text upstream (pymupdf4llm) and
    # tagged content_format="markdown" -- dispatch must key off that tag,
    # not the ".md" suffix, so a .pdf source gets the same header-aware
    # splitting a .md source does. Same long-section setup as the markdown
    # heading-survival test above, just with a non-.md source path.
    text = "## Refund Policy\n" + "Customers may request a refund within 30 days of purchase, subject to conditions. " * 15
    doc = Document(
        page_content=text,
        metadata={"source": "policy.pdf", "source_doc_id": "doc1", "content_format": "markdown"},
    )

    chunks = chunk_documents([doc], embedding_model="m")

    assert len(chunks) > 1
    assert all("Refund Policy" in c.page_content for c in chunks)


def test_get_collection_name_differs_by_backend(tmp_path):
    ollama_settings = make_settings(tmp_path, embeddings_backend="ollama")
    hf_settings = make_settings(tmp_path, embeddings_backend="huggingface")

    assert get_collection_name(ollama_settings) != get_collection_name(hf_settings)


def test_build_index_then_load_index_round_trips(tmp_path, monkeypatch):
    monkeypatch.setattr("ask_the_docs.index.get_embeddings", lambda settings: FakeEmbeddings())
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


def test_build_index_reingesting_unchanged_doc_is_a_noop(tmp_path, monkeypatch):
    counting = CountingEmbeddings()
    monkeypatch.setattr("ask_the_docs.index.get_embeddings", lambda settings: counting)
    settings = make_settings(tmp_path)
    doc = Document(page_content="word " * 400, metadata={"source": "a.pdf", "source_doc_id": "docA"})

    build_index([doc], settings)
    first_count = get_chunk_count(settings)
    calls_after_first = counting.embed_documents_call_count

    build_index([doc], settings)

    assert get_chunk_count(settings) == first_count
    assert counting.embed_documents_call_count == calls_after_first


def test_build_index_reingesting_changed_doc_replaces_old_chunks(tmp_path, monkeypatch):
    monkeypatch.setattr("ask_the_docs.index.get_embeddings", lambda settings: FakeEmbeddings())
    settings = make_settings(tmp_path)
    doc_v1 = Document(page_content="alpha " * 200, metadata={"source": "a.pdf", "source_doc_id": "docA"})
    doc_v2 = Document(page_content="omega " * 200, metadata={"source": "a.pdf", "source_doc_id": "docA"})

    build_index([doc_v1], settings)
    build_index([doc_v2], settings)

    chunks = get_all_chunks(settings)
    assert all("alpha" not in c.page_content for c in chunks)
    assert all("omega" in c.page_content for c in chunks)
    assert len(chunks) == len(chunk_documents([doc_v2], embedding_model="m"))


def test_build_index_mixed_batch_only_replaces_the_changed_source(tmp_path, monkeypatch):
    counting = CountingEmbeddings()
    monkeypatch.setattr("ask_the_docs.index.get_embeddings", lambda settings: counting)
    settings = make_settings(tmp_path)
    doc_a = Document(page_content="alpha " * 200, metadata={"source": "a.pdf", "source_doc_id": "docA"})
    doc_b_v1 = Document(page_content="beta " * 200, metadata={"source": "b.pdf", "source_doc_id": "docB"})
    doc_b_v2 = Document(page_content="gamma " * 200, metadata={"source": "b.pdf", "source_doc_id": "docB"})

    build_index([doc_a, doc_b_v1], settings)
    calls_after_first = counting.embed_documents_call_count

    build_index([doc_a, doc_b_v2], settings)

    assert counting.embed_documents_call_count == calls_after_first + 1
    chunks = get_all_chunks(settings)
    a_chunks = [c for c in chunks if c.metadata["source_doc_id"] == "docA"]
    b_chunks = [c for c in chunks if c.metadata["source_doc_id"] == "docB"]
    assert all("alpha" in c.page_content for c in a_chunks)
    assert all("gamma" in c.page_content for c in b_chunks)


def test_build_index_treats_a_stale_partial_write_as_changed(tmp_path, monkeypatch):
    """Simulates a prior crash: a leftover chunk set whose content_hash
    doesn't match what the current input would produce. Must be replaced,
    not left in place alongside the fresh chunks."""
    monkeypatch.setattr("ask_the_docs.index.get_embeddings", lambda settings: FakeEmbeddings())
    settings = make_settings(tmp_path)
    stale_chunk = Document(
        page_content="leftover from a crashed ingest",
        metadata={"source_doc_id": "docA", "chunk_index": 0, "content_hash": "stale-hash"},
    )
    load_index(settings).add_documents([stale_chunk])
    doc = Document(page_content="fresh content", metadata={"source": "a.pdf", "source_doc_id": "docA"})

    build_index([doc], settings)

    chunks = get_all_chunks(settings)
    assert all("leftover" not in c.page_content for c in chunks)
    assert all("fresh content" in c.page_content for c in chunks)


def test_build_index_with_empty_docs_list_is_a_noop(tmp_path, monkeypatch):
    monkeypatch.setattr("ask_the_docs.index.get_embeddings", lambda settings: FakeEmbeddings())
    settings = make_settings(tmp_path)

    build_index([], settings)  # must not raise

    assert get_chunk_count(settings) == 0


def test_build_index_wraps_non_ollama_write_failures(tmp_path, monkeypatch):
    monkeypatch.setattr("ask_the_docs.index.get_embeddings", lambda settings: FakeEmbeddings())

    class BrokenChroma:
        def get(self, where, include):
            return {"ids": [], "metadatas": []}

        def add_documents(self, docs):
            raise RuntimeError("disk full")

    monkeypatch.setattr("ask_the_docs.index.load_index", lambda settings: BrokenChroma())
    doc = Document(page_content="hello", metadata={"source": "a.txt", "source_doc_id": "a"})

    with pytest.raises(IndexWriteError) as exc_info:
        build_index([doc], make_settings(tmp_path))
    assert isinstance(exc_info.value.__cause__, RuntimeError)


def test_build_index_lets_ollama_unavailable_propagate_unwrapped(tmp_path, monkeypatch):
    monkeypatch.setattr("ask_the_docs.index.get_embeddings", lambda settings: FakeEmbeddings())

    class UnreachableChroma:
        def get(self, where, include):
            return {"ids": [], "metadatas": []}

        def add_documents(self, docs):
            raise OllamaUnavailableError("could not reach ollama")

    monkeypatch.setattr("ask_the_docs.index.load_index", lambda settings: UnreachableChroma())
    doc = Document(page_content="hello", metadata={"source": "a.txt", "source_doc_id": "a"})

    with pytest.raises(OllamaUnavailableError):
        build_index([doc], make_settings(tmp_path))


def test_get_all_chunks_returns_every_indexed_chunk(tmp_path, monkeypatch):
    monkeypatch.setattr("ask_the_docs.index.get_embeddings", lambda settings: FakeEmbeddings())
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


def _build_two_doc_index(tmp_path, monkeypatch):
    monkeypatch.setattr("ask_the_docs.index.get_embeddings", lambda settings: FakeEmbeddings())
    settings = make_settings(tmp_path)
    docs = [
        Document(page_content="alpha " * 200, metadata={"source": "a.pdf", "source_doc_id": "docA"}),
        Document(page_content="beta " * 200, metadata={"source": "b.pdf", "source_doc_id": "docB"}),
    ]
    build_index(docs, settings)
    return settings


def _refs_of(chunks):
    return {f"{c.metadata['source_doc_id']}:{c.metadata['chunk_index']}" for c in chunks}


def test_get_chunks_by_refs_fetches_only_the_requested_chunks(tmp_path, monkeypatch):
    settings = _build_two_doc_index(tmp_path, monkeypatch)

    chunks = get_chunks_by_refs(settings, ["docA:0", "docB:1"])

    assert _refs_of(chunks) == {"docA:0", "docB:1"}
    assert all(c.page_content for c in chunks)


def test_get_chunks_by_refs_handles_a_single_ref(tmp_path, monkeypatch):
    settings = _build_two_doc_index(tmp_path, monkeypatch)

    chunks = get_chunks_by_refs(settings, ["docA:1"])

    assert _refs_of(chunks) == {"docA:1"}


def test_get_chunks_by_refs_skips_refs_not_in_index(tmp_path, monkeypatch):
    settings = _build_two_doc_index(tmp_path, monkeypatch)

    chunks = get_chunks_by_refs(settings, ["docA:0", "doc_deleted:5"])

    assert _refs_of(chunks) == {"docA:0"}


def test_get_chunks_by_refs_returns_empty_list_for_no_refs(tmp_path, monkeypatch):
    def no_fetch(settings):
        raise AssertionError("no refs must not touch Chroma")

    monkeypatch.setattr("ask_the_docs.index.load_index", no_fetch)

    assert get_chunks_by_refs(make_settings(tmp_path), []) == []


def test_get_chunk_count_matches_number_of_chunks_built(tmp_path, monkeypatch):
    monkeypatch.setattr("ask_the_docs.index.get_embeddings", lambda settings: FakeEmbeddings())
    settings = make_settings(tmp_path)
    doc = Document(page_content="word " * 400, metadata={"source": "a.pdf", "source_doc_id": "docA"})

    expected = len(chunk_documents([doc], embedding_model="m"))
    build_index([doc], settings)

    assert get_chunk_count(settings) == expected


def test_build_graph_skipped_when_graph_disabled(tmp_path, monkeypatch):
    called = []
    monkeypatch.setattr("ask_the_docs.index.extract_triplets", lambda *a, **k: called.append(1) or [])
    settings = make_settings(tmp_path, graph_enabled=False)

    build_graph([make_chunk("doc1", 0, "text")], settings)

    assert called == []


def test_build_graph_degrades_when_graph_client_unreachable(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr("ask_the_docs.index.extract_triplets", lambda *a, **k: [Triplet("A", "uses", "B")])
    monkeypatch.setattr("ask_the_docs.index.resolve_entities", lambda triplets, s: triplets)

    def boom(settings):
        raise ServiceUnavailable("down")

    monkeypatch.setattr("ask_the_docs.index.get_graph_client", boom)
    detect_calls = []
    summarize_calls = []
    monkeypatch.setattr(
        "ask_the_docs.index.run_community_detection", lambda d: detect_calls.append(d)
    )
    monkeypatch.setattr(
        "ask_the_docs.index.summarize_communities", lambda d, s: summarize_calls.append(d)
    )
    settings = make_settings(tmp_path, graph_enabled=True)

    with caplog.at_level(logging.WARNING):
        build_graph([make_chunk("doc1", 0, "text")], settings)  # must not raise

    assert any(
        record.name == "ask_the_docs.index" and "graph" in record.message.lower()
        for record in caplog.records
    )
    # No driver was ever obtained, so there's nothing to run community
    # detection/summarization with -- and no second ("Graph build failed")
    # warning from trying to anyway.
    assert detect_calls == []
    assert summarize_calls == []
    assert len(caplog.records) == 1


def test_build_graph_degrades_when_write_graph_raises(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr("ask_the_docs.index.extract_triplets", lambda *a, **k: [Triplet("A", "uses", "B")])
    monkeypatch.setattr("ask_the_docs.index.resolve_entities", lambda triplets, s: triplets)
    monkeypatch.setattr("ask_the_docs.index.get_graph_client", lambda s: "fake-driver")

    def boom(driver, chunk_triplets):
        raise Neo4jError("boom")

    monkeypatch.setattr("ask_the_docs.index.write_graph", boom)
    settings = make_settings(tmp_path, graph_enabled=True)

    with caplog.at_level(logging.WARNING):
        build_graph([make_chunk("doc1", 0, "text")], settings)  # must not raise

    assert any(
        record.name == "ask_the_docs.index" and "graph" in record.message.lower()
        for record in caplog.records
    )


def test_build_graph_degrades_when_extract_triplets_raises(tmp_path, monkeypatch, caplog):
    # A hard failure (e.g. the LLM backend itself unreachable), not just the
    # malformed-output case extract_triplets already degrades internally --
    # must not crash build_graph, same as the narrower Neo4j-specific catch.
    def boom(chunk_text, settings):
        raise OllamaUnavailableError("ollama is down")

    monkeypatch.setattr("ask_the_docs.index.extract_triplets", boom)
    settings = make_settings(tmp_path, graph_enabled=True)

    with caplog.at_level(logging.WARNING):
        build_graph([make_chunk("doc1", 0, "text")], settings)  # must not raise

    assert any(
        record.name == "ask_the_docs.index" and "graph" in record.message.lower()
        for record in caplog.records
    )


def test_build_graph_continues_past_extraction_failure_for_one_chunk(tmp_path, monkeypatch, caplog):
    """A transient LLM failure on one chunk must not abort graph building for every other chunk."""

    def flaky_extract(chunk_text, settings):
        if chunk_text == "doc1 text 2":
            raise OllamaUnavailableError("down")
        return {"doc1 text": [Triplet("A", "uses", "B")], "doc2 text": [Triplet("C", "implements", "D")]}[
            chunk_text
        ]

    written = {}
    monkeypatch.setattr("ask_the_docs.index.extract_triplets", flaky_extract)
    monkeypatch.setattr("ask_the_docs.index.resolve_entities", lambda triplets, s: triplets)
    monkeypatch.setattr("ask_the_docs.index.get_graph_client", lambda s: "fake-driver")
    monkeypatch.setattr(
        "ask_the_docs.index.write_graph", lambda driver, chunk_triplets: written.update(ct=chunk_triplets)
    )
    monkeypatch.setattr("ask_the_docs.index.run_community_detection", lambda driver: None)
    monkeypatch.setattr("ask_the_docs.index.summarize_communities", lambda driver, s: None)

    chunks = [
        make_chunk("doc1", 0, "doc1 text"),
        make_chunk("doc1", 1, "doc1 text 2"),
        make_chunk("doc2", 0, "doc2 text"),
    ]
    with caplog.at_level(logging.WARNING):
        build_graph(chunks, make_settings(tmp_path, graph_enabled=True))

    assert written["ct"] == [
        (Triplet("A", "uses", "B"), "doc1:0"),
        (Triplet("C", "implements", "D"), "doc2:0"),
    ]
    assert any(record.levelname == "WARNING" for record in caplog.records)


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

    monkeypatch.setattr("ask_the_docs.index.extract_triplets", fake_extract)
    monkeypatch.setattr("ask_the_docs.index.resolve_entities", fake_resolve)
    monkeypatch.setattr("ask_the_docs.index.get_graph_client", lambda s: "fake-driver")
    monkeypatch.setattr("ask_the_docs.index.write_graph", fake_write_graph)

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
    monkeypatch.setattr("ask_the_docs.index.extract_triplets", lambda *a, **k: extract_calls.append(1) or [])
    monkeypatch.setattr(
        "ask_the_docs.index.resolve_entities", lambda triplets, s: resolve_calls.append(1) or triplets
    )
    monkeypatch.setattr(
        "ask_the_docs.index.get_graph_client", lambda s: get_client_calls.append(1) or "fake-driver"
    )
    monkeypatch.setattr(
        "ask_the_docs.index.write_graph", lambda driver, chunk_triplets: write_calls.append(chunk_triplets)
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
    monkeypatch.setattr("ask_the_docs.index.extract_triplets", lambda *a, **k: [])
    monkeypatch.setattr(
        "ask_the_docs.index.resolve_entities", lambda triplets, s: resolve_calls.append(1) or triplets
    )
    monkeypatch.setattr(
        "ask_the_docs.index.get_graph_client", lambda s: get_client_calls.append(1) or "fake-driver"
    )
    settings = make_settings(tmp_path, graph_enabled=True)

    build_graph([make_chunk("doc1", 0, "text")], settings)

    assert resolve_calls == []
    assert get_client_calls == []


def test_build_graph_runs_community_detection_and_summarization_after_write(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr("ask_the_docs.index.extract_triplets", lambda *a, **k: [Triplet("A", "uses", "B")])
    monkeypatch.setattr("ask_the_docs.index.resolve_entities", lambda triplets, s: triplets)
    monkeypatch.setattr("ask_the_docs.index.get_graph_client", lambda s: "fake-driver")
    monkeypatch.setattr(
        "ask_the_docs.index.write_graph", lambda *a, **k: calls.append(("write", a[0]))
    )
    monkeypatch.setattr(
        "ask_the_docs.index.run_community_detection", lambda d: calls.append(("detect", d))
    )
    monkeypatch.setattr(
        "ask_the_docs.index.summarize_communities", lambda d, s: calls.append(("summarize", d, s))
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
    monkeypatch.setattr("ask_the_docs.index.get_embeddings", lambda settings: FakeEmbeddings())
    captured = {}
    monkeypatch.setattr(
        "ask_the_docs.index.build_graph",
        lambda chunks, settings: captured.update(chunks=chunks, settings=settings),
    )
    settings = make_settings(tmp_path, graph_enabled=True)
    doc = Document(page_content="word " * 400, metadata={"source": "a.pdf", "source_doc_id": "docA"})

    build_index([doc], settings)

    assert captured["settings"] is settings
    assert [c.metadata["source_doc_id"] for c in captured["chunks"]] == ["docA"] * len(captured["chunks"])


def test_build_index_completes_vector_indexing_when_graph_build_degrades(tmp_path, monkeypatch):
    monkeypatch.setattr("ask_the_docs.index.get_embeddings", lambda settings: FakeEmbeddings())
    monkeypatch.setattr("ask_the_docs.index.extract_triplets", lambda *a, **k: [Triplet("A", "uses", "B")])
    monkeypatch.setattr("ask_the_docs.index.resolve_entities", lambda triplets, s: triplets)

    def boom(settings):
        raise ServiceUnavailable("down")

    monkeypatch.setattr("ask_the_docs.index.get_graph_client", boom)
    settings = make_settings(tmp_path, graph_enabled=True)
    doc = Document(page_content="word " * 400, metadata={"source": "a.pdf", "source_doc_id": "docA"})

    build_index([doc], settings)  # must not raise

    assert get_chunk_count(settings) > 0


def test_build_index_completes_vector_indexing_when_graph_extraction_raises(tmp_path, monkeypatch):
    monkeypatch.setattr("ask_the_docs.index.get_embeddings", lambda settings: FakeEmbeddings())

    def boom(chunk_text, settings):
        raise OllamaUnavailableError("ollama is down")

    monkeypatch.setattr("ask_the_docs.index.extract_triplets", boom)
    settings = make_settings(tmp_path, graph_enabled=True)
    doc = Document(page_content="word " * 400, metadata={"source": "a.pdf", "source_doc_id": "docA"})

    build_index([doc], settings)  # must not raise

    assert get_chunk_count(settings) > 0


def test_build_index_reingesting_a_source_with_no_content_hash_metadata_replaces_it(tmp_path, monkeypatch):
    """Reproduces the reviewer's Critical finding: chunks indexed before
    content-hash dedup existed have no content_hash key at all, not a
    mismatched one -- m["content_hash"] must not KeyError on them."""
    monkeypatch.setattr("ask_the_docs.index.get_embeddings", lambda settings: FakeEmbeddings())
    settings = make_settings(tmp_path)
    legacy_chunk = Document(
        page_content="pre-dedup content",
        metadata={"source_doc_id": "docA", "chunk_index": 0},
    )
    load_index(settings).add_documents([legacy_chunk])
    doc = Document(page_content="fresh content", metadata={"source": "a.pdf", "source_doc_id": "docA"})

    build_index([doc], settings)  # must not raise KeyError

    chunks = get_all_chunks(settings)
    assert all("pre-dedup" not in c.page_content for c in chunks)
    assert all("fresh content" in c.page_content for c in chunks)


def test_get_index_generation_starts_at_zero(tmp_path):
    settings = make_settings(tmp_path)
    assert get_index_generation(settings) == 0


def test_build_index_bumps_generation_on_a_real_write(tmp_path, monkeypatch):
    monkeypatch.setattr("ask_the_docs.index.get_embeddings", lambda settings: FakeEmbeddings())
    settings = make_settings(tmp_path)
    doc = Document(page_content="hello world", metadata={"source": "a.txt", "source_doc_id": "a"})

    build_index([doc], settings)

    assert get_index_generation(settings) == 1


def test_build_index_does_not_bump_generation_on_a_noop_reingest(tmp_path, monkeypatch):
    monkeypatch.setattr("ask_the_docs.index.get_embeddings", lambda settings: FakeEmbeddings())
    settings = make_settings(tmp_path)
    doc = Document(page_content="hello world", metadata={"source": "a.txt", "source_doc_id": "a"})

    build_index([doc], settings)
    build_index([doc], settings)

    assert get_index_generation(settings) == 1


def test_build_index_bumps_generation_on_a_same_chunk_count_replace(tmp_path, monkeypatch):
    """Reproduces the reviewer's Important finding: a content change that
    happens to produce the same number of chunks must still be visible to
    cache-key construction -- chunk count alone can't distinguish it."""
    monkeypatch.setattr("ask_the_docs.index.get_embeddings", lambda settings: FakeEmbeddings())
    settings = make_settings(tmp_path)
    doc_v1 = Document(page_content="hello world", metadata={"source": "a.txt", "source_doc_id": "a"})
    doc_v2 = Document(page_content="hello earth", metadata={"source": "a.txt", "source_doc_id": "a"})

    build_index([doc_v1], settings)
    gen_after_first = get_index_generation(settings)
    build_index([doc_v2], settings)

    assert get_chunk_count(settings) == 1
    assert get_index_generation(settings) == gen_after_first + 1


def test_build_index_writes_large_inserts_in_bounded_batches(tmp_path, monkeypatch):
    """Reproduces the reviewer's Important finding: build_index used to go
    through Chroma.from_documents, which internally batches via
    create_batches; load_index().add_documents() doesn't. A large ingest
    must be chunked here instead, or a single oversized embed request
    risks an Ollama timeout and a single upsert risks exceeding Chroma's
    batch limit."""
    call_sizes = []

    class RecordingEmbeddings(Embeddings):
        def embed_documents(self, texts):
            call_sizes.append(len(texts))
            return [_vector(t) for t in texts]

        def embed_query(self, text):
            return _vector(text)

    monkeypatch.setattr("ask_the_docs.index.get_embeddings", lambda settings: RecordingEmbeddings())
    settings = make_settings(tmp_path)
    docs = [
        Document(
            page_content=f"unique content block number {i} " * 5,
            metadata={"source": f"{i}.txt", "source_doc_id": f"d{i}"},
        )
        for i in range(100)
    ]

    build_index(docs, settings)

    assert len(call_sizes) > 1
    assert all(size <= INDEX_WRITE_BATCH_SIZE for size in call_sizes)
    assert sum(call_sizes) == 100


def test_build_index_deletes_chunks_for_a_source_that_becomes_empty(tmp_path, monkeypatch):
    """Reproduces the reviewer's Important finding: a source whose new
    content chunks to zero (e.g. a PDF replaced by a scanned version with
    no extractable text) never appears in the per-source chunk grouping,
    so its old chunks were never deleted. The spec says changed content
    means delete all existing chunks for that source_doc_id -- "no new
    chunks" is a change too."""
    monkeypatch.setattr("ask_the_docs.index.get_embeddings", lambda settings: FakeEmbeddings())
    settings = make_settings(tmp_path)
    doc_v1 = Document(page_content="real extracted text", metadata={"source": "a.pdf", "source_doc_id": "docA"})
    build_index([doc_v1], settings)

    doc_v2_now_empty = Document(page_content="", metadata={"source": "a.pdf", "source_doc_id": "docA"})
    build_index([doc_v2_now_empty], settings)

    assert get_all_chunks(settings) == []
