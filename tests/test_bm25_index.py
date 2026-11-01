import logging
from unittest.mock import Mock

from langchain_core.documents import Document

from ask_the_docs.bm25_index import get_bm25_index, score
from ask_the_docs.config import ProviderConfig, Settings

FIXTURE_CHUNKS = [
    Document(
        page_content="transformer attention mechanism explained in detail",
        metadata={"source": "a.pdf", "source_doc_id": "a", "chunk_index": 0},
    ),
    Document(
        page_content="gardening tips for spring flowers and soil",
        metadata={"source": "b.pdf", "source_doc_id": "b", "chunk_index": 0},
    ),
    # Third, unrelated doc: with only 2 docs, BM25's IDF formula degenerates
    # to exactly 0 when a term's document frequency is precisely half the
    # corpus (log(1.5) - log(1.5) == 0) -- this breaks that tie.
    Document(
        page_content="recipe for baking sourdough bread at home",
        metadata={"source": "c.pdf", "source_doc_id": "c", "chunk_index": 0},
    ),
]


class FakeCache:
    def __init__(self):
        self._store: dict[str, bytes] = {}

    def get(self, key):
        return self._store.get(key)

    def set(self, key, value, ex=None):
        self._store[key] = value


def make_settings(tmp_path) -> Settings:
    return Settings(
        generator=ProviderConfig(provider="ollama", model="llama3.1:8b"),
        judge=ProviderConfig(provider="anthropic", model="claude-sonnet-5-5", api_key="x"),
        index_dir=tmp_path / "index",
    )


def test_score_matches_keyword_despite_surrounding_punctuation(tmp_path, monkeypatch):
    chunks = [
        Document(
            page_content="LoRA: Low-Rank Adaptation of Large Language Models.",
            metadata={"source": "lora.pdf", "source_doc_id": "lora", "chunk_index": 0},
        ),
        Document(
            page_content="recipe for baking sourdough bread at home",
            metadata={"source": "c.pdf", "source_doc_id": "c", "chunk_index": 0},
        ),
        Document(
            page_content="gardening tips for spring flowers and soil",
            metadata={"source": "b.pdf", "source_doc_id": "b", "chunk_index": 0},
        ),
    ]
    settings = make_settings(tmp_path)
    monkeypatch.setattr("ask_the_docs.bm25_index.get_all_chunks", lambda s: chunks)
    monkeypatch.setattr("ask_the_docs.bm25_index.get_index_generation", lambda s: len(chunks))
    monkeypatch.setattr("ask_the_docs.bm25_index.get_cache_client", lambda s: FakeCache())

    scores = score("What is LoRA?", settings, k=10)

    by_source = {doc.metadata["source"]: s for doc, s in scores}
    assert by_source["lora.pdf"] > by_source["c.pdf"]
    assert by_source["lora.pdf"] > by_source["b.pdf"]


def test_score_ranks_keyword_matches_higher(tmp_path, monkeypatch):
    settings = make_settings(tmp_path)
    monkeypatch.setattr("ask_the_docs.bm25_index.get_all_chunks", lambda s: FIXTURE_CHUNKS)
    monkeypatch.setattr("ask_the_docs.bm25_index.get_index_generation", lambda s: len(FIXTURE_CHUNKS))
    monkeypatch.setattr("ask_the_docs.bm25_index.get_cache_client", lambda s: FakeCache())

    scores = score("transformer attention", settings, k=10)

    by_source = {doc.metadata["source"]: s for doc, s in scores}
    assert by_source["a.pdf"] > by_source["b.pdf"]


def test_get_bm25_index_skips_rebuild_on_cache_hit(tmp_path, monkeypatch):
    settings = make_settings(tmp_path)
    calls = {"count": 0}

    def fake_get_all_chunks(s):
        calls["count"] += 1
        return FIXTURE_CHUNKS

    shared_cache = FakeCache()
    monkeypatch.setattr("ask_the_docs.bm25_index.get_all_chunks", fake_get_all_chunks)
    monkeypatch.setattr("ask_the_docs.bm25_index.get_index_generation", lambda s: 2)
    monkeypatch.setattr("ask_the_docs.bm25_index.get_cache_client", lambda s: shared_cache)

    get_bm25_index(settings)
    get_bm25_index(settings)

    assert calls["count"] == 1


def test_get_bm25_index_rebuilds_when_cached_value_is_corrupted(tmp_path, monkeypatch):
    settings = make_settings(tmp_path)
    calls = {"count": 0}

    def fake_get_all_chunks(s):
        calls["count"] += 1
        return FIXTURE_CHUNKS

    class CorruptedCache:
        def get(self, key):
            return b"not a valid pickle"

        def set(self, key, value, ex=None):
            pass

    monkeypatch.setattr("ask_the_docs.bm25_index.get_all_chunks", fake_get_all_chunks)
    monkeypatch.setattr("ask_the_docs.bm25_index.get_index_generation", lambda s: 3)
    monkeypatch.setattr("ask_the_docs.bm25_index.get_cache_client", lambda s: CorruptedCache())

    bm25, chunks = get_bm25_index(settings)

    assert calls["count"] == 1
    assert chunks == FIXTURE_CHUNKS


def test_score_returns_empty_list_for_an_empty_corpus(tmp_path, monkeypatch):
    settings = make_settings(tmp_path)
    monkeypatch.setattr("ask_the_docs.bm25_index.get_all_chunks", lambda s: [])
    monkeypatch.setattr("ask_the_docs.bm25_index.get_index_generation", lambda s: 0)
    monkeypatch.setattr("ask_the_docs.bm25_index.get_cache_client", lambda s: FakeCache())

    assert score("anything", settings, k=10) == []


def test_score_clamps_k_to_corpus_size(tmp_path, monkeypatch):
    """bm25s.BM25.retrieve() raises ValueError if k exceeds the corpus size
    -- score() must clamp instead of propagating that crash."""
    settings = make_settings(tmp_path)
    monkeypatch.setattr("ask_the_docs.bm25_index.get_all_chunks", lambda s: FIXTURE_CHUNKS)
    monkeypatch.setattr("ask_the_docs.bm25_index.get_index_generation", lambda s: 1)
    monkeypatch.setattr("ask_the_docs.bm25_index.get_cache_client", lambda s: FakeCache())

    results = score("transformer attention", settings, k=1000)

    assert len(results) == len(FIXTURE_CHUNKS)


def test_score_respects_k(tmp_path, monkeypatch):
    settings = make_settings(tmp_path)
    monkeypatch.setattr("ask_the_docs.bm25_index.get_all_chunks", lambda s: FIXTURE_CHUNKS)
    monkeypatch.setattr("ask_the_docs.bm25_index.get_index_generation", lambda s: 1)
    monkeypatch.setattr("ask_the_docs.bm25_index.get_cache_client", lambda s: FakeCache())

    results = score("transformer attention", settings, k=1)

    assert len(results) == 1


def test_score_returns_empty_list_for_an_empty_corpus_with_bm25s(tmp_path, monkeypatch):
    """bm25s.BM25().index([]) raises ValueError (verified) -- the empty-
    corpus guard must run before .index() is ever called, not rely on the
    library degrading gracefully."""
    settings = make_settings(tmp_path)
    monkeypatch.setattr("ask_the_docs.bm25_index.get_all_chunks", lambda s: [])
    monkeypatch.setattr("ask_the_docs.bm25_index.get_index_generation", lambda s: 0)
    monkeypatch.setattr("ask_the_docs.bm25_index.get_cache_client", lambda s: FakeCache())

    assert score("anything", settings, k=10) == []


def test_get_bm25_index_rebuilds_when_index_generation_changes_even_with_same_chunk_count(tmp_path, monkeypatch):
    """A content change that happens to produce the same number of chunks
    (e.g. a same-length edit) must still bust this cache -- chunk count
    alone can't distinguish it, which is why the cache key uses the
    index generation (bumped on every real write) instead."""
    settings = make_settings(tmp_path)
    calls = {"count": 0}
    # get_index_generation is now read twice per get_bm25_index() call (once
    # for the in-process cache key, once inside _rebuild_bm25_index for the
    # Valkey key) -- the first two reads (both within the first call) return
    # 2, the next two (within the second call) return 3.
    generation_reads = {"n": 0}

    def fake_get_index_generation(s):
        generation_reads["n"] += 1
        return 2 if generation_reads["n"] <= 2 else 3

    def fake_get_all_chunks(s):
        calls["count"] += 1
        return FIXTURE_CHUNKS

    shared_cache = FakeCache()
    monkeypatch.setattr("ask_the_docs.bm25_index.get_all_chunks", fake_get_all_chunks)
    monkeypatch.setattr("ask_the_docs.bm25_index.get_index_generation", fake_get_index_generation)
    monkeypatch.setattr("ask_the_docs.bm25_index.get_cache_client", lambda s: shared_cache)

    get_bm25_index(settings)
    get_bm25_index(settings)

    assert calls["count"] == 2


def test_get_bm25_index_is_resident_in_process_across_calls(tmp_path, monkeypatch):
    """Distinct from the existing Valkey-cache tests above: this proves no
    cache_get() round trip happens at all on the second call, not just that
    get_all_chunks() is skipped."""
    settings = make_settings(tmp_path)
    monkeypatch.setattr("ask_the_docs.bm25_index.get_all_chunks", lambda s: FIXTURE_CHUNKS)
    monkeypatch.setattr("ask_the_docs.bm25_index.get_index_generation", lambda s: 1)
    mock_cache_get = Mock(return_value=None)
    monkeypatch.setattr("ask_the_docs.bm25_index.cache_get", mock_cache_get)
    monkeypatch.setattr("ask_the_docs.bm25_index.cache_set", Mock())
    monkeypatch.setattr("ask_the_docs.bm25_index.get_cache_client", lambda s: FakeCache())

    get_bm25_index(settings)
    get_bm25_index(settings)

    assert mock_cache_get.call_count == 1


def test_get_bm25_index_does_not_collide_across_different_index_dirs_at_the_same_generation(tmp_path, monkeypatch):
    monkeypatch.setattr("ask_the_docs.bm25_index.get_all_chunks", lambda s: FIXTURE_CHUNKS)
    monkeypatch.setattr("ask_the_docs.bm25_index.get_index_generation", lambda s: 0)
    monkeypatch.setattr("ask_the_docs.bm25_index.get_cache_client", lambda s: FakeCache())
    settings_a = make_settings(tmp_path / "a")
    settings_b = make_settings(tmp_path / "b")

    first_bm25, _ = get_bm25_index(settings_a)
    second_bm25, _ = get_bm25_index(settings_b)

    assert first_bm25 is not second_bm25


def test_get_bm25_index_degrades_to_no_results_when_chroma_fails(tmp_path, monkeypatch, caplog):
    # Deliberately does not mock get_index_generation: the cache key must
    # be computable (it reads a local file, not Chroma) without going near
    # the broken get_all_chunks below, so this proves the fail-open fires
    # for a real Chroma outage, not just a mocked-around one.
    monkeypatch.setattr("ask_the_docs.bm25_index.get_cache_client", lambda s: FakeCache())

    def boom(settings):
        raise RuntimeError("chroma unavailable")

    monkeypatch.setattr("ask_the_docs.bm25_index.get_all_chunks", boom)

    with caplog.at_level(logging.WARNING):
        bm25, chunks = get_bm25_index(make_settings(tmp_path))

    assert bm25 is None
    assert chunks == []
    assert any("chroma" in record.message.lower() or "bm25" in record.message.lower() for record in caplog.records)
