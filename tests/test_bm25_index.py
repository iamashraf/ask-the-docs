from langchain_core.documents import Document

from research_assistant.bm25_index import get_bm25_index, score
from research_assistant.config import ProviderConfig, Settings

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
    monkeypatch.setattr("research_assistant.bm25_index.get_all_chunks", lambda s: chunks)
    monkeypatch.setattr("research_assistant.bm25_index.get_chunk_count", lambda s: len(chunks))
    monkeypatch.setattr("research_assistant.bm25_index.get_cache_client", lambda s: FakeCache())

    scores = score("What is LoRA?", settings)

    by_source = {doc.metadata["source"]: s for doc, s in scores}
    assert by_source["lora.pdf"] > by_source["c.pdf"]
    assert by_source["lora.pdf"] > by_source["b.pdf"]


def test_score_ranks_keyword_matches_higher(tmp_path, monkeypatch):
    settings = make_settings(tmp_path)
    monkeypatch.setattr("research_assistant.bm25_index.get_all_chunks", lambda s: FIXTURE_CHUNKS)
    monkeypatch.setattr("research_assistant.bm25_index.get_chunk_count", lambda s: len(FIXTURE_CHUNKS))
    monkeypatch.setattr("research_assistant.bm25_index.get_cache_client", lambda s: FakeCache())

    scores = score("transformer attention", settings)

    by_source = {doc.metadata["source"]: s for doc, s in scores}
    assert by_source["a.pdf"] > by_source["b.pdf"]


def test_get_bm25_index_skips_rebuild_on_cache_hit(tmp_path, monkeypatch):
    settings = make_settings(tmp_path)
    calls = {"count": 0}

    def fake_get_all_chunks(s):
        calls["count"] += 1
        return FIXTURE_CHUNKS

    shared_cache = FakeCache()
    monkeypatch.setattr("research_assistant.bm25_index.get_all_chunks", fake_get_all_chunks)
    monkeypatch.setattr("research_assistant.bm25_index.get_chunk_count", lambda s: 2)
    monkeypatch.setattr("research_assistant.bm25_index.get_cache_client", lambda s: shared_cache)

    get_bm25_index(settings)
    get_bm25_index(settings)

    assert calls["count"] == 1


def test_score_returns_empty_list_for_an_empty_corpus(tmp_path, monkeypatch):
    settings = make_settings(tmp_path)
    monkeypatch.setattr("research_assistant.bm25_index.get_all_chunks", lambda s: [])
    monkeypatch.setattr("research_assistant.bm25_index.get_chunk_count", lambda s: 0)
    monkeypatch.setattr("research_assistant.bm25_index.get_cache_client", lambda s: FakeCache())

    assert score("anything", settings) == []


def test_get_bm25_index_rebuilds_when_live_chunk_count_changes(tmp_path, monkeypatch):
    settings = make_settings(tmp_path)
    calls = {"count": 0}
    counts = [2, 3]

    def fake_get_all_chunks(s):
        calls["count"] += 1
        return FIXTURE_CHUNKS

    shared_cache = FakeCache()
    monkeypatch.setattr("research_assistant.bm25_index.get_all_chunks", fake_get_all_chunks)
    monkeypatch.setattr("research_assistant.bm25_index.get_chunk_count", lambda s: counts.pop(0))
    monkeypatch.setattr("research_assistant.bm25_index.get_cache_client", lambda s: shared_cache)

    get_bm25_index(settings)
    get_bm25_index(settings)

    assert calls["count"] == 2
