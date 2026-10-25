import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from research_assistant.config import ProviderConfig, Settings
from research_assistant.index import build_index
from research_assistant.retrieve import hybrid_merge, retrieve


def make_settings(tmp_path) -> Settings:
    # query_rewrite/rerank disabled here: they involve a real LLM call and a
    # real CrossEncoder model load, which these tests aren't exercising --
    # Task 7's own tests below cover the rewrite-fallback and rerank paths
    # with mocks instead.
    return Settings(
        generator=ProviderConfig(provider="ollama", model="llama3.1:8b"),
        judge=ProviderConfig(provider="anthropic", model="claude-sonnet-5-5", api_key="x"),
        index_dir=tmp_path / "index",
        query_rewrite_enabled=False,
        rerank_enabled=False,
    )


def _doc(doc_id: str, chunk_index: int = 0) -> Document:
    return Document(
        page_content=doc_id,
        metadata={"source_doc_id": doc_id, "chunk_index": chunk_index, "source": f"{doc_id}.txt"},
    )


class FakeCache:
    """retrieve() reaches bm25_index.get_bm25_index(), which calls
    get_cache_client() -- without this, these tests would try a real Valkey
    connection that doesn't exist in the test environment."""

    def __init__(self):
        self._store: dict[str, bytes] = {}

    def get(self, key):
        return self._store.get(key)

    def set(self, key, value, ex=None):
        self._store[key] = value


VECTORS = {
    "apples are red fruit": [1.0, 0.0],
    "bananas are yellow fruit": [0.9, 0.1],
    "rockets launch into space": [0.0, 1.0],
}


class FakeEmbeddings(Embeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [VECTORS[t] for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return VECTORS[text]


def test_retrieve_returns_empty_list_when_index_not_built(tmp_path, monkeypatch):
    monkeypatch.setattr("research_assistant.index.get_embeddings", lambda settings: FakeEmbeddings())
    monkeypatch.setattr("research_assistant.bm25_index.get_cache_client", lambda settings: FakeCache())
    settings = make_settings(tmp_path)

    assert retrieve("apples are red fruit", settings) == []


def test_retrieve_returns_top_k_ordered_by_relevance(tmp_path, monkeypatch):
    monkeypatch.setattr("research_assistant.index.get_embeddings", lambda settings: FakeEmbeddings())
    monkeypatch.setattr("research_assistant.bm25_index.get_cache_client", lambda settings: FakeCache())
    settings = make_settings(tmp_path)

    docs = [
        Document(page_content="apples are red fruit", metadata={"source": "a.txt", "source_doc_id": "a"}),
        Document(
            page_content="bananas are yellow fruit", metadata={"source": "b.txt", "source_doc_id": "b"}
        ),
        Document(
            page_content="rockets launch into space", metadata={"source": "c.txt", "source_doc_id": "c"}
        ),
    ]
    build_index(docs, settings)

    results = retrieve("apples are red fruit", settings, k=2)

    assert len(results) == 2
    assert results[0].page_content == "apples are red fruit"
    assert results[1].page_content == "bananas are yellow fruit"


def test_hybrid_merge_keeps_vector_only_and_bm25_only_matches():
    doc_a, doc_b, doc_c = _doc("a"), _doc("b"), _doc("c")
    vector_hits = [(doc_a, 0.9), (doc_c, 0.5)]
    bm25_hits = [(doc_b, 10.0), (doc_c, 8.0)]

    merged = hybrid_merge(vector_hits, bm25_hits, alpha=0.5)

    ids = {doc.metadata["source_doc_id"] for doc, _ in merged}
    assert ids == {"a", "b", "c"}


def test_hybrid_merge_scores_a_vector_only_match_with_zero_bm25():
    doc_a = _doc("a")
    # doc_a is the only candidate on the vector side; with one candidate,
    # min==max so its normalized vector score is 1.0 (the "all-equal" rule).
    # The missing BM25 side must contribute exactly 0.0, not get dropped
    # and not get treated as if it tied for best.
    merged = hybrid_merge([(doc_a, 0.9)], [], alpha=0.5)

    assert merged == [(doc_a, 0.5)]


def test_hybrid_merge_normalizes_each_side_independently():
    doc_a, doc_b = _doc("a"), _doc("b")
    # Vector side spans [0, 10]; BM25 side spans [0, 100] -- wildly different
    # raw scales that must not be compared directly.
    vector_hits = [(doc_a, 10.0), (doc_b, 0.0)]
    bm25_hits = [(doc_a, 0.0), (doc_b, 100.0)]

    merged = hybrid_merge(vector_hits, bm25_hits, alpha=0.5)

    scores = {doc.metadata["source_doc_id"]: s for doc, s in merged}
    # doc_a: vector norm 1.0, bm25 norm 0.0 -> blended 0.5
    # doc_b: vector norm 0.0, bm25 norm 1.0 -> blended 0.5
    assert scores["a"] == pytest.approx(0.5)
    assert scores["b"] == pytest.approx(0.5)


def test_hybrid_merge_handles_all_equal_scores_without_dividing_by_zero():
    doc_a, doc_b = _doc("a"), _doc("b")

    merged = hybrid_merge([(doc_a, 1.0), (doc_b, 1.0)], [], alpha=0.5)

    assert len(merged) == 2


def test_retrieve_falls_back_to_raw_query_when_rewrite_fails(tmp_path, monkeypatch):
    monkeypatch.setattr("research_assistant.index.get_embeddings", lambda settings: FakeEmbeddings())
    monkeypatch.setattr("research_assistant.bm25_index.get_cache_client", lambda settings: FakeCache())

    def failing_rewrite(query, settings):
        raise RuntimeError("ollama unreachable")

    monkeypatch.setattr("research_assistant.retrieve.rewrite_query", failing_rewrite)

    settings = Settings(
        generator=ProviderConfig(provider="ollama", model="llama3.1:8b"),
        judge=ProviderConfig(provider="anthropic", model="claude-sonnet-5-5", api_key="x"),
        index_dir=tmp_path / "index",
        query_rewrite_enabled=True,
        rerank_enabled=False,
    )
    docs = [Document(page_content="apples are red fruit", metadata={"source": "a.txt", "source_doc_id": "a"})]
    build_index(docs, settings)

    results = retrieve("apples are red fruit", settings, k=1)

    assert len(results) == 1
    assert results[0].page_content == "apples are red fruit"
