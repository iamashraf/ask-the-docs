import logging

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from neo4j.exceptions import SessionExpired
from prometheus_client import REGISTRY

from ask_the_docs.config import ProviderConfig, Settings
from ask_the_docs.index import build_index
from ask_the_docs.query_analyzer import QueryClassification
from ask_the_docs.retrieve import hybrid_merge, retrieve


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


def make_chunk(source_doc_id: str, chunk_index: int, text: str) -> Document:
    return Document(page_content=text, metadata={"source_doc_id": source_doc_id, "chunk_index": chunk_index})


class FakeSpan:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


class FakeTracer:
    def __init__(self):
        self.spans: list[str] = []

    def start_as_current_span(self, name, **kwargs):
        self.spans.append(name)
        return FakeSpan()


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
    monkeypatch.setattr("ask_the_docs.index.get_embeddings", lambda settings: FakeEmbeddings())
    monkeypatch.setattr("ask_the_docs.bm25_index.get_cache_client", lambda settings: FakeCache())
    settings = make_settings(tmp_path)

    assert retrieve("apples are red fruit", settings) == []


def test_retrieve_returns_top_k_ordered_by_relevance(tmp_path, monkeypatch):
    monkeypatch.setattr("ask_the_docs.index.get_embeddings", lambda settings: FakeEmbeddings())
    monkeypatch.setattr("ask_the_docs.bm25_index.get_cache_client", lambda settings: FakeCache())
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


def test_retrieve_truncates_merged_candidates_to_pool_size_before_rerank(tmp_path, monkeypatch):
    # 3 vector-only docs + 3 disjoint bm25-only docs = 6 merged candidates,
    # with candidate_pool_size=2 only the top 2 should reach the reranker.
    vector_docs = [_doc(f"v{i}") for i in range(3)]
    bm25_docs = [_doc(f"b{i}") for i in range(3)]

    class FakeIndex:
        def similarity_search_with_score(self, query, k):
            return [(doc, 0.1 * (i + 1)) for i, doc in enumerate(vector_docs)]

    monkeypatch.setattr("ask_the_docs.retrieve.load_index", lambda settings: FakeIndex())
    monkeypatch.setattr(
        "ask_the_docs.retrieve.bm25_score",
        lambda query, settings: [(doc, 1.0) for doc in bm25_docs],
    )

    captured = {}

    class CapturingReranker:
        def rerank(self, query, candidates, k):
            captured["candidates"] = candidates
            return candidates[:k]

    monkeypatch.setattr("ask_the_docs.retrieve.get_reranker", lambda model_name: CapturingReranker())

    settings = make_settings(tmp_path).model_copy(update={"rerank_enabled": True, "candidate_pool_size": 2})
    retrieve("anything", settings, k=2)

    assert len(captured["candidates"]) == 2


def test_retrieve_excludes_zero_score_bm25_hits_from_merge(tmp_path, monkeypatch):
    # Simulates an out-of-vocabulary query: BM25 finds no real keyword
    # match for any chunk, so every chunk ties at score 0.0. Those must be
    # excluded, not treated as "all equal -> normalize to 1.0" and tie the
    # best vector match, crowding out real vector hits (rerank_enabled is
    # False here -- that's exactly the comparison toggle this would bite).
    doc_v1, doc_v2 = _doc("v1"), _doc("v2")
    noise1, noise2, noise3 = _doc("n1"), _doc("n2"), _doc("n3")

    class FakeIndex:
        def similarity_search_with_score(self, query, k):
            return [(doc_v1, 0.1), (doc_v2, 0.2)]

    monkeypatch.setattr("ask_the_docs.retrieve.load_index", lambda settings: FakeIndex())
    monkeypatch.setattr(
        "ask_the_docs.retrieve.bm25_score",
        lambda query, settings: [(noise1, 0.0), (noise2, 0.0), (noise3, 0.0)],
    )

    settings = make_settings(tmp_path)
    results = retrieve("out of vocabulary gibberish", settings, k=5)

    result_ids = {d.metadata["source_doc_id"] for d in results}
    assert result_ids == {"v1", "v2"}


def test_retrieve_falls_back_to_raw_query_when_rewrite_fails(tmp_path, monkeypatch):
    monkeypatch.setattr("ask_the_docs.index.get_embeddings", lambda settings: FakeEmbeddings())
    monkeypatch.setattr("ask_the_docs.bm25_index.get_cache_client", lambda settings: FakeCache())

    def failing_rewrite(query, settings):
        raise RuntimeError("ollama unreachable")

    monkeypatch.setattr("ask_the_docs.retrieve.rewrite_query", failing_rewrite)

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


def test_retrieve_skips_classification_when_graph_disabled(tmp_path, monkeypatch):
    called = []
    monkeypatch.setattr("ask_the_docs.retrieve.classify_query", lambda *a, **k: called.append(1))

    doc = _doc("a")

    class FakeIndex:
        def similarity_search_with_score(self, query, k):
            return [(doc, 0.1)]

    monkeypatch.setattr("ask_the_docs.retrieve.load_index", lambda settings: FakeIndex())
    monkeypatch.setattr("ask_the_docs.retrieve.bm25_score", lambda query, settings: [])

    settings = make_settings(tmp_path).model_copy(update={"graph_enabled": False})
    retrieve("a query", settings)

    assert called == []  # existing hybrid-only behavior, untouched


def test_retrieve_returns_global_search_results_directly(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "ask_the_docs.retrieve.classify_query",
        lambda q, s: QueryClassification(label="global", entities=[]),
    )
    fake_docs = [make_chunk("community-0", 0, "a theme summary")]
    monkeypatch.setattr("ask_the_docs.retrieve.global_search", lambda q, s: fake_docs)
    # hybrid must never be reached when global search succeeds
    monkeypatch.setattr(
        "ask_the_docs.retrieve.load_index",
        lambda settings: pytest.fail("hybrid path must be bypassed when global search succeeds"),
    )

    settings = make_settings(tmp_path).model_copy(update={"graph_enabled": True})
    result = retrieve("what are the themes?", settings, k=5)

    assert result == fake_docs


def test_retrieve_opens_a_hybrid_retrieve_span(tmp_path, monkeypatch):
    monkeypatch.setattr("ask_the_docs.index.get_embeddings", lambda settings: FakeEmbeddings())
    monkeypatch.setattr("ask_the_docs.bm25_index.get_cache_client", lambda settings: FakeCache())
    fake_tracer = FakeTracer()
    monkeypatch.setattr("ask_the_docs.retrieve.tracer", fake_tracer)
    settings = make_settings(tmp_path)
    build_index(
        [Document(page_content="apples are red fruit", metadata={"source": "a.txt", "source_doc_id": "a"})],
        settings,
    )

    retrieve("apples are red fruit", settings, k=1)

    assert "hybrid_retrieve" in fake_tracer.spans


def test_retrieve_does_not_open_hybrid_retrieve_span_on_the_global_search_short_circuit(tmp_path, monkeypatch):
    """Review Focus: the early-return graph path must not open spans for
    code that never runs."""
    monkeypatch.setattr(
        "ask_the_docs.retrieve.classify_query",
        lambda q, s: QueryClassification(label="global", entities=[]),
    )
    monkeypatch.setattr(
        "ask_the_docs.retrieve.global_search", lambda q, s: [make_chunk("community-0", 0, "a theme summary")]
    )
    fake_tracer = FakeTracer()
    monkeypatch.setattr("ask_the_docs.retrieve.tracer", fake_tracer)
    settings = make_settings(tmp_path).model_copy(update={"graph_enabled": True})

    retrieve("what are the themes?", settings, k=5)

    assert "hybrid_retrieve" not in fake_tracer.spans
    assert "rerank" not in fake_tracer.spans


def test_retrieve_merges_local_search_into_candidate_pool(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "ask_the_docs.retrieve.classify_query",
        lambda q, s: QueryClassification(label="local", entities=["BM25"]),
    )
    graph_only_doc = make_chunk("doc9", 0, "only found via graph traversal")
    monkeypatch.setattr("ask_the_docs.retrieve.local_search", lambda entities, s: [graph_only_doc])

    class EmptyIndex:
        def similarity_search_with_score(self, query, k):
            return []

    monkeypatch.setattr("ask_the_docs.retrieve.load_index", lambda settings: EmptyIndex())
    monkeypatch.setattr("ask_the_docs.retrieve.bm25_score", lambda query, settings: [])

    settings = make_settings(tmp_path).model_copy(update={"graph_enabled": True})
    result = retrieve("how does BM25 relate to vectors?", settings, k=5)

    assert graph_only_doc in result


def test_retrieve_falls_back_to_hybrid_when_global_search_finds_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "ask_the_docs.retrieve.classify_query",
        lambda q, s: QueryClassification(label="global", entities=[]),
    )
    monkeypatch.setattr("ask_the_docs.retrieve.global_search", lambda q, s: [])

    hybrid_doc = _doc("hybrid-hit")

    class FakeIndex:
        def similarity_search_with_score(self, query, k):
            return [(hybrid_doc, 0.1)]

    monkeypatch.setattr("ask_the_docs.retrieve.load_index", lambda settings: FakeIndex())
    monkeypatch.setattr("ask_the_docs.retrieve.bm25_score", lambda query, settings: [])

    settings = make_settings(tmp_path).model_copy(update={"graph_enabled": True})
    result = retrieve("what are the themes?", settings, k=5)

    assert result == [hybrid_doc]  # must not return [] just because global_search found nothing


def test_retrieve_falls_back_to_hybrid_when_global_search_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "ask_the_docs.retrieve.classify_query",
        lambda q, s: QueryClassification(label="global", entities=[]),
    )

    def failing_global_search(query, settings):
        raise RuntimeError("neo4j unreachable")

    monkeypatch.setattr("ask_the_docs.retrieve.global_search", failing_global_search)

    hybrid_doc = _doc("hybrid-hit")

    class FakeIndex:
        def similarity_search_with_score(self, query, k):
            return [(hybrid_doc, 0.1)]

    monkeypatch.setattr("ask_the_docs.retrieve.load_index", lambda settings: FakeIndex())
    monkeypatch.setattr("ask_the_docs.retrieve.bm25_score", lambda query, settings: [])

    settings = make_settings(tmp_path).model_copy(update={"graph_enabled": True})
    # must not propagate -- a crashing global_search must degrade to hybrid, not crash the request
    result = retrieve("what are the themes?", settings, k=5)

    assert result == [hybrid_doc]


def test_retrieve_falls_back_to_hybrid_when_classification_raises(tmp_path, monkeypatch):
    def failing_classify(query, settings):
        raise RuntimeError("ollama unreachable")

    monkeypatch.setattr("ask_the_docs.retrieve.classify_query", failing_classify)

    hybrid_doc = _doc("hybrid-hit")

    class FakeIndex:
        def similarity_search_with_score(self, query, k):
            return [(hybrid_doc, 0.1)]

    monkeypatch.setattr("ask_the_docs.retrieve.load_index", lambda settings: FakeIndex())
    monkeypatch.setattr("ask_the_docs.retrieve.bm25_score", lambda query, settings: [])

    settings = make_settings(tmp_path).model_copy(update={"graph_enabled": True})
    result = retrieve("anything", settings, k=5)

    assert result == [hybrid_doc]


def test_retrieve_dedupes_local_search_against_existing_hybrid_candidate(tmp_path, monkeypatch):
    shared_doc = _doc("shared")

    monkeypatch.setattr(
        "ask_the_docs.retrieve.classify_query",
        lambda q, s: QueryClassification(label="local", entities=["BM25"]),
    )
    monkeypatch.setattr("ask_the_docs.retrieve.local_search", lambda entities, s: [shared_doc])

    class FakeIndex:
        def similarity_search_with_score(self, query, k):
            return [(shared_doc, 0.1)]

    monkeypatch.setattr("ask_the_docs.retrieve.load_index", lambda settings: FakeIndex())
    monkeypatch.setattr("ask_the_docs.retrieve.bm25_score", lambda query, settings: [])

    settings = make_settings(tmp_path).model_copy(update={"graph_enabled": True})
    result = retrieve("how does BM25 relate to vectors?", settings, k=5)

    assert result.count(shared_doc) == 1  # hybrid already found it -- local_search must not duplicate it


def test_retrieve_keeps_local_search_match_when_hybrid_pool_is_already_full(tmp_path, monkeypatch):
    # vector_hits alone fills candidate_pool_size on any corpus >= that size,
    # so hybrid_merge's output is already at the pool-size limit before the
    # local_search union happens. The graph-only doc must still survive.
    pool_size = 3
    hybrid_docs = [_doc(f"hybrid{i}") for i in range(pool_size)]
    graph_only_doc = make_chunk("doc9", 0, "only found via graph traversal")

    monkeypatch.setattr(
        "ask_the_docs.retrieve.classify_query",
        lambda q, s: QueryClassification(label="local", entities=["BM25"]),
    )
    monkeypatch.setattr("ask_the_docs.retrieve.local_search", lambda entities, s: [graph_only_doc])

    class FakeIndex:
        def similarity_search_with_score(self, query, k):
            return [(doc, 0.1 * (i + 1)) for i, doc in enumerate(hybrid_docs)]

    monkeypatch.setattr("ask_the_docs.retrieve.load_index", lambda settings: FakeIndex())
    monkeypatch.setattr("ask_the_docs.retrieve.bm25_score", lambda query, settings: [])

    settings = make_settings(tmp_path).model_copy(update={"graph_enabled": True, "candidate_pool_size": pool_size})
    result = retrieve("how does BM25 relate to vectors?", settings, k=10)

    assert graph_only_doc in result


def test_retrieve_falls_back_to_hybrid_when_local_search_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "ask_the_docs.retrieve.classify_query",
        lambda q, s: QueryClassification(label="local", entities=["BM25"]),
    )

    def failing_local_search(entities, settings):
        raise SessionExpired("connection dropped mid-session")

    monkeypatch.setattr("ask_the_docs.retrieve.local_search", failing_local_search)

    hybrid_doc = _doc("hybrid-hit")

    class FakeIndex:
        def similarity_search_with_score(self, query, k):
            return [(hybrid_doc, 0.1)]

    monkeypatch.setattr("ask_the_docs.retrieve.load_index", lambda settings: FakeIndex())
    monkeypatch.setattr("ask_the_docs.retrieve.bm25_score", lambda query, settings: [])

    settings = make_settings(tmp_path).model_copy(update={"graph_enabled": True})
    # must not propagate -- a crashing local_search must degrade to hybrid, not crash the request
    result = retrieve("how does BM25 relate to vectors?", settings, k=5)

    assert result == [hybrid_doc]


def test_retrieve_keeps_local_search_match_in_top_k_when_rerank_disabled(tmp_path, monkeypatch):
    # Hybrid alone already has more than k matches; without a reranker to
    # rescore the pool, the final candidates[:k] slice must still keep the
    # graph-only match rather than cutting it off behind the hybrid list.
    hybrid_docs = [_doc(f"hybrid{i}") for i in range(5)]
    graph_only_doc = make_chunk("doc9", 0, "only found via graph traversal")

    monkeypatch.setattr(
        "ask_the_docs.retrieve.classify_query",
        lambda q, s: QueryClassification(label="local", entities=["BM25"]),
    )
    monkeypatch.setattr("ask_the_docs.retrieve.local_search", lambda entities, s: [graph_only_doc])

    class FakeIndex:
        def similarity_search_with_score(self, query, k):
            return [(doc, 0.1 * (i + 1)) for i, doc in enumerate(hybrid_docs)]

    monkeypatch.setattr("ask_the_docs.retrieve.load_index", lambda settings: FakeIndex())
    monkeypatch.setattr("ask_the_docs.retrieve.bm25_score", lambda query, settings: [])

    settings = make_settings(tmp_path).model_copy(
        update={"graph_enabled": True, "rerank_enabled": False, "candidate_pool_size": 5}
    )
    result = retrieve("how does BM25 relate to vectors?", settings, k=2)

    assert len(result) == 2
    assert graph_only_doc in result


def test_retrieve_falls_back_to_bm25_only_when_vector_search_fails(tmp_path, monkeypatch, caplog):
    class BrokenIndex:
        def similarity_search_with_score(self, query, k):
            raise RuntimeError("chroma unavailable")

    monkeypatch.setattr("ask_the_docs.retrieve.load_index", lambda settings: BrokenIndex())
    bm25_doc = _doc("b1")
    monkeypatch.setattr("ask_the_docs.retrieve.bm25_score", lambda query, settings: [(bm25_doc, 1.0)])

    settings = make_settings(tmp_path)
    with caplog.at_level(logging.WARNING):
        results = retrieve("anything", settings, k=1)

    assert len(results) == 1
    assert results[0].metadata["source_doc_id"] == "b1"
    assert any("vector search" in record.message.lower() for record in caplog.records)


def _sample(name: str) -> float:
    return REGISTRY.get_sample_value(name) or 0.0


def _stub_hybrid_with_one_hit(monkeypatch) -> None:
    doc = _doc("a")

    class FakeIndex:
        def similarity_search_with_score(self, query, k):
            return [(doc, 0.1)]

    monkeypatch.setattr("ask_the_docs.retrieve.load_index", lambda settings: FakeIndex())
    monkeypatch.setattr("ask_the_docs.retrieve.bm25_score", lambda query, settings: [])


def test_retrieve_observes_hybrid_retrieve_latency(tmp_path, monkeypatch):
    _stub_hybrid_with_one_hit(monkeypatch)
    before = _sample("retrieve_latency_seconds_count")

    retrieve("a query", make_settings(tmp_path), k=1)

    assert _sample("retrieve_latency_seconds_count") - before == 1.0


def test_retrieve_does_not_observe_rerank_latency_when_rerank_disabled(tmp_path, monkeypatch):
    _stub_hybrid_with_one_hit(monkeypatch)
    before = _sample("rerank_latency_seconds_count")

    retrieve("a query", make_settings(tmp_path).model_copy(update={"rerank_enabled": False}), k=1)

    assert _sample("rerank_latency_seconds_count") - before == 0.0


def test_retrieve_observes_rerank_latency_when_rerank_enabled(tmp_path, monkeypatch):
    _stub_hybrid_with_one_hit(monkeypatch)

    class PassThroughReranker:
        def rerank(self, query, candidates, k):
            return candidates[:k]

    monkeypatch.setattr("ask_the_docs.retrieve.get_reranker", lambda model_name: PassThroughReranker())
    rerank_before = _sample("rerank_latency_seconds_count")
    retrieve_before = _sample("retrieve_latency_seconds_count")

    retrieve("a query", make_settings(tmp_path).model_copy(update={"rerank_enabled": True}), k=1)

    assert _sample("rerank_latency_seconds_count") - rerank_before == 1.0
    assert _sample("retrieve_latency_seconds_count") - retrieve_before == 1.0
