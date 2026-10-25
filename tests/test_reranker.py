from langchain_core.documents import Document

from research_assistant.reranker import Reranker, get_reranker

DOC_A = Document(page_content="low relevance", metadata={"source": "a.pdf"})
DOC_B = Document(page_content="high relevance", metadata={"source": "b.pdf"})
DOC_C = Document(page_content="medium relevance", metadata={"source": "c.pdf"})


class FakeCrossEncoder:
    def __init__(self, model_name):
        self.model_name = model_name

    def predict(self, pairs):
        # doc order passed in is [A, B, C]; return scores so B > C > A
        return [0.1, 0.9, 0.5]


def test_get_reranker_loads_the_model_only_once(monkeypatch):
    load_count = {"count": 0}

    class CountingCrossEncoder:
        def __init__(self, model_name):
            load_count["count"] += 1

        def predict(self, pairs):
            return [0.0 for _ in pairs]

    monkeypatch.setattr("research_assistant.reranker.CrossEncoder", CountingCrossEncoder)
    get_reranker.cache_clear()

    a = get_reranker("fake-model")
    b = get_reranker("fake-model")

    assert a is b
    assert load_count["count"] == 1


def test_rerank_returns_top_k_sorted_descending_by_model_score(monkeypatch):
    monkeypatch.setattr("research_assistant.reranker.CrossEncoder", FakeCrossEncoder)

    reranker = Reranker(model_name="fake-model")
    result = reranker.rerank("some query", [DOC_A, DOC_B, DOC_C], k=2)

    assert result == [DOC_B, DOC_C]


def test_rerank_returns_all_candidates_when_k_exceeds_pool_size(monkeypatch):
    monkeypatch.setattr("research_assistant.reranker.CrossEncoder", FakeCrossEncoder)

    reranker = Reranker(model_name="fake-model")
    result = reranker.rerank("some query", [DOC_A, DOC_B, DOC_C], k=10)

    assert len(result) == 3


def test_rerank_handles_empty_candidate_list(monkeypatch):
    class EmptyCrossEncoder:
        def __init__(self, model_name):
            pass

        def predict(self, pairs):
            return []

    monkeypatch.setattr("research_assistant.reranker.CrossEncoder", EmptyCrossEncoder)

    reranker = Reranker(model_name="fake-model")
    result = reranker.rerank("some query", [], k=5)

    assert result == []


def test_rerank_passes_query_paired_with_each_candidate(monkeypatch):
    captured = {}

    class CapturingCrossEncoder:
        def __init__(self, model_name):
            pass

        def predict(self, pairs):
            captured["pairs"] = pairs
            return [0.0 for _ in pairs]

    monkeypatch.setattr("research_assistant.reranker.CrossEncoder", CapturingCrossEncoder)

    reranker = Reranker(model_name="fake-model")
    reranker.rerank("my query", [DOC_A, DOC_B], k=2)

    assert captured["pairs"] == [("my query", "low relevance"), ("my query", "high relevance")]
