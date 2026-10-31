from langchain_core.documents import Document

from ask_the_docs.config import ProviderConfig, Settings
from ask_the_docs.retrieval_eval import (
    RetrievalCase,
    RetrievalResult,
    load_retrieval_cases,
    retrieval_metrics,
    retrieval_report,
    run_retrieval_eval,
)


def make_settings(**overrides) -> Settings:
    fields = dict(
        generator=ProviderConfig(provider="ollama", model="llama3.1:8b"),
        judge=ProviderConfig(provider="anthropic", model="claude-sonnet-5-5", api_key="x"),
    )
    fields.update(overrides)
    return Settings(**fields)


def _docs(*sources: str) -> list[Document]:
    return [Document(page_content="", metadata={"source": source}) for source in sources]


def test_load_retrieval_cases_parses_yaml(tmp_path):
    path = tmp_path / "cases.yaml"
    path.write_text(
        """
- question: "What is multi-head attention?"
  expected_source: "https://arxiv.org/pdf/1706.03762"
- question: "What is LoRA?"
  expected_source: "https://arxiv.org/pdf/2106.09685"
"""
    )

    cases = load_retrieval_cases(path)

    assert cases == [
        RetrievalCase(question="What is multi-head attention?", expected_source="https://arxiv.org/pdf/1706.03762"),
        RetrievalCase(question="What is LoRA?", expected_source="https://arxiv.org/pdf/2106.09685"),
    ]


def test_run_retrieval_eval_computes_rank_from_first_matching_chunk(tmp_path, monkeypatch):
    path = tmp_path / "cases.yaml"
    path.write_text(
        """
- question: "What is LoRA?"
  expected_source: "expected"
"""
    )
    monkeypatch.setattr(
        "ask_the_docs.retrieval_eval.retrieve",
        lambda query, settings, k: _docs("x", "y", "expected"),
    )

    results = run_retrieval_eval(make_settings(), path)

    assert results[0].rank == 3
    assert results[0].returned_sources == ["x", "y", "expected"]


def test_run_retrieval_eval_rank_none_when_source_absent(tmp_path, monkeypatch):
    path = tmp_path / "cases.yaml"
    path.write_text(
        """
- question: "What is LoRA?"
  expected_source: "expected"
"""
    )
    monkeypatch.setattr(
        "ask_the_docs.retrieval_eval.retrieve",
        lambda query, settings, k: _docs("x", "y", "z"),
    )

    results = run_retrieval_eval(make_settings(), path)

    assert results[0].rank is None
    assert results[0].returned_sources == ["x", "y", "z"]


def test_run_retrieval_eval_rank_is_first_matching_chunk_not_last(tmp_path, monkeypatch):
    path = tmp_path / "cases.yaml"
    path.write_text(
        """
- question: "What is LoRA?"
  expected_source: "expected"
"""
    )
    monkeypatch.setattr(
        "ask_the_docs.retrieval_eval.retrieve",
        lambda query, settings, k: _docs("x", "expected", "expected"),
    )

    results = run_retrieval_eval(make_settings(), path)

    assert results[0].rank == 2


def test_run_retrieval_eval_empty_retrieve_result(tmp_path, monkeypatch):
    path = tmp_path / "cases.yaml"
    path.write_text(
        """
- question: "What is LoRA?"
  expected_source: "expected"
"""
    )
    monkeypatch.setattr("ask_the_docs.retrieval_eval.retrieve", lambda query, settings, k: [])

    results = run_retrieval_eval(make_settings(), path)

    assert results[0].rank is None
    assert results[0].returned_sources == []


def test_run_retrieval_eval_calls_retrieve_with_question_and_k(tmp_path, monkeypatch):
    path = tmp_path / "cases.yaml"
    path.write_text(
        """
- question: "What is LoRA?"
  expected_source: "expected"
- question: "What is RAG?"
  expected_source: "expected-2"
"""
    )
    captured = []
    monkeypatch.setattr(
        "ask_the_docs.retrieval_eval.retrieve",
        lambda query, settings, k: captured.append((query, k)) or [],
    )

    run_retrieval_eval(make_settings(), path, k=3)

    assert captured == [("What is LoRA?", 3), ("What is RAG?", 3)]


def test_retrieval_metrics_hit_at_1_hit_at_k_and_mrr():
    results = [
        RetrievalResult(case=RetrievalCase(question="q1", expected_source="s1"), rank=1, returned_sources=["s1"] * 5),
        RetrievalResult(case=RetrievalCase(question="q2", expected_source="s2"), rank=2, returned_sources=["s2"] * 5),
        RetrievalResult(case=RetrievalCase(question="q3", expected_source="s3"), rank=None, returned_sources=["x"] * 5),
        RetrievalResult(case=RetrievalCase(question="q4", expected_source="s4"), rank=5, returned_sources=["s4"] * 5),
    ]

    metrics = retrieval_metrics(results)

    assert metrics["hit_at_1"] == 0.25
    assert metrics["hit_at_k"] == 0.75
    assert metrics["mrr"] == (1 + 0.5 + 0 + 0.2) / 4
    assert metrics["k"] == 5


def test_retrieval_metrics_empty_results_all_zero():
    assert retrieval_metrics([]) == {"hit_at_1": 0.0, "hit_at_k": 0.0, "mrr": 0.0, "k": 0.0}


def test_retrieval_report_lists_misses_with_returned_sources():
    results = [
        RetrievalResult(case=RetrievalCase(question="q1", expected_source="s1"), rank=1, returned_sources=["s1"]),
        RetrievalResult(
            case=RetrievalCase(question="q2", expected_source="s2"), rank=None, returned_sources=["x", "y"]
        ),
    ]

    report = retrieval_report(results, k=5)

    assert "hit@1" in report
    assert "hit@5" in report
    assert "mrr" in report
    assert "q2" in report
    assert "s2" in report
    assert "['x', 'y']" in report
