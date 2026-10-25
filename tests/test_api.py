import logging
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from langchain_core.documents import Document

from research_assistant.answer import AnswerResult
from research_assistant.api import app
from research_assistant.config import ProviderConfig, Settings
from research_assistant.evaluate import EvalResult, JudgeScore
from research_assistant.ingest import PathSecurityError, UnsupportedFileTypeError, UrlSecurityError
from research_assistant.llm import OllamaUnavailableError


def make_settings() -> Settings:
    return Settings(
        generator=ProviderConfig(provider="ollama", model="llama3.1:8b"),
        judge=ProviderConfig(provider="anthropic", model="claude-sonnet-5-5", api_key="x"),
    )


def test_startup_triggers_seeding(monkeypatch):
    monkeypatch.setattr("research_assistant.api.get_settings", make_settings)
    captured = {}
    monkeypatch.setattr(
        "research_assistant.api.seed_if_needed", lambda settings: captured.setdefault("called", True)
    )

    with TestClient(app):
        pass

    assert captured.get("called") is True


def test_health_returns_ok():
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ingest_calls_dispatch_and_build_index(monkeypatch):
    docs = [Document(page_content="x", metadata={"source": "a.pdf", "source_doc_id": "1"})]
    monkeypatch.setattr("research_assistant.api.dispatch", lambda source: docs)
    monkeypatch.setattr("research_assistant.api.get_settings", lambda: object())
    captured = {}
    monkeypatch.setattr(
        "research_assistant.api.build_index", lambda d, settings: captured.setdefault("docs", d)
    )

    response = TestClient(app).post("/v1/ingest", json={"source": "a.pdf"})

    assert response.status_code == 200
    assert response.json() == {"documents_ingested": 1}
    assert captured["docs"] == docs


def test_ingest_rejects_empty_source():
    response = TestClient(app).post("/v1/ingest", json={"source": ""})
    assert response.status_code == 422


@pytest.mark.parametrize("exc_class", [PathSecurityError, UrlSecurityError, UnsupportedFileTypeError])
def test_ingest_returns_400_for_security_and_type_errors(monkeypatch, exc_class):
    def raise_error(source):
        raise exc_class("bad source")

    monkeypatch.setattr("research_assistant.api.dispatch", raise_error)

    response = TestClient(app).post("/v1/ingest", json={"source": "bad"})

    assert response.status_code == 400


def test_ingest_returns_404_for_missing_file(monkeypatch):
    def raise_error(source):
        raise FileNotFoundError(source)

    monkeypatch.setattr("research_assistant.api.dispatch", raise_error)

    response = TestClient(app).post("/v1/ingest", json={"source": "missing.pdf"})

    assert response.status_code == 404


def test_index_discovers_docs_dir_and_builds_index(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "research_assistant.api.get_settings", lambda: SimpleNamespace(docs_dir=tmp_path)
    )
    monkeypatch.setattr(
        "research_assistant.api.discover_sources",
        lambda docs_dir: [tmp_path / "a.pdf", tmp_path / "b.pdf"],
    )
    monkeypatch.setattr("research_assistant.api.dispatch", lambda source: [f"doc-for-{source}"])
    captured = {}
    monkeypatch.setattr(
        "research_assistant.api.build_index", lambda d, s: captured.setdefault("docs", d)
    )

    response = TestClient(app).post("/v1/index")

    assert response.status_code == 200
    assert response.json() == {"documents_ingested": 2, "files_found": 2}
    assert captured["docs"] == ["doc-for-a.pdf", "doc-for-b.pdf"]


def test_ask_returns_answer_and_citations(monkeypatch):
    monkeypatch.setattr("research_assistant.api.get_settings", lambda: object())
    captured = {}

    def fake_answer(query, settings, k):
        captured["query"] = query
        captured["k"] = k
        return AnswerResult(text="42", citations=["a.pdf"], context="ctx")

    monkeypatch.setattr("research_assistant.api.answer", fake_answer)

    response = TestClient(app).post("/v1/ask", json={"query": "What is X?"})

    assert response.status_code == 200
    assert response.json() == {"answer": "42", "citations": ["a.pdf"]}
    assert captured["query"] == "What is X?"
    assert captured["k"] == 5


def test_ask_rejects_empty_query():
    response = TestClient(app).post("/v1/ask", json={"query": ""})
    assert response.status_code == 422


def test_ask_rejects_non_positive_k():
    response = TestClient(app).post("/v1/ask", json={"query": "hi", "k": 0})
    assert response.status_code == 422


def test_ask_returns_503_when_ollama_unavailable(monkeypatch):
    monkeypatch.setattr("research_assistant.api.get_settings", lambda: object())

    def raise_error(query, settings, k):
        raise OllamaUnavailableError("could not reach ollama")

    monkeypatch.setattr("research_assistant.api.answer", raise_error)

    response = TestClient(app).post("/v1/ask", json={"query": "hi"})

    assert response.status_code == 503


def test_evaluate_returns_results(monkeypatch):
    monkeypatch.setattr("research_assistant.api.get_settings", make_settings)
    results = [
        EvalResult(
            question="Q1",
            score=JudgeScore(
                accuracy=1.0,
                answer_relevance=1.0,
                context_relevance=1.0,
                completeness=1.0,
                justification="ok",
            ),
            error=None,
        ),
        EvalResult(question="Q2", score=None, error="boom"),
    ]
    monkeypatch.setattr("research_assistant.api.run_evaluation", lambda settings, path: results)

    response = TestClient(app).post("/v1/evaluate", json={})

    assert response.status_code == 200
    body = response.json()
    assert body["results"][0]["question"] == "Q1"
    assert body["results"][0]["score"]["accuracy"] == 1.0
    assert body["results"][0]["error"] is None
    assert body["results"][1]["score"] is None
    assert body["results"][1]["error"] == "boom"


def test_evaluate_overrides_embeddings_backend(monkeypatch):
    monkeypatch.setattr("research_assistant.api.get_settings", make_settings)
    captured = {}

    def fake_run_evaluation(settings, path):
        captured["backend"] = settings.embeddings_backend
        return []

    monkeypatch.setattr("research_assistant.api.run_evaluation", fake_run_evaluation)

    response = TestClient(app).post("/v1/evaluate", json={"embeddings": "huggingface"})

    assert response.status_code == 200
    assert captured["backend"] == "huggingface"


def test_evaluate_rejects_invalid_embeddings_value():
    response = TestClient(app).post("/v1/evaluate", json={"embeddings": "bogus"})
    assert response.status_code == 422


def test_package_logger_configured_for_info_with_a_handler():
    """Importing api.py must configure the research_assistant logger so
    INFO-level messages (e.g. seed.py's progress logs) actually reach
    console output -- uvicorn only configures its own loggers, not ours."""
    package_logger = logging.getLogger("research_assistant")
    assert package_logger.getEffectiveLevel() <= logging.INFO
    assert len(package_logger.handlers) >= 1
