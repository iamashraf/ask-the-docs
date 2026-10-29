import logging
import uuid

import pytest
from fastapi.testclient import TestClient

from ask_the_docs.answer import AnswerResult
from ask_the_docs.api import app
from ask_the_docs.config import ProviderConfig, Settings
from ask_the_docs.evaluate import EvalResult, JudgeScore
from ask_the_docs.ingest import PathSecurityError, UnsupportedFileTypeError, UrlSecurityError
from ask_the_docs.jobs import JobStoreError
from ask_the_docs.llm import OllamaUnavailableError
from ask_the_docs.queue import QueuePublishError


def make_settings() -> Settings:
    return Settings(
        generator=ProviderConfig(provider="ollama", model="llama3.1:8b"),
        judge=ProviderConfig(provider="anthropic", model="claude-sonnet-5-5", api_key="x"),
    )


def test_startup_triggers_seeding(monkeypatch):
    monkeypatch.setattr("ask_the_docs.api.get_settings", make_settings)
    captured = {}
    monkeypatch.setattr(
        "ask_the_docs.api.seed_if_needed", lambda settings: captured.setdefault("called", True)
    )

    with TestClient(app):
        pass

    assert captured.get("called") is True


def test_health_returns_ok():
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ingest_returns_202_with_a_job_id(monkeypatch):
    monkeypatch.setattr("ask_the_docs.api.get_settings", lambda: object())
    monkeypatch.setattr("ask_the_docs.api.validate_source", lambda source: None)
    captured = {}

    async def fake_create_job(client, job_id, kind, source):
        captured["create"] = (job_id, kind, source)

    async def fake_publish(settings, job_id, kind, source):
        captured["publish"] = (job_id, kind, source)

    monkeypatch.setattr("ask_the_docs.api.create_job", fake_create_job)
    monkeypatch.setattr("ask_the_docs.api.publish_ingest_job", fake_publish)
    monkeypatch.setattr("ask_the_docs.api.get_async_cache_client", lambda settings: object())

    response = TestClient(app).post("/v1/ingest", json={"source": "a.pdf"})

    assert response.status_code == 202
    body = response.json()
    assert body["status"] == "queued"
    job_id = body["job_id"]
    uuid.UUID(job_id)  # raises if not a valid uuid
    assert captured["create"] == (job_id, "ingest", "a.pdf")
    assert captured["publish"] == (job_id, "ingest", "a.pdf")


def test_ingest_marks_the_job_failed_but_still_returns_202_when_publish_fails(monkeypatch):
    monkeypatch.setattr("ask_the_docs.api.get_settings", lambda: object())
    monkeypatch.setattr("ask_the_docs.api.validate_source", lambda source: None)

    async def fake_create_job(client, job_id, kind, source):
        pass

    async def failing_publish(settings, job_id, kind, source):
        raise QueuePublishError("nats down")

    updated = {}

    async def fake_update_job(client, job_id, *, status, result=None, error=None):
        updated["call"] = (job_id, status, error)

    monkeypatch.setattr("ask_the_docs.api.create_job", fake_create_job)
    monkeypatch.setattr("ask_the_docs.api.publish_ingest_job", failing_publish)
    monkeypatch.setattr("ask_the_docs.api.update_job", fake_update_job)
    monkeypatch.setattr("ask_the_docs.api.get_async_cache_client", lambda settings: object())

    response = TestClient(app).post("/v1/ingest", json={"source": "a.pdf"})

    assert response.status_code == 202
    job_id = response.json()["job_id"]
    assert updated["call"][0] == job_id
    assert updated["call"][1] == "failed"


def test_ingest_returns_503_when_job_store_write_fails(monkeypatch):
    monkeypatch.setattr("ask_the_docs.api.get_settings", lambda: object())
    monkeypatch.setattr("ask_the_docs.api.validate_source", lambda source: None)

    async def failing_create_job(client, job_id, kind, source):
        raise JobStoreError("redis down")

    monkeypatch.setattr("ask_the_docs.api.create_job", failing_create_job)
    monkeypatch.setattr("ask_the_docs.api.get_async_cache_client", lambda settings: object())

    response = TestClient(app).post("/v1/ingest", json={"source": "a.pdf"})

    assert response.status_code == 503


def test_ingest_rejects_empty_source():
    response = TestClient(app).post("/v1/ingest", json={"source": ""})
    assert response.status_code == 422


@pytest.mark.parametrize("exc_class", [PathSecurityError, UrlSecurityError, UnsupportedFileTypeError])
def test_ingest_returns_400_for_security_and_type_errors(monkeypatch, exc_class):
    monkeypatch.setattr("ask_the_docs.api.get_settings", lambda: object())

    def raise_error(source):
        raise exc_class("bad source")

    monkeypatch.setattr("ask_the_docs.api.validate_source", raise_error)

    response = TestClient(app).post("/v1/ingest", json={"source": "bad"})

    assert response.status_code == 400


def test_ingest_returns_404_for_missing_file(monkeypatch):
    monkeypatch.setattr("ask_the_docs.api.get_settings", lambda: object())

    def raise_error(source):
        raise FileNotFoundError(source)

    monkeypatch.setattr("ask_the_docs.api.validate_source", raise_error)

    response = TestClient(app).post("/v1/ingest", json={"source": "missing.pdf"})

    assert response.status_code == 404


def test_index_returns_202_with_a_job_id(monkeypatch):
    monkeypatch.setattr("ask_the_docs.api.get_settings", lambda: object())
    captured = {}

    async def fake_create_job(client, job_id, kind, source):
        captured["create"] = (job_id, kind, source)

    async def fake_publish(settings, job_id, kind, source):
        captured["publish"] = (job_id, kind, source)

    monkeypatch.setattr("ask_the_docs.api.create_job", fake_create_job)
    monkeypatch.setattr("ask_the_docs.api.publish_ingest_job", fake_publish)
    monkeypatch.setattr("ask_the_docs.api.get_async_cache_client", lambda settings: object())

    response = TestClient(app).post("/v1/index")

    assert response.status_code == 202
    job_id = response.json()["job_id"]
    assert captured["create"] == (job_id, "index", None)
    assert captured["publish"] == (job_id, "index", None)


def test_get_job_status_returns_the_job_record(monkeypatch):
    monkeypatch.setattr("ask_the_docs.api.get_settings", lambda: object())
    record = {
        "job_id": "job-1", "kind": "ingest", "source": "a.pdf", "status": "done",
        "result": {"documents_ingested": 1}, "error": None,
    }

    async def fake_get_job(client, job_id):
        return record if job_id == "job-1" else None

    monkeypatch.setattr("ask_the_docs.api.get_job", fake_get_job)
    monkeypatch.setattr("ask_the_docs.api.get_async_cache_client", lambda settings: object())

    response = TestClient(app).get("/v1/jobs/job-1")

    assert response.status_code == 200
    assert response.json()["status"] == "done"


def test_get_job_status_returns_404_for_an_unknown_job(monkeypatch):
    monkeypatch.setattr("ask_the_docs.api.get_settings", lambda: object())

    async def fake_get_job(client, job_id):
        return None

    monkeypatch.setattr("ask_the_docs.api.get_job", fake_get_job)
    monkeypatch.setattr("ask_the_docs.api.get_async_cache_client", lambda settings: object())

    response = TestClient(app).get("/v1/jobs/nope")

    assert response.status_code == 404


def test_ask_returns_answer_and_citations(monkeypatch):
    monkeypatch.setattr("ask_the_docs.api.get_settings", lambda: object())
    captured = {}

    def fake_answer(query, settings, k):
        captured["query"] = query
        captured["k"] = k
        return AnswerResult(text="42", citations=["a.pdf"], context="ctx")

    monkeypatch.setattr("ask_the_docs.api.answer", fake_answer)

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
    monkeypatch.setattr("ask_the_docs.api.get_settings", lambda: object())

    def raise_error(query, settings, k):
        raise OllamaUnavailableError("could not reach ollama")

    monkeypatch.setattr("ask_the_docs.api.answer", raise_error)

    response = TestClient(app).post("/v1/ask", json={"query": "hi"})

    assert response.status_code == 503


def test_evaluate_returns_results(monkeypatch):
    monkeypatch.setattr("ask_the_docs.api.get_settings", make_settings)
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
    monkeypatch.setattr("ask_the_docs.api.run_evaluation", lambda settings, path: results)

    response = TestClient(app).post("/v1/evaluate", json={})

    assert response.status_code == 200
    body = response.json()
    assert body["results"][0]["question"] == "Q1"
    assert body["results"][0]["score"]["accuracy"] == 1.0
    assert body["results"][0]["error"] is None
    assert body["results"][1]["score"] is None
    assert body["results"][1]["error"] == "boom"


def test_evaluate_overrides_embeddings_backend(monkeypatch):
    monkeypatch.setattr("ask_the_docs.api.get_settings", make_settings)
    captured = {}

    def fake_run_evaluation(settings, path):
        captured["backend"] = settings.embeddings_backend
        return []

    monkeypatch.setattr("ask_the_docs.api.run_evaluation", fake_run_evaluation)

    response = TestClient(app).post("/v1/evaluate", json={"embeddings": "huggingface"})

    assert response.status_code == 200
    assert captured["backend"] == "huggingface"


def test_evaluate_rejects_invalid_embeddings_value():
    response = TestClient(app).post("/v1/evaluate", json={"embeddings": "bogus"})
    assert response.status_code == 422


def test_evaluate_overrides_hybrid_alpha(monkeypatch):
    monkeypatch.setattr("ask_the_docs.api.get_settings", make_settings)
    captured = {}

    def fake_run_evaluation(settings, path):
        captured["alpha"] = settings.hybrid_alpha
        return []

    monkeypatch.setattr("ask_the_docs.api.run_evaluation", fake_run_evaluation)

    response = TestClient(app).post("/v1/evaluate", json={"hybrid_alpha": 0.8})

    assert response.status_code == 200
    assert captured["alpha"] == 0.8


def test_evaluate_overrides_rerank_enabled(monkeypatch):
    monkeypatch.setattr("ask_the_docs.api.get_settings", make_settings)
    captured = {}

    def fake_run_evaluation(settings, path):
        captured["rerank"] = settings.rerank_enabled
        return []

    monkeypatch.setattr("ask_the_docs.api.run_evaluation", fake_run_evaluation)

    response = TestClient(app).post("/v1/evaluate", json={"rerank_enabled": False})

    assert response.status_code == 200
    assert captured["rerank"] is False


def test_evaluate_overrides_query_rewrite_enabled(monkeypatch):
    monkeypatch.setattr("ask_the_docs.api.get_settings", make_settings)
    captured = {}

    def fake_run_evaluation(settings, path):
        captured["rewrite"] = settings.query_rewrite_enabled
        return []

    monkeypatch.setattr("ask_the_docs.api.run_evaluation", fake_run_evaluation)

    response = TestClient(app).post("/v1/evaluate", json={"query_rewrite_enabled": False})

    assert response.status_code == 200
    assert captured["rewrite"] is False


def test_evaluate_rejects_hybrid_alpha_out_of_range():
    response = TestClient(app).post("/v1/evaluate", json={"hybrid_alpha": 1.5})
    assert response.status_code == 422


def test_evaluate_accepts_hybrid_alpha_boundary_values(monkeypatch):
    monkeypatch.setattr("ask_the_docs.api.get_settings", make_settings)
    captured = []
    monkeypatch.setattr(
        "ask_the_docs.api.run_evaluation",
        lambda settings, path: captured.append(settings.hybrid_alpha) or [],
    )

    low = TestClient(app).post("/v1/evaluate", json={"hybrid_alpha": 0.0})
    high = TestClient(app).post("/v1/evaluate", json={"hybrid_alpha": 1.0})

    assert low.status_code == 200 and high.status_code == 200
    assert captured == [0.0, 1.0]


def test_evaluate_combines_multiple_overrides_in_one_call(monkeypatch):
    monkeypatch.setattr("ask_the_docs.api.get_settings", make_settings)
    captured = {}

    def fake_run_evaluation(settings, path):
        captured["backend"] = settings.embeddings_backend
        captured["alpha"] = settings.hybrid_alpha
        captured["rerank"] = settings.rerank_enabled
        captured["rewrite"] = settings.query_rewrite_enabled
        return []

    monkeypatch.setattr("ask_the_docs.api.run_evaluation", fake_run_evaluation)

    response = TestClient(app).post(
        "/v1/evaluate",
        json={
            "embeddings": "huggingface",
            "hybrid_alpha": 0.8,
            "rerank_enabled": False,
            "query_rewrite_enabled": False,
        },
    )

    assert response.status_code == 200
    assert captured == {"backend": "huggingface", "alpha": 0.8, "rerank": False, "rewrite": False}


def test_package_logger_configured_for_info_with_a_handler():
    """Importing api.py must configure the ask_the_docs logger so
    INFO-level messages (e.g. seed.py's progress logs) actually reach
    console output -- uvicorn only configures its own loggers, not ours."""
    package_logger = logging.getLogger("ask_the_docs")
    assert package_logger.getEffectiveLevel() <= logging.INFO
    assert len(package_logger.handlers) >= 1
