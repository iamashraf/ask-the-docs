import pytest
from fastapi.testclient import TestClient
from langchain_core.documents import Document

from research_assistant.api import app
from research_assistant.ingest import PathSecurityError, UnsupportedFileTypeError, UrlSecurityError


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
