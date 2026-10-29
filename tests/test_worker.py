import pytest
from langchain_core.documents import Document

from ask_the_docs.config import ProviderConfig, Settings
from ask_the_docs.worker import process_job


def make_settings() -> Settings:
    return Settings(
        generator=ProviderConfig(provider="ollama", model="llama3.1:8b"),
        judge=ProviderConfig(provider="anthropic", model="claude-sonnet-5-5", api_key="x"),
    )


class RecordingAsyncRedis:
    def __init__(self):
        self.updates = []

    async def set(self, key, value, ex=None):
        pass

    async def get(self, key):
        return None


async def test_process_job_ingests_and_marks_done(monkeypatch):
    docs = [Document(page_content="hi", metadata={"source": "a.pdf", "source_doc_id": "a"})]
    monkeypatch.setattr("ask_the_docs.worker.dispatch", lambda source: docs)
    built = {}
    monkeypatch.setattr("ask_the_docs.worker.build_index", lambda d, s: built.setdefault("docs", d))
    updates = []

    async def fake_update_job(client, job_id, *, status, result=None, error=None):
        updates.append((job_id, status, result, error))

    monkeypatch.setattr("ask_the_docs.worker.update_job", fake_update_job)
    monkeypatch.setattr("ask_the_docs.worker.get_async_cache_client", lambda settings: RecordingAsyncRedis())

    await process_job(make_settings(), {"job_id": "job-1", "kind": "ingest", "source": "a.pdf"})

    assert built["docs"] == docs
    assert updates[0] == ("job-1", "processing", None, None)
    assert updates[1] == ("job-1", "done", {"documents_ingested": 1}, None)


async def test_process_job_marks_failed_on_an_exception(monkeypatch):
    def boom(source):
        raise RuntimeError("load failed")

    monkeypatch.setattr("ask_the_docs.worker.dispatch", boom)
    updates = []

    async def fake_update_job(client, job_id, *, status, result=None, error=None):
        updates.append((job_id, status, error))

    monkeypatch.setattr("ask_the_docs.worker.update_job", fake_update_job)
    monkeypatch.setattr("ask_the_docs.worker.get_async_cache_client", lambda settings: RecordingAsyncRedis())

    await process_job(make_settings(), {"job_id": "job-2", "kind": "ingest", "source": "bad.pdf"})

    assert updates[-1][0] == "job-2"
    assert updates[-1][1] == "failed"
    assert "load failed" in updates[-1][2]


async def test_process_job_handles_the_index_kind(monkeypatch, tmp_path):
    docs = [Document(page_content="hi", metadata={"source": "a.txt", "source_doc_id": "a"})]
    monkeypatch.setattr("ask_the_docs.worker.dispatch_all_under_docs_dir", lambda s: (docs, 3))
    monkeypatch.setattr("ask_the_docs.worker.build_index", lambda d, s: None)
    updates = []

    async def fake_update_job(client, job_id, *, status, result=None, error=None):
        updates.append((job_id, status, result, error))

    monkeypatch.setattr("ask_the_docs.worker.update_job", fake_update_job)
    monkeypatch.setattr("ask_the_docs.worker.get_async_cache_client", lambda settings: RecordingAsyncRedis())

    await process_job(make_settings(), {"job_id": "job-3", "kind": "index", "source": None})

    assert updates[-1] == ("job-3", "done", {"documents_ingested": 1, "files_found": 3}, None)


async def test_process_job_is_safe_to_run_twice_for_the_same_job_id(monkeypatch):
    """Covers the Review Focus risk: NATS redelivers a message if a worker
    crashes before acking, so the same job can run twice. process_job does
    nothing to special-case this -- it must be safe purely because
    dispatch/build_index are idempotent (roadmap item 1) and update_job
    simply overwrites status each time, not because of anything new here."""
    docs = [Document(page_content="hi", metadata={"source": "a.pdf", "source_doc_id": "a"})]
    monkeypatch.setattr("ask_the_docs.worker.dispatch", lambda source: docs)
    build_calls = []
    monkeypatch.setattr("ask_the_docs.worker.build_index", lambda d, s: build_calls.append(d))
    updates = []

    async def fake_update_job(client, job_id, *, status, result=None, error=None):
        updates.append((job_id, status, result, error))

    monkeypatch.setattr("ask_the_docs.worker.update_job", fake_update_job)
    monkeypatch.setattr("ask_the_docs.worker.get_async_cache_client", lambda settings: RecordingAsyncRedis())

    payload = {"job_id": "job-4", "kind": "ingest", "source": "a.pdf"}
    await process_job(make_settings(), payload)
    await process_job(make_settings(), payload)

    assert len(build_calls) == 2
    assert updates[-1] == ("job-4", "done", {"documents_ingested": 1}, None)
