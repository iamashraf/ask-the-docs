import asyncio
import json
import logging
from datetime import UTC, datetime
from unittest.mock import Mock

import pytest
from langchain_core.documents import Document

from ask_the_docs import worker
from ask_the_docs.config import ProviderConfig, Settings
from ask_the_docs.worker import (
    DURABLE_CONSUMER_NAME,
    WORKER_ACK_WAIT_SECONDS,
    _consume_once,
    _sample_queue_depth,
    process_job,
    run_worker,
)
from tests.test_metrics import find_data_point, find_metric, get_metrics_data


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


def _metric_point(name, **attributes):
    """Looks up a recorded data point, tolerating one that doesn't exist yet."""
    try:
        metric = find_metric(get_metrics_data(), name)
    except AssertionError:
        return None
    try:
        return find_data_point(metric, **attributes)
    except AssertionError:
        return None


def _counter_value(name, **attributes):
    point = _metric_point(name, **attributes)
    return point.value if point is not None else 0


def _histogram_count(name, **attributes):
    point = _metric_point(name, **attributes)
    return point.count if point is not None else 0


async def test_process_job_ingests_and_marks_done(monkeypatch, caplog):
    docs = [Document(page_content="hi", metadata={"source": "a.pdf", "source_doc_id": "a"})]
    monkeypatch.setattr("ask_the_docs.worker.dispatch", lambda source: docs)
    built = {}
    monkeypatch.setattr("ask_the_docs.worker.build_index", lambda d, s: built.setdefault("docs", d))
    updates = []

    async def fake_update_job(client, job_id, *, status, result=None, error=None):
        updates.append((job_id, status, result, error))

    monkeypatch.setattr("ask_the_docs.worker.update_job", fake_update_job)
    monkeypatch.setattr("ask_the_docs.worker.get_async_cache_client", lambda settings: RecordingAsyncRedis())

    caplog.set_level(logging.INFO, logger="ask_the_docs.worker")
    await process_job(make_settings(), {"job_id": "job-1", "kind": "ingest", "source": "a.pdf"})

    assert built["docs"] == docs
    assert updates[0] == ("job-1", "processing", None, None)
    assert updates[1] == ("job-1", "done", {"documents_ingested": 1}, None)
    assert any("Job job-1 completed" in record.message for record in caplog.records)


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


async def test_process_job_increments_ingest_job_total_on_success(monkeypatch):
    docs = [Document(page_content="hi", metadata={"source": "a.pdf", "source_doc_id": "a"})]
    monkeypatch.setattr("ask_the_docs.worker.dispatch", lambda source: docs)
    monkeypatch.setattr("ask_the_docs.worker.build_index", lambda d, s: None)
    monkeypatch.setattr("ask_the_docs.worker.update_job", lambda *a, **k: _noop())
    monkeypatch.setattr("ask_the_docs.worker.get_async_cache_client", lambda settings: RecordingAsyncRedis())

    before = _counter_value("ingest_job", status="success")

    await process_job(make_settings(), {"job_id": "job-5", "kind": "ingest", "source": "a.pdf"})

    after = _counter_value("ingest_job", status="success")
    assert after - before == 1.0


async def test_process_job_increments_ingest_job_total_on_failure(monkeypatch):
    def boom(source):
        raise RuntimeError("load failed")

    monkeypatch.setattr("ask_the_docs.worker.dispatch", boom)
    monkeypatch.setattr("ask_the_docs.worker.update_job", lambda *a, **k: _noop())
    monkeypatch.setattr("ask_the_docs.worker.get_async_cache_client", lambda settings: RecordingAsyncRedis())

    before = _counter_value("ingest_job", status="failure")

    await process_job(make_settings(), {"job_id": "job-6", "kind": "ingest", "source": "bad.pdf"})

    after = _counter_value("ingest_job", status="failure")
    assert after - before == 1.0


async def test_process_job_observes_build_index_latency(monkeypatch):
    docs = [Document(page_content="hi", metadata={"source": "a.pdf", "source_doc_id": "a"})]
    monkeypatch.setattr("ask_the_docs.worker.dispatch", lambda source: docs)
    monkeypatch.setattr("ask_the_docs.worker.build_index", lambda d, s: None)
    monkeypatch.setattr("ask_the_docs.worker.update_job", lambda *a, **k: _noop())
    monkeypatch.setattr("ask_the_docs.worker.get_async_cache_client", lambda settings: RecordingAsyncRedis())

    before = _histogram_count("build_index_latency")

    await process_job(make_settings(), {"job_id": "job-7", "kind": "ingest", "source": "a.pdf"})

    after = _histogram_count("build_index_latency")
    assert after - before == 1.0


async def test_process_job_skips_latency_observation_when_enqueued_at_missing(monkeypatch):
    """Review Focus: a payload from an older publisher (or any caller that
    never set enqueued_at) must not crash process_job."""
    docs = [Document(page_content="hi", metadata={"source": "a.pdf", "source_doc_id": "a"})]
    monkeypatch.setattr("ask_the_docs.worker.dispatch", lambda source: docs)
    monkeypatch.setattr("ask_the_docs.worker.build_index", lambda d, s: None)
    monkeypatch.setattr("ask_the_docs.worker.update_job", lambda *a, **k: _noop())
    monkeypatch.setattr("ask_the_docs.worker.get_async_cache_client", lambda settings: RecordingAsyncRedis())

    before = _histogram_count("worker_job_latency")

    payload = {"job_id": "job-8", "kind": "ingest", "source": "a.pdf"}
    assert "enqueued_at" not in payload
    await process_job(make_settings(), payload)  # must not raise

    after = _histogram_count("worker_job_latency")
    assert after == before


async def test_process_job_skips_latency_observation_when_enqueued_at_unparsable(monkeypatch):
    docs = [Document(page_content="hi", metadata={"source": "a.pdf", "source_doc_id": "a"})]
    monkeypatch.setattr("ask_the_docs.worker.dispatch", lambda source: docs)
    monkeypatch.setattr("ask_the_docs.worker.build_index", lambda d, s: None)
    monkeypatch.setattr("ask_the_docs.worker.update_job", lambda *a, **k: _noop())
    monkeypatch.setattr("ask_the_docs.worker.get_async_cache_client", lambda settings: RecordingAsyncRedis())

    before = _histogram_count("worker_job_latency")

    payload = {"job_id": "job-9", "kind": "ingest", "source": "a.pdf", "enqueued_at": "not-a-timestamp"}
    await process_job(make_settings(), payload)  # must not raise

    after = _histogram_count("worker_job_latency")
    assert after == before


async def test_process_job_observes_worker_job_latency_when_enqueued_at_present(monkeypatch):
    docs = [Document(page_content="hi", metadata={"source": "a.pdf", "source_doc_id": "a"})]
    monkeypatch.setattr("ask_the_docs.worker.dispatch", lambda source: docs)
    monkeypatch.setattr("ask_the_docs.worker.build_index", lambda d, s: None)
    monkeypatch.setattr("ask_the_docs.worker.update_job", lambda *a, **k: _noop())
    monkeypatch.setattr("ask_the_docs.worker.get_async_cache_client", lambda settings: RecordingAsyncRedis())

    before = _histogram_count("worker_job_latency")

    payload = {
        "job_id": "job-10",
        "kind": "ingest",
        "source": "a.pdf",
        "enqueued_at": datetime.now(UTC).isoformat(),
    }
    await process_job(make_settings(), payload)

    after = _histogram_count("worker_job_latency")
    assert after - before == 1.0


class FakeMsg:
    def __init__(self, data: bytes):
        self.data = data
        self.acked = False

    async def ack(self):
        self.acked = True


class FakeSub:
    def __init__(self, batches):
        self._batches = list(batches)

    async def fetch(self, n, timeout):
        if not self._batches:
            raise TimeoutError()
        return self._batches.pop(0)


async def test_consume_once_acks_a_successfully_processed_message(monkeypatch):
    msg = FakeMsg(json.dumps({"job_id": "job-1", "kind": "ingest", "source": "a.pdf"}).encode())
    sub = FakeSub([[msg]])
    processed = []

    async def fake_process_job(settings, payload):
        processed.append(payload)

    monkeypatch.setattr("ask_the_docs.worker.process_job", fake_process_job)

    await _consume_once(make_settings(), sub)

    assert processed == [{"job_id": "job-1", "kind": "ingest", "source": "a.pdf"}]
    assert msg.acked is True


async def test_consume_once_leaves_a_message_unacked_and_does_not_raise_when_process_job_raises(monkeypatch):
    """Review Focus: a worker crash mid-job must not take the whole worker
    down -- the message stays unacked so NATS redelivers it, and the next
    message still gets a chance to run."""
    msg = FakeMsg(json.dumps({"job_id": "job-1", "kind": "ingest", "source": "a.pdf"}).encode())
    sub = FakeSub([[msg]])

    async def failing_process_job(settings, payload):
        raise ConnectionError("redis down mid-job")

    monkeypatch.setattr("ask_the_docs.worker.process_job", failing_process_job)

    await _consume_once(make_settings(), sub)  # must not raise

    assert msg.acked is False


async def test_consume_once_parents_process_job_on_the_extracted_trace_context(monkeypatch):
    """Proves the extracted traceparent actually becomes process_job's
    parent context -- not just that extraction doesn't crash (that's the
    separate no-key test below)."""
    from opentelemetry import trace as trace_api

    known_trace_id = "4bf92f3577b34da6a3ce929d0e0e4736"
    traceparent = f"00-{known_trace_id}-00f067aa0ba902b7-01"
    msg = FakeMsg(
        json.dumps({"job_id": "job-1", "kind": "ingest", "source": "a.pdf", "traceparent": traceparent}).encode()
    )
    sub = FakeSub([[msg]])
    captured = {}

    async def fake_process_job(settings, payload):
        captured["trace_id"] = format(trace_api.get_current_span().get_span_context().trace_id, "032x")

    monkeypatch.setattr("ask_the_docs.worker.process_job", fake_process_job)

    await _consume_once(make_settings(), sub)

    assert captured["trace_id"] == known_trace_id


async def test_consume_once_extracts_trace_context_without_a_traceparent_key(monkeypatch):
    """Review Focus: a message with no "traceparent" key at all (e.g. from
    an older publisher) must not crash extraction."""
    msg = FakeMsg(json.dumps({"job_id": "job-1", "kind": "ingest", "source": "a.pdf"}).encode())
    sub = FakeSub([[msg]])
    monkeypatch.setattr("ask_the_docs.worker.process_job", lambda settings, payload: _noop())

    await _consume_once(make_settings(), sub)  # must not raise

    assert msg.acked is True


async def _noop():
    pass


async def test_consume_once_drops_an_unparseable_message_without_crashing(monkeypatch):
    msg = FakeMsg(b"not valid json")
    sub = FakeSub([[msg]])
    called = []
    monkeypatch.setattr("ask_the_docs.worker.process_job", lambda *a: called.append(a))

    await _consume_once(make_settings(), sub)  # must not raise

    assert called == []


async def test_run_worker_subscribes_with_a_generous_ack_wait(monkeypatch):
    monkeypatch.setattr("ask_the_docs.worker.configure_tracing", lambda service_name, settings: None)
    monkeypatch.setattr("ask_the_docs.worker.configure_logging", lambda service_name, settings: None)
    monkeypatch.setattr("ask_the_docs.worker.configure_metrics", lambda service_name, settings: None)
    calls = {}

    class FakeJetStream:
        async def add_stream(self, **kwargs):
            pass

        async def pull_subscribe(self, subject, durable=None, config=None):
            calls["config"] = config
            raise asyncio.CancelledError()

    class FakeNatsClient:
        def jetstream(self):
            return FakeJetStream()

    async def fake_connect(url):
        return FakeNatsClient()

    monkeypatch.setattr("ask_the_docs.worker.nats.connect", fake_connect)

    with pytest.raises(asyncio.CancelledError):
        await run_worker(make_settings())

    assert calls["config"].ack_wait == WORKER_ACK_WAIT_SECONDS


async def test_run_worker_configures_tracing(monkeypatch):
    """The real worker entrypoint is cli.py's _worker, which calls
    run_worker() directly -- never worker.py's own main(). Tracing must be
    configured here, not in main(), or the deployed worker never exports a
    single span (the actual bug this test pins)."""
    captured = {}
    monkeypatch.setattr(
        "ask_the_docs.worker.configure_tracing",
        lambda service_name, settings: captured.setdefault("service_name", service_name),
    )
    monkeypatch.setattr("ask_the_docs.worker.configure_logging", lambda service_name, settings: None)
    monkeypatch.setattr("ask_the_docs.worker.configure_metrics", lambda service_name, settings: None)

    class FakeJetStream:
        async def add_stream(self, **kwargs):
            pass

        async def pull_subscribe(self, subject, durable=None, config=None):
            raise asyncio.CancelledError()

    class FakeNatsClient:
        def jetstream(self):
            return FakeJetStream()

    async def fake_connect(url):
        return FakeNatsClient()

    monkeypatch.setattr("ask_the_docs.worker.nats.connect", fake_connect)

    with pytest.raises(asyncio.CancelledError):
        await run_worker(make_settings())

    assert captured["service_name"] == "ask-the-docs-worker"


async def test_run_worker_configures_logging_and_metrics(monkeypatch):
    """configure_logging/configure_metrics must be wired in alongside
    configure_tracing -- the worker emits logs and records metrics, so both
    need their OTel providers set up before the loop starts. The call order
    is pinned too, matching run_worker's configure_tracing/_logging/_metrics
    sequence."""
    order = []
    monkeypatch.setattr(
        "ask_the_docs.worker.configure_tracing",
        lambda service_name, settings: order.append(("tracing", service_name)),
    )
    monkeypatch.setattr(
        "ask_the_docs.worker.configure_logging",
        lambda service_name, settings: order.append(("logging", service_name)),
    )
    monkeypatch.setattr(
        "ask_the_docs.worker.configure_metrics",
        lambda service_name, settings: order.append(("metrics", service_name)),
    )

    class FakeJetStream:
        async def add_stream(self, **kwargs):
            pass

        async def pull_subscribe(self, subject, durable=None, config=None):
            raise asyncio.CancelledError()

    class FakeNatsClient:
        def jetstream(self):
            return FakeJetStream()

    async def fake_connect(url):
        return FakeNatsClient()

    monkeypatch.setattr("ask_the_docs.worker.nats.connect", fake_connect)

    with pytest.raises(asyncio.CancelledError):
        await run_worker(make_settings())

    assert order == [
        ("tracing", "ask-the-docs-worker"),
        ("logging", "ask-the-docs-worker"),
        ("metrics", "ask-the-docs-worker"),
    ]


class FakeJetStreamWithConsumerInfo:
    def __init__(self, num_pending=None, raises=None):
        self._num_pending = num_pending
        self._raises = raises
        self.calls = []

    async def consumer_info(self, stream, durable):
        self.calls.append((stream, durable))
        if self._raises is not None:
            raise self._raises
        return Mock(num_pending=self._num_pending)


async def test_sample_queue_depth_sets_gauge_from_consumer_info(monkeypatch):
    js = FakeJetStreamWithConsumerInfo(num_pending=7)

    await _sample_queue_depth(js)

    point = find_data_point(find_metric(get_metrics_data(), "job_queue_depth"))
    assert point.value == 7
    assert js.calls == [(worker.INGEST_JOBS_STREAM, DURABLE_CONSUMER_NAME)]


async def test_sample_queue_depth_logs_and_continues_on_consumer_info_failure(monkeypatch, caplog):
    """Review Focus: a failed NATS admin call (e.g. consumer not found yet)
    must not take the worker loop down -- it's logged and skipped."""
    js = FakeJetStreamWithConsumerInfo(raises=RuntimeError("consumer not found"))

    with caplog.at_level(logging.WARNING, logger="ask_the_docs.worker"):
        await _sample_queue_depth(js)  # must not raise

    assert any("queue depth" in r.getMessage().lower() for r in caplog.records if r.levelno == logging.WARNING)
