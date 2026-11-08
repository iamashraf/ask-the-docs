import json

import pytest

from ask_the_docs.config import ProviderConfig, Settings
from ask_the_docs import queue
from ask_the_docs.queue import QueuePublishError, publish_ingest_job


def make_settings() -> Settings:
    return Settings(
        generator=ProviderConfig(provider="ollama", model="llama3.1:8b"),
        judge=ProviderConfig(provider="anthropic", model="claude-sonnet-5-5", api_key="x"),
    )


@pytest.fixture(autouse=True)
def _fresh_connection(monkeypatch):
    monkeypatch.setattr(queue, "_nc", None)
    monkeypatch.setattr(queue, "_js", None)


class FakeJetStream:
    def __init__(self):
        self.published = []

    async def add_stream(self, **kwargs):
        self.streams_added = getattr(self, "streams_added", 0) + 1

    async def publish(self, subject, payload):
        self.published.append((subject, payload))


class FakeNatsClient:
    def __init__(self, js):
        self._js = js
        self.closed = False

    @property
    def is_closed(self):
        return self.closed

    def jetstream(self):
        return self._js

    async def close(self):
        self.closed = True


async def test_publish_ingest_job_sends_the_job_payload(monkeypatch):
    js = FakeJetStream()
    client = FakeNatsClient(js)

    async def fake_connect(url):
        return client

    monkeypatch.setattr("ask_the_docs.queue.nats.connect", fake_connect)

    await publish_ingest_job(make_settings(), "job-1", kind="ingest", source="a.pdf")

    assert len(js.published) == 1
    subject, payload = js.published[0]
    assert subject == "ingest.jobs"
    assert b'"job_id": "job-1"' in payload or b'"job_id":"job-1"' in payload
    assert client.closed is False


async def test_publish_ingest_job_injects_trace_context(monkeypatch):
    """A fixed key present with a null value (the no-active-span case,
    covered separately below) would pass a bare "traceparent" in body
    check without proving propagate.inject() does anything -- this starts
    a real span (from a local, unregistered TracerProvider -- no global
    registration, no span processor, so nothing to leak at process exit)
    and asserts the published traceparent actually encodes that span's
    trace id."""
    from opentelemetry.sdk.trace import TracerProvider

    js = FakeJetStream()
    client = FakeNatsClient(js)

    async def fake_connect(url):
        return client

    monkeypatch.setattr("ask_the_docs.queue.nats.connect", fake_connect)

    local_tracer = TracerProvider().get_tracer(__name__)
    with local_tracer.start_as_current_span("test-span") as span:
        trace_id = format(span.get_span_context().trace_id, "032x")
        await publish_ingest_job(make_settings(), "job-1", kind="ingest", source="a.pdf")

    _, payload = js.published[0]
    body = json.loads(payload)
    assert body["traceparent"].startswith(f"00-{trace_id}-")


async def test_publish_ingest_job_with_no_active_span_still_publishes(monkeypatch):
    """Review Focus: inject() into an empty carrier when nothing is tracing
    must not raise -- the payload is still valid."""
    js = FakeJetStream()
    client = FakeNatsClient(js)

    async def fake_connect(url):
        return client

    monkeypatch.setattr("ask_the_docs.queue.nats.connect", fake_connect)

    await publish_ingest_job(make_settings(), "job-1", kind="ingest", source="a.pdf")

    assert len(js.published) == 1


async def test_publish_ingest_job_includes_enqueued_at(monkeypatch):
    js = FakeJetStream()
    client = FakeNatsClient(js)

    async def fake_connect(url):
        return client

    monkeypatch.setattr("ask_the_docs.queue.nats.connect", fake_connect)

    await publish_ingest_job(make_settings(), "job-1", kind="ingest", source="a.pdf")

    _, payload = js.published[0]
    body = json.loads(payload)
    assert "enqueued_at" in body


async def test_publish_ingest_job_wraps_connection_failures(monkeypatch):
    async def failing_connect(url):
        raise OSError("connection refused")

    monkeypatch.setattr("ask_the_docs.queue.nats.connect", failing_connect)

    with pytest.raises(QueuePublishError):
        await publish_ingest_job(make_settings(), "job-1", kind="ingest", source="a.pdf")


async def test_publish_ingest_job_reuses_one_connection_and_stream_setup(monkeypatch):
    js = FakeJetStream()
    client = FakeNatsClient(js)
    connects = []

    async def fake_connect(url):
        connects.append(url)
        return client

    monkeypatch.setattr("ask_the_docs.queue.nats.connect", fake_connect)

    await publish_ingest_job(make_settings(), "job-1", kind="ingest", source="a.pdf")
    await publish_ingest_job(make_settings(), "job-2", kind="ingest", source="b.pdf")

    assert len(connects) == 1
    assert js.streams_added == 1
    assert len(js.published) == 2


async def test_publish_ingest_job_replaces_a_closed_connection(monkeypatch):
    clients = [FakeNatsClient(FakeJetStream()), FakeNatsClient(FakeJetStream())]

    async def fake_connect(url):
        return clients.pop(0)

    first, second = list(clients)
    monkeypatch.setattr("ask_the_docs.queue.nats.connect", fake_connect)

    await publish_ingest_job(make_settings(), "job-1", kind="ingest", source="a.pdf")
    first.closed = True
    await publish_ingest_job(make_settings(), "job-2", kind="ingest", source="b.pdf")

    assert len(first._js.published) == 1
    assert len(second._js.published) == 1


async def test_close_publisher_closes_the_shared_connection_and_next_publish_reconnects(monkeypatch):
    client = FakeNatsClient(FakeJetStream())
    connects = []

    async def fake_connect(url):
        connects.append(url)
        return client

    monkeypatch.setattr("ask_the_docs.queue.nats.connect", fake_connect)

    await publish_ingest_job(make_settings(), "job-1", kind="ingest", source="a.pdf")
    await queue.close_publisher()
    assert client.closed is True

    client.closed = False
    await publish_ingest_job(make_settings(), "job-2", kind="ingest", source="a.pdf")
    assert len(connects) == 2


async def test_publish_failure_raises_and_does_not_cache_a_broken_connection(monkeypatch):
    class FailingJetStream(FakeJetStream):
        async def publish(self, subject, payload):
            raise OSError("boom")

    client = FakeNatsClient(FailingJetStream())

    async def fake_connect(url):
        return client

    monkeypatch.setattr("ask_the_docs.queue.nats.connect", fake_connect)

    with pytest.raises(QueuePublishError):
        await publish_ingest_job(make_settings(), "job-1", kind="ingest", source="a.pdf")
