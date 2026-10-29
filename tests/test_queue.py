import pytest

from ask_the_docs.config import ProviderConfig, Settings
from ask_the_docs.queue import QueuePublishError, publish_ingest_job


def make_settings() -> Settings:
    return Settings(
        generator=ProviderConfig(provider="ollama", model="llama3.1:8b"),
        judge=ProviderConfig(provider="anthropic", model="claude-sonnet-5-5", api_key="x"),
    )


class FakeJetStream:
    def __init__(self):
        self.published = []

    async def add_stream(self, **kwargs):
        pass

    async def publish(self, subject, payload):
        self.published.append((subject, payload))


class FakeNatsClient:
    def __init__(self, js):
        self._js = js
        self.closed = False

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
    assert client.closed is True


async def test_publish_ingest_job_wraps_connection_failures(monkeypatch):
    async def failing_connect(url):
        raise OSError("connection refused")

    monkeypatch.setattr("ask_the_docs.queue.nats.connect", failing_connect)

    with pytest.raises(QueuePublishError):
        await publish_ingest_job(make_settings(), "job-1", kind="ingest", source="a.pdf")
