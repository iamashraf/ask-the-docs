from watchfiles import Change

from ask_the_docs.config import ProviderConfig, Settings
from ask_the_docs.queue import QueuePublishError
from ask_the_docs.watcher import watch_docs_dir


def make_settings(**overrides) -> Settings:
    fields = dict(
        generator=ProviderConfig(provider="ollama", model="llama3.1:8b"),
        judge=ProviderConfig(provider="anthropic", model="claude-sonnet-5-5", api_key="x"),
    )
    fields.update(overrides)
    return Settings(**fields)


class RecordingAsyncRedis:
    def __init__(self):
        self.created = []
        self.updated = []

    async def set(self, key, value, ex=None):
        pass

    async def get(self, key):
        return None


def _patch_pipeline(monkeypatch, publish=None):
    published = []
    created = []
    updated = []

    async def fake_create_job(client, job_id, kind, source):
        created.append((job_id, kind, source))

    async def fake_update_job(client, job_id, *, status, result=None, error=None):
        updated.append((job_id, status, error))

    async def fake_publish_ingest_job(settings, job_id, *, kind, source):
        if publish is not None:
            await publish(job_id, kind, source)
        published.append((job_id, kind, source))

    monkeypatch.setattr("ask_the_docs.watcher.get_async_cache_client", lambda settings: RecordingAsyncRedis())
    monkeypatch.setattr("ask_the_docs.watcher.create_job", fake_create_job)
    monkeypatch.setattr("ask_the_docs.watcher.update_job", fake_update_job)
    monkeypatch.setattr("ask_the_docs.watcher.publish_ingest_job", fake_publish_ingest_job)
    return created, published, updated


async def test_watch_docs_dir_publishes_an_ingest_job_for_a_new_file(tmp_path, monkeypatch):
    settings = make_settings(docs_dir=tmp_path)
    (tmp_path / "a.pdf").write_bytes(b"%PDF-1.4")

    async def fake_awatch(path):
        yield {(Change.added, str(tmp_path / "a.pdf"))}

    monkeypatch.setattr("ask_the_docs.watcher.awatch", fake_awatch)
    created, published, _ = _patch_pipeline(monkeypatch)

    await watch_docs_dir(settings)

    assert [(kind, source) for _, kind, source in published] == [("ingest", "a.pdf")]


async def test_watch_docs_dir_publishes_an_ingest_job_for_a_modified_file(tmp_path, monkeypatch):
    settings = make_settings(docs_dir=tmp_path)
    (tmp_path / "a.txt").write_text("hello")

    async def fake_awatch(path):
        yield {(Change.modified, str(tmp_path / "a.txt"))}

    monkeypatch.setattr("ask_the_docs.watcher.awatch", fake_awatch)
    created, published, _ = _patch_pipeline(monkeypatch)

    await watch_docs_dir(settings)

    assert [(kind, source) for _, kind, source in published] == [("ingest", "a.txt")]


async def test_watch_docs_dir_ignores_deleted_files(tmp_path, monkeypatch):
    settings = make_settings(docs_dir=tmp_path)

    async def fake_awatch(path):
        yield {(Change.deleted, str(tmp_path / "a.pdf"))}

    monkeypatch.setattr("ask_the_docs.watcher.awatch", fake_awatch)
    created, published, _ = _patch_pipeline(monkeypatch)

    await watch_docs_dir(settings)

    assert published == []


async def test_watch_docs_dir_ignores_disallowed_extensions(tmp_path, monkeypatch):
    settings = make_settings(docs_dir=tmp_path)
    (tmp_path / "notes.swp").write_text("junk")

    async def fake_awatch(path):
        yield {(Change.added, str(tmp_path / "notes.swp"))}

    monkeypatch.setattr("ask_the_docs.watcher.awatch", fake_awatch)
    created, published, _ = _patch_pipeline(monkeypatch)

    await watch_docs_dir(settings)

    assert published == []


async def test_watch_docs_dir_publishes_once_per_file_per_batch(tmp_path, monkeypatch):
    """A single save often fires more than one filesystem event for the same
    path within one debounced batch -- that must not enqueue two jobs."""
    settings = make_settings(docs_dir=tmp_path)
    (tmp_path / "a.pdf").write_bytes(b"%PDF-1.4")

    async def fake_awatch(path):
        yield {
            (Change.added, str(tmp_path / "a.pdf")),
            (Change.modified, str(tmp_path / "a.pdf")),
        }

    monkeypatch.setattr("ask_the_docs.watcher.awatch", fake_awatch)
    created, published, _ = _patch_pipeline(monkeypatch)

    await watch_docs_dir(settings)

    assert [(kind, source) for _, kind, source in published] == [("ingest", "a.pdf")]


async def test_watch_docs_dir_creates_a_job_record_before_publishing(tmp_path, monkeypatch):
    """Mirrors the API-triggered path (api.py's _create_and_publish_job):
    a job record must exist in Redis before the queue message goes out, or
    GET /v1/jobs/{id} 404s for watcher-triggered ingests."""
    settings = make_settings(docs_dir=tmp_path)
    (tmp_path / "a.pdf").write_bytes(b"%PDF-1.4")

    async def fake_awatch(path):
        yield {(Change.added, str(tmp_path / "a.pdf"))}

    monkeypatch.setattr("ask_the_docs.watcher.awatch", fake_awatch)
    created, published, _ = _patch_pipeline(monkeypatch)

    await watch_docs_dir(settings)

    assert len(created) == 1
    assert len(published) == 1
    job_id, kind, source = created[0]
    assert (kind, source) == ("ingest", "a.pdf")
    assert published[0] == (job_id, "ingest", "a.pdf")


async def test_watch_docs_dir_marks_job_failed_and_continues_after_publish_error(tmp_path, monkeypatch):
    """A single bad publish (e.g. NATS unreachable) must not kill the watch
    loop, mirroring worker.py's _consume_once: one bad event is isolated,
    the rest keep going."""
    settings = make_settings(docs_dir=tmp_path)
    (tmp_path / "a.pdf").write_bytes(b"%PDF-1.4")
    (tmp_path / "b.pdf").write_bytes(b"%PDF-1.4")

    async def failing_publish(job_id, kind, source):
        if source == "a.pdf":
            raise QueuePublishError("boom")

    async def fake_awatch(path):
        yield {(Change.added, str(tmp_path / "a.pdf"))}
        yield {(Change.added, str(tmp_path / "b.pdf"))}

    monkeypatch.setattr("ask_the_docs.watcher.awatch", fake_awatch)
    created, published, updated = _patch_pipeline(monkeypatch, publish=failing_publish)

    await watch_docs_dir(settings)

    assert [source for _, _, source in created] == ["a.pdf", "b.pdf"]
    assert [source for _, _, source in published] == ["b.pdf"]
    assert len(updated) == 1
    job_id, status, error = updated[0]
    assert status == "failed"
    assert error == "boom"
    assert job_id == created[0][0]


async def test_watch_docs_dir_skips_a_file_outside_docs_dir(tmp_path, monkeypatch):
    """A watched path that doesn't resolve under docs_dir (e.g. a symlink
    pointing outside it) must be skipped, not crash the loop."""
    settings = make_settings(docs_dir=tmp_path / "docs")
    settings.docs_dir.mkdir()
    outside = tmp_path / "outside.pdf"
    outside.write_bytes(b"%PDF-1.4")
    (settings.docs_dir / "a.pdf").write_bytes(b"%PDF-1.4")

    async def fake_awatch(path):
        yield {(Change.added, str(outside)), (Change.added, str(settings.docs_dir / "a.pdf"))}

    monkeypatch.setattr("ask_the_docs.watcher.awatch", fake_awatch)
    created, published, _ = _patch_pipeline(monkeypatch)

    await watch_docs_dir(settings)

    assert [(kind, source) for _, kind, source in published] == [("ingest", "a.pdf")]
