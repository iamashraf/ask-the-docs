import pytest

from ask_the_docs.jobs import JobStoreError, create_job, get_job, update_job


class FakeAsyncRedis:
    def __init__(self):
        self._store: dict[str, bytes] = {}

    async def set(self, key, value, ex=None):
        self._store[key] = value

    async def get(self, key):
        return self._store.get(key)


class BrokenAsyncRedis:
    async def set(self, key, value, ex=None):
        raise ConnectionError("down")

    async def get(self, key):
        raise ConnectionError("down")


async def test_create_and_get_job_roundtrips():
    client = FakeAsyncRedis()
    await create_job(client, "job-1", kind="ingest", source="a.pdf")

    job = await get_job(client, "job-1")

    assert job["job_id"] == "job-1"
    assert job["kind"] == "ingest"
    assert job["source"] == "a.pdf"
    assert job["status"] == "queued"
    assert job["result"] is None
    assert job["error"] is None


async def test_get_job_returns_none_for_unknown_id():
    client = FakeAsyncRedis()
    assert await get_job(client, "nope") is None


async def test_update_job_changes_status_and_result():
    client = FakeAsyncRedis()
    await create_job(client, "job-1", kind="index", source=None)

    await update_job(client, "job-1", status="done", result={"documents_ingested": 3})

    job = await get_job(client, "job-1")
    assert job["status"] == "done"
    assert job["result"] == {"documents_ingested": 3}


async def test_update_job_is_a_noop_for_an_unknown_id():
    client = FakeAsyncRedis()
    await update_job(client, "nope", status="done")  # must not raise
    assert await get_job(client, "nope") is None


async def test_create_job_wraps_redis_failures():
    with pytest.raises(JobStoreError):
        await create_job(BrokenAsyncRedis(), "job-1", kind="ingest", source="a.pdf")
