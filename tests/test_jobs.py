from datetime import datetime

import pytest

from ask_the_docs.jobs import JobStoreError, create_job, get_job, list_jobs, update_job


class FakeAsyncRedis:
    def __init__(self):
        self._store: dict[str, bytes] = {}
        self._zsets: dict[str, dict[str, float]] = {}

    async def set(self, key, value, ex=None):
        self._store[key] = value

    async def get(self, key):
        return self._store.get(key)

    async def zadd(self, key, mapping):
        self._zsets.setdefault(key, {}).update(mapping)

    async def zrevrange(self, key, start, end):
        # Real redis-asyncio (no decode_responses) returns bytes -- match that
        # so tests exercise the same decode path production code takes.
        members = sorted(self._zsets.get(key, {}).items(), key=lambda kv: kv[1], reverse=True)
        ids = [member.encode() for member, _score in members]
        return ids[start:] if end == -1 else ids[start : end + 1]

    async def zrem(self, key, *members):
        zset = self._zsets.get(key, {})
        for member in members:
            zset.pop(member, None)


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


async def test_list_jobs_returns_most_recent_first():
    client = FakeAsyncRedis()
    await create_job(client, "job-1", kind="ingest", source="a.pdf")
    await create_job(client, "job-2", kind="ingest", source="b.pdf")
    await create_job(client, "job-3", kind="index", source=None)
    # Pin deterministic scores -- create_job's real epoch timestamps could tie
    # when the three calls land within the same clock tick.
    await client.zadd("jobs:index", {"job-1": 100, "job-2": 200, "job-3": 300})

    jobs = await list_jobs(client)

    assert [job["job_id"] for job in jobs] == ["job-3", "job-2", "job-1"]


async def test_list_jobs_respects_limit():
    client = FakeAsyncRedis()
    await create_job(client, "job-1", kind="ingest", source="a.pdf")
    await create_job(client, "job-2", kind="ingest", source="b.pdf")
    await client.zadd("jobs:index", {"job-1": 100, "job-2": 200})

    jobs = await list_jobs(client, limit=1)

    assert [job["job_id"] for job in jobs] == ["job-2"]


async def test_list_jobs_prunes_expired_entries_from_the_index():
    client = FakeAsyncRedis()
    await create_job(client, "job-1", kind="ingest", source="a.pdf")
    await create_job(client, "job-2", kind="ingest", source="b.pdf")
    # Simulate the 24h TTL expiring the hash while the index entry lingers.
    del client._store["job:job-1"]

    jobs = await list_jobs(client)

    assert [job["job_id"] for job in jobs] == ["job-2"]
    assert "job-1" not in client._zsets["jobs:index"]


async def test_list_jobs_returns_empty_list_for_an_empty_index():
    client = FakeAsyncRedis()
    assert await list_jobs(client) == []


async def test_list_jobs_returns_empty_list_for_a_non_positive_limit():
    client = FakeAsyncRedis()
    await create_job(client, "job-1", kind="ingest", source="a.pdf")

    assert await list_jobs(client, limit=0) == []


async def test_create_job_indexes_by_its_created_at_epoch_seconds():
    client = FakeAsyncRedis()
    await create_job(client, "job-1", kind="ingest", source="a.pdf")

    job = await get_job(client, "job-1")
    expected_score = datetime.fromisoformat(job["created_at"]).timestamp()
    assert client._zsets["jobs:index"]["job-1"] == expected_score
