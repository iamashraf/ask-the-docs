"""Job status tracking for async ingestion, stored in Redis."""

import json
from datetime import UTC, datetime
from typing import Literal

import redis.asyncio as redis

from ask_the_docs.config import Settings

JOB_TTL_SECONDS = 86400
JOBS_INDEX_KEY = "jobs:index"
JobStatus = Literal["queued", "processing", "done", "failed"]


class JobStoreError(RuntimeError):
    pass


def get_async_cache_client(settings: Settings) -> redis.Redis:
    return redis.Redis.from_url(settings.cache_url)


def _key(job_id: str) -> str:
    return f"job:{job_id}"


async def create_job(client, job_id: str, kind: str, source: str | None) -> None:
    created_at = datetime.now(UTC)
    now = created_at.isoformat()
    payload = {
        "job_id": job_id,
        "kind": kind,
        "source": source,
        "status": "queued",
        "result": None,
        "error": None,
        "created_at": now,
        "updated_at": now,
    }
    try:
        await client.set(_key(job_id), json.dumps(payload), ex=JOB_TTL_SECONDS)
        await client.zadd(JOBS_INDEX_KEY, {job_id: created_at.timestamp()})
    except Exception as exc:
        raise JobStoreError(f"Failed to create job {job_id}") from exc


async def get_job(client, job_id: str) -> dict | None:
    raw = await client.get(_key(job_id))
    return json.loads(raw) if raw is not None else None


async def update_job(
    client, job_id: str, *, status: JobStatus, result: dict | None = None, error: str | None = None
) -> None:
    existing = await get_job(client, job_id)
    if existing is None:
        return
    existing.update(status=status, result=result, error=error, updated_at=datetime.now(UTC).isoformat())
    await client.set(_key(job_id), json.dumps(existing), ex=JOB_TTL_SECONDS)


async def list_jobs(client, limit: int = 50) -> list[dict]:
    """Most-recent-first; prunes index entries whose job hash already expired."""
    if limit <= 0:
        return []
    job_ids = await client.zrevrange(JOBS_INDEX_KEY, 0, limit - 1)
    jobs = []
    for raw_id in job_ids:
        job_id = raw_id.decode() if isinstance(raw_id, bytes) else raw_id
        job = await get_job(client, job_id)
        if job is None:
            await client.zrem(JOBS_INDEX_KEY, job_id)
            continue
        jobs.append(job)
    return jobs
