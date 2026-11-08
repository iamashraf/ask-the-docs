"""NATS JetStream client: publishes ingestion jobs for the worker to consume."""

import asyncio
import json
from datetime import UTC, datetime

import nats
from opentelemetry import propagate

from ask_the_docs.config import Settings

INGEST_JOBS_STREAM = "INGEST_JOBS"
INGEST_JOBS_SUBJECT = "ingest.jobs"


class QueuePublishError(RuntimeError):
    pass


async def ensure_stream(js) -> None:
    await js.add_stream(name=INGEST_JOBS_STREAM, subjects=[INGEST_JOBS_SUBJECT])


_nc = None
_js = None
_connect_lock = asyncio.Lock()


async def _get_jetstream(settings: Settings):
    """One connection (and one add_stream) per process; reconnects if it was closed."""
    global _nc, _js
    async with _connect_lock:
        if _nc is None or _nc.is_closed:
            nc = await nats.connect(settings.nats_url)
            js = nc.jetstream()
            try:
                await ensure_stream(js)
            except Exception:
                await nc.close()
                raise
            _nc, _js = nc, js
        return _js


async def close_publisher() -> None:
    global _nc, _js
    nc, _nc, _js = _nc, None, None
    if nc is not None and not nc.is_closed:
        await nc.close()


async def publish_ingest_job(settings: Settings, job_id: str, kind: str, source: str | None) -> None:
    carrier: dict[str, str] = {}
    propagate.inject(carrier)
    payload = {
        "job_id": job_id,
        "kind": kind,
        "source": source,
        "traceparent": carrier.get("traceparent"),
        "enqueued_at": datetime.now(UTC).isoformat(),
    }
    try:
        js = await _get_jetstream(settings)
        await js.publish(INGEST_JOBS_SUBJECT, json.dumps(payload).encode())
    except Exception as exc:
        raise QueuePublishError(f"Failed to publish job {job_id} to the queue") from exc
