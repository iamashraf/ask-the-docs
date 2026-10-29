"""NATS JetStream client: publishes ingestion jobs for the worker to consume."""

import json

import nats
from opentelemetry import propagate

from ask_the_docs.config import Settings

INGEST_JOBS_STREAM = "INGEST_JOBS"
INGEST_JOBS_SUBJECT = "ingest.jobs"


class QueuePublishError(RuntimeError):
    pass


async def ensure_stream(js) -> None:
    await js.add_stream(name=INGEST_JOBS_STREAM, subjects=[INGEST_JOBS_SUBJECT])


async def publish_ingest_job(settings: Settings, job_id: str, kind: str, source: str | None) -> None:
    carrier: dict[str, str] = {}
    propagate.inject(carrier)
    payload = {"job_id": job_id, "kind": kind, "source": source, "traceparent": carrier.get("traceparent")}
    try:
        nc = await nats.connect(settings.nats_url)
        try:
            js = nc.jetstream()
            await ensure_stream(js)
            await js.publish(INGEST_JOBS_SUBJECT, json.dumps(payload).encode())
        finally:
            await nc.close()
    except Exception as exc:
        raise QueuePublishError(f"Failed to publish job {job_id} to the queue") from exc
