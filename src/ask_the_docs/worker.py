"""Async ingestion worker: consumes jobs from NATS JetStream and runs them
through the existing dispatch/build_index pipeline -- unchanged by this
module, so a job is exactly as idempotent/fail-open as a direct CLI call."""

import asyncio
import json
import logging

import nats
from nats.js.api import ConsumerConfig

from ask_the_docs.config import Settings, get_settings
from ask_the_docs.index import build_index
from ask_the_docs.ingest import dispatch, dispatch_all_under_docs_dir
from ask_the_docs.jobs import get_async_cache_client, update_job
from ask_the_docs.queue import INGEST_JOBS_SUBJECT, ensure_stream

logger = logging.getLogger(__name__)

DURABLE_CONSUMER_NAME = "ingest-workers"
# Comfortably above the bulk /v1/index job's own 180s timeout budget (see
# e2e/test_ingest_batch_e2e.py) so a long-running job's ack doesn't become
# eligible for redelivery while it is still in progress.
WORKER_ACK_WAIT_SECONDS = 300


async def process_job(settings: Settings, payload: dict) -> None:
    job_id = payload["job_id"]
    kind = payload["kind"]
    source = payload.get("source")
    client = get_async_cache_client(settings)
    await update_job(client, job_id, status="processing")
    try:
        if kind == "ingest":
            docs = dispatch(source)
            file_count = 1
        else:
            docs, file_count = dispatch_all_under_docs_dir(settings)
        if docs:
            build_index(docs, settings)
        result = {"documents_ingested": len(docs)}
        if kind == "index":
            result["files_found"] = file_count
        await update_job(client, job_id, status="done", result=result)
    except Exception as exc:
        logger.warning("Job %s failed: %s", job_id, exc, exc_info=True)
        await update_job(client, job_id, status="failed", error=str(exc))


async def _consume_once(settings: Settings, sub) -> None:
    """One fetch-and-process cycle, isolated from run_worker's infinite loop
    so a single bad message or a transient failure inside process_job can't
    kill the worker process -- the message is just left unacked for NATS to
    redeliver, and the loop moves on."""
    try:
        msgs = await sub.fetch(1, timeout=30)
    except TimeoutError:
        return
    for msg in msgs:
        try:
            payload = json.loads(msg.data)
        except Exception:
            logger.warning("Dropping an unparseable job message", exc_info=True)
            continue
        try:
            await process_job(settings, payload)
        except Exception:
            logger.warning(
                "process_job raised for job %s; leaving unacked for redelivery",
                payload.get("job_id"),
                exc_info=True,
            )
            continue
        await msg.ack()


async def run_worker(settings: Settings) -> None:
    nc = await nats.connect(settings.nats_url)
    js = nc.jetstream()
    await ensure_stream(js)
    sub = await js.pull_subscribe(
        INGEST_JOBS_SUBJECT,
        durable=DURABLE_CONSUMER_NAME,
        config=ConsumerConfig(ack_wait=WORKER_ACK_WAIT_SECONDS),
    )
    logger.info("Worker started, consuming %s", INGEST_JOBS_SUBJECT)
    while True:
        await _consume_once(settings, sub)


def main() -> None:
    asyncio.run(run_worker(get_settings()))
