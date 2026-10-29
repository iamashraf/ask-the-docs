"""File-watcher mode: observes docs_dir and enqueues an ingest job per
new/changed file over the existing async pipeline (jobs.py + queue.py +
worker.py), instead of ingesting directly. Mirrors api.py's
_create_and_publish_job so a watcher-triggered job is trackable via
GET /v1/jobs/{id} exactly like an API-triggered one."""

import logging
import uuid
from pathlib import Path

from watchfiles import Change, awatch

from ask_the_docs.config import Settings
from ask_the_docs.ingest import ALLOWED_EXTENSIONS
from ask_the_docs.jobs import create_job, get_async_cache_client, update_job
from ask_the_docs.queue import QueuePublishError, publish_ingest_job

logger = logging.getLogger(__name__)


async def _enqueue_ingest(settings: Settings, source: str) -> None:
    job_id = str(uuid.uuid4())
    client = get_async_cache_client(settings)
    await create_job(client, job_id, kind="ingest", source=source)
    try:
        await publish_ingest_job(settings, job_id, kind="ingest", source=source)
    except QueuePublishError as exc:
        logger.warning("Failed to publish watcher job %s for %s: %s", job_id, source, exc)
        try:
            await update_job(client, job_id, status="failed", error=str(exc))
        except Exception:
            logger.warning("Failed to mark job %s failed after a publish error", job_id, exc_info=True)


async def watch_docs_dir(settings: Settings) -> None:
    docs_dir = settings.docs_dir.resolve()
    async for changes in awatch(settings.docs_dir):
        seen_sources: set[str] = set()
        for change, raw_path in changes:
            if change == Change.deleted:
                continue
            path = Path(raw_path).resolve()
            if path.suffix.lower() not in ALLOWED_EXTENSIONS:
                continue
            try:
                source = str(path.relative_to(docs_dir))
            except ValueError:
                logger.warning("Ignoring change outside docs_dir: %s", path)
                continue
            if source in seen_sources:
                continue
            seen_sources.add(source)
            await _enqueue_ingest(settings, source)
