"""One-time startup seeding: ingests a fixed list of sources on first
server start. A marker file in index_dir prevents re-ingesting (and
duplicating chunks, since there's no dedup yet) on every restart."""

import logging
from pathlib import Path

import yaml

from ask_the_docs.config import Settings
from ask_the_docs.index import build_index
from ask_the_docs.ingest import dispatch

SEED_SOURCES_PATH = Path("seed_sources.yaml")

logger = logging.getLogger(__name__)


def load_seed_sources(path: Path) -> list[str]:
    data = yaml.safe_load(path.read_text())
    return list(data) if data else []


def seed_marker_path(settings: Settings) -> Path:
    return settings.index_dir / ".seeded"


def seed_if_needed(settings: Settings, seed_sources_path: Path = SEED_SOURCES_PATH) -> int:
    marker = seed_marker_path(settings)
    if marker.exists():
        logger.info("Seed marker already present, skipping startup seeding.")
        return 0

    sources = load_seed_sources(seed_sources_path)
    logger.info("Seeding %d source(s) from %s", len(sources), seed_sources_path)

    docs = []
    for i, source in enumerate(sources, start=1):
        logger.info("Seeding %d/%d: %s", i, len(sources), source)
        try:
            docs.extend(dispatch(source))
        except Exception as exc:
            logger.warning("Failed to seed %s: %s", source, exc)
            continue

    if docs:
        logger.info("Chunking and embedding %d document(s)...", len(docs))
        build_index(docs, settings)

    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text("seeded\n")
    logger.info("Seeding complete: %d document(s) ingested.", len(docs))
    return len(docs)
