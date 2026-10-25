"""One-time startup seeding: ingests a fixed list of sources on first
server start. A marker file in index_dir prevents re-ingesting (and
duplicating chunks, since there's no dedup yet) on every restart."""

from pathlib import Path

import yaml

from research_assistant.config import Settings
from research_assistant.index import build_index
from research_assistant.ingest import dispatch

SEED_SOURCES_PATH = Path("seed_sources.yaml")


def load_seed_sources(path: Path) -> list[str]:
    data = yaml.safe_load(path.read_text())
    return list(data) if data else []


def seed_marker_path(settings: Settings) -> Path:
    return settings.index_dir / ".seeded"


def seed_if_needed(settings: Settings, seed_sources_path: Path = SEED_SOURCES_PATH) -> int:
    marker = seed_marker_path(settings)
    if marker.exists():
        return 0

    docs = []
    for source in load_seed_sources(seed_sources_path):
        try:
            docs.extend(dispatch(source))
        except Exception:
            continue
    if docs:
        build_index(docs, settings)

    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text("seeded\n")
    return len(docs)
