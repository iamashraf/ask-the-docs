import os
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def base_url() -> str:
    return os.environ.get("E2E_BASE_URL", "http://localhost:8000")


@pytest.fixture(scope="session")
def docs_dir() -> Path:
    path = Path(os.environ.get("E2E_DOCS_DIR", Path(__file__).parent.parent / "data" / "documents"))
    path.mkdir(parents=True, exist_ok=True)
    return path
