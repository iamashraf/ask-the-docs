import os
import time
from pathlib import Path

import pytest
import requests


@pytest.fixture(scope="session")
def base_url() -> str:
    return os.environ.get("E2E_BASE_URL", "http://localhost:8000")


def wait_for_job(base_url: str, job_id: str, timeout: float = 120.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        response = requests.get(f"{base_url}/v1/jobs/{job_id}", timeout=10)
        assert response.status_code == 200, f"job {job_id} lookup failed: {response.status_code}"
        job = response.json()
        if job["status"] in ("done", "failed"):
            return job
        time.sleep(0.5)
    raise AssertionError(f"job {job_id} did not finish within {timeout}s")


@pytest.fixture(scope="session")
def docs_dir() -> Path:
    path = Path(os.environ.get("E2E_DOCS_DIR", Path(__file__).parent.parent / "data" / "documents"))
    path.mkdir(parents=True, exist_ok=True)
    return path
