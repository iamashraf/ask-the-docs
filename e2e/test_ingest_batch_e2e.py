import uuid

import requests

from conftest import wait_for_job


def test_bulk_index_ingests_a_large_number_of_chunks_without_timing_out(base_url, docs_dir):
    marker = uuid.uuid4().hex[:8]

    # Baseline first: docs_dir may already hold real content (it does on
    # this stack), so an exact-delta assertion needs a known starting
    # point, not an absolute "at least N" that a silently-empty fixture
    # could still satisfy via pre-existing files.
    baseline_response = requests.post(f"{base_url}/v1/index", timeout=180)
    baseline = wait_for_job(base_url, baseline_response.json()["job_id"], timeout=180)["result"]

    fixture_paths = []
    # 150 distinct one-chunk docs -- well past INDEX_WRITE_BATCH_SIZE (64),
    # forcing 3 real sub-batches (64 + 64 + 22) against live Ollama, not
    # just 1. A smaller count (e.g. 30) stays under the batch size and
    # never actually exercises the batching/timeout-scaling fix.
    for i in range(150):
        path = docs_dir / f"e2e-batch-{marker}-{i}.txt"
        path.write_text(f"Fictional fact {marker}-{i}: the moon of Vezrai-{i} orbits every {100 + i} days.")
        fixture_paths.append(path)

    try:
        response = requests.post(f"{base_url}/v1/index", timeout=180)
        assert response.status_code == 202
        job = wait_for_job(base_url, response.json()["job_id"], timeout=180)
        assert job["status"] == "done"
        body = job["result"]
        # Exact deltas, not "at least 150": a fixture that silently loads
        # as zero documents must be visible here, not hidden by whatever
        # else is already in docs_dir.
        assert body["documents_ingested"] == baseline["documents_ingested"] + 150
        assert body["files_found"] == baseline["files_found"] + 150
    finally:
        for path in fixture_paths:
            path.unlink(missing_ok=True)
