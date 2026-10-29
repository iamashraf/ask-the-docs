import uuid

import requests


def test_bulk_index_ingests_a_large_number_of_chunks_without_timing_out(base_url, docs_dir):
    marker = uuid.uuid4().hex[:8]
    fixture_paths = []
    # 30 distinct small docs -- well past INDEX_WRITE_BATCH_SIZE (64 chunks
    # total across sources; each doc here is short enough to be its own
    # single chunk, so 30 files ~= 30 chunks, enough to force at least one
    # real batched embed call without needing one giant document).
    for i in range(30):
        path = docs_dir / f"e2e-batch-{marker}-{i}.txt"
        path.write_text(f"Fictional fact {marker}-{i}: the moon of Vezrai-{i} orbits every {100 + i} days.")
        fixture_paths.append(path)

    try:
        response = requests.post(f"{base_url}/v1/index", timeout=180)
        assert response.status_code == 200
        body = response.json()
        assert body["documents_ingested"] >= 30
        assert body["files_found"] >= 30
    finally:
        for path in fixture_paths:
            path.unlink(missing_ok=True)
