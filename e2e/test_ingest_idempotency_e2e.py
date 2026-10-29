import time
import uuid

import requests


def test_reingesting_an_unchanged_source_is_much_faster_than_the_first_ingest(base_url, docs_dir):
    marker = uuid.uuid4().hex[:8]
    fixture_path = docs_dir / f"e2e-idempotent-{marker}.txt"
    fixture_path.write_text(
        f"The fictional Kethrin-{marker} protocol was invented in the year 3042 by Dr. Yevra Sol."
    )
    try:
        first = requests.post(
            f"{base_url}/v1/ingest", json={"source": fixture_path.name}, timeout=60
        )
        assert first.status_code == 200
        start = time.monotonic()
        second = requests.post(
            f"{base_url}/v1/ingest", json={"source": fixture_path.name}, timeout=60
        )
        elapsed = time.monotonic() - start

        assert second.status_code == 200
        # A real (non-skipped) ingest involves a real embedding call; a
        # skipped no-op is a handful of local Chroma reads. 2s is generous
        # slack above local-network/Chroma overhead, well below any real
        # embedding call's latency.
        assert elapsed < 2.0, f"unchanged re-ingest took {elapsed:.2f}s -- dedup skip may not have fired"
    finally:
        fixture_path.unlink(missing_ok=True)
