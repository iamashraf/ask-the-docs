import time
import uuid

import requests


def test_reingesting_an_unchanged_source_does_not_invalidate_the_answer_cache(base_url, docs_dir):
    """Raw ingest latency is too fast and noisy on this stack (~0.15-0.25s
    for a one-sentence doc either way) to tell a dedup skip apart from a
    real re-embed -- an earlier version of this test used a 2.0s threshold
    that passed regardless of whether dedup fired at all. The answer
    cache gives a much wider, reliable margin: a cache hit is Redis-speed
    (milliseconds); a cache miss means a real LLM call (seconds on this
    stack). The answer cache key includes the index generation, which
    only bumps on a real write -- so an unchanged re-ingest keeping the
    second /v1/ask fast is a real proof dedup skipped the write."""
    marker = uuid.uuid4().hex[:8]
    fixture_path = docs_dir / f"e2e-idempotent-{marker}.txt"
    fixture_path.write_text(
        f"The fictional Kethrin-{marker} protocol was invented in the year 3042 by Dr. Yevra Sol."
    )
    query = f"In what year was the fictional Kethrin-{marker} protocol invented?"
    try:
        assert requests.post(f"{base_url}/v1/ingest", json={"source": fixture_path.name}, timeout=60).status_code == 200

        # Primes the answer cache -- a real LLM call, several seconds.
        first = requests.post(f"{base_url}/v1/ask", json={"query": query}, timeout=60)
        assert first.status_code == 200
        assert "3042" in first.json()["answer"]

        # Unchanged re-ingest: dedup must skip it entirely, so the index
        # generation (part of the answer cache key) must not bump.
        assert requests.post(f"{base_url}/v1/ingest", json={"source": fixture_path.name}, timeout=60).status_code == 200

        start = time.monotonic()
        second = requests.post(f"{base_url}/v1/ask", json={"query": query}, timeout=60)
        elapsed = time.monotonic() - start

        assert second.status_code == 200
        assert second.json()["answer"] == first.json()["answer"]
        # A cache hit is milliseconds; a real regeneration (what happens
        # if dedup incorrectly bumped the generation) takes several
        # seconds on this stack -- 3s is a wide, safe margin between the
        # two, not a guess (see test_ask_e2e.py's real timings: 6-12s for
        # an uncached /v1/ask).
        assert elapsed < 3.0, (
            f"identical question took {elapsed:.2f}s after an unchanged re-ingest -- "
            "the answer cache was invalidated, meaning dedup skip may not have fired"
        )
    finally:
        fixture_path.unlink(missing_ok=True)
