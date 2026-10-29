import uuid

import requests


def test_reingesting_changed_content_is_reflected_in_a_fresh_answer(base_url, docs_dir):
    marker = uuid.uuid4().hex[:8]
    fixture_path = docs_dir / f"e2e-replace-{marker}.txt"

    fixture_path.write_text(f"The fictional Hollow-{marker} device was invented in 1850.")
    try:
        assert requests.post(f"{base_url}/v1/ingest", json={"source": fixture_path.name}, timeout=60).status_code == 200
        first_answer = requests.post(
            f"{base_url}/v1/ask",
            json={"query": f"In what year was the fictional Hollow-{marker} device invented?"},
            timeout=60,
        ).json()
        assert "1850" in first_answer["answer"]

        fixture_path.write_text(f"The fictional Hollow-{marker} device was invented in 1962.")
        assert requests.post(f"{base_url}/v1/ingest", json={"source": fixture_path.name}, timeout=60).status_code == 200
        second_answer = requests.post(
            f"{base_url}/v1/ask",
            json={"query": f"In what year was the fictional Hollow-{marker} device invented?"},
            timeout=60,
        ).json()
        assert "1962" in second_answer["answer"]
        assert "1850" not in second_answer["answer"]
    finally:
        fixture_path.unlink(missing_ok=True)
