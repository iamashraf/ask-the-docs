import uuid

import requests

from conftest import wait_for_job


def test_ask_returns_a_grounded_answer_with_a_citation(base_url, docs_dir):
    marker = uuid.uuid4().hex[:8]
    fixture_path = docs_dir / f"e2e-grounded-{marker}.txt"
    fixture_path.write_text(f"The fictional Qwerzil-{marker} tower has exactly {7 + len(marker)} floors.")
    try:
        ingest = requests.post(f"{base_url}/v1/ingest", json={"source": fixture_path.name}, timeout=60)
        assert ingest.status_code == 202
        assert wait_for_job(base_url, ingest.json()["job_id"])["status"] == "done"
        response = requests.post(
            f"{base_url}/v1/ask",
            json={"query": f"How many floors does the fictional Qwerzil-{marker} tower have?"},
            timeout=60,
        )
        body = response.json()
        assert response.status_code == 200
        assert str(7 + len(marker)) in body["answer"]
        assert len(body["citations"]) > 0
    finally:
        fixture_path.unlink(missing_ok=True)


def test_ask_refuses_a_question_unrelated_to_anything_ingested(base_url):
    response = requests.post(
        f"{base_url}/v1/ask",
        json={"query": "What is the maximum flight speed of a fictional Blorthak space whale on planet Zyn-9?"},
        timeout=60,
    )
    body = response.json()
    assert response.status_code == 200
    assert body["answer"] == "I don't have enough information in the available documents to answer that."
    assert body["citations"] == []
