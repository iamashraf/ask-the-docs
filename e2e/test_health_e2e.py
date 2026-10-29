import requests


def test_health_endpoint_is_reachable(base_url):
    response = requests.get(f"{base_url}/health", timeout=10)
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
