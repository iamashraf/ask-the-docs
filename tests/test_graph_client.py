import neo4j

from research_assistant.config import ProviderConfig, Settings
from research_assistant.graph_client import _get_graph_client_cached, get_graph_client


def make_settings(**overrides) -> Settings:
    return Settings(
        generator=ProviderConfig(provider="ollama", model="llama3.1:8b"),
        judge=ProviderConfig(provider="anthropic", model="claude-sonnet-5-5", api_key="x"),
        **overrides,
    )


def test_get_graph_client_builds_from_settings(monkeypatch):
    captured = {}

    def fake_driver(uri, auth):
        captured["uri"] = uri
        captured["auth"] = auth
        return "fake-driver"

    monkeypatch.setattr(neo4j.GraphDatabase, "driver", fake_driver)
    _get_graph_client_cached.cache_clear()
    settings = make_settings(
        neo4j_uri="bolt://example:7687", neo4j_user="neo4j", neo4j_password="secret"
    )

    client = get_graph_client(settings)

    assert client == "fake-driver"
    assert captured["uri"] == "bolt://example:7687"
    assert captured["auth"] == ("neo4j", "secret")
