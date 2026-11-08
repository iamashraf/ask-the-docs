from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from ask_the_docs.config import REFUSAL_MESSAGE, ProviderConfig, Settings


def test_settings_defaults():
    settings = Settings(
        generator=ProviderConfig(provider="ollama", model="llama3.1:8b"),
        judge=ProviderConfig(provider="anthropic", model="claude-sonnet-5-5", api_key="x"),
    )
    assert settings.embeddings_backend == "ollama"
    assert settings.ollama_base_url == "http://localhost:11434"
    assert settings.index_dir == Path("data/index")
    assert settings.docs_dir == Path("data/documents")
    assert settings.max_url_fetch_size_mb == 25
    assert settings.otel_exporter_endpoint == "http://otel-collector:4317"


def test_settings_allows_anthropic_without_api_key():
    settings = Settings(
        generator=ProviderConfig(provider="anthropic", model="claude-haiku-5-5"),
        judge=ProviderConfig(provider="anthropic", model="claude-sonnet-5-5"),
    )
    assert settings.generator.api_key is None


def test_provider_config_rejects_non_positive_max_tokens():
    with pytest.raises(ValueError):
        ProviderConfig(provider="anthropic", model="claude-haiku-5-5", max_tokens=0)


def test_settings_rejects_missing_api_key_for_openai():
    with pytest.raises(ValidationError):
        Settings(
            generator=ProviderConfig(provider="openai", model="gpt-4o-mini"),
            judge=ProviderConfig(provider="anthropic", model="claude-sonnet-5-5", api_key="x"),
        )


def test_settings_allows_ollama_without_api_key():
    settings = Settings(
        generator=ProviderConfig(provider="ollama", model="llama3.1:8b"),
        judge=ProviderConfig(provider="ollama", model="llama3.1:70b"),
    )
    assert settings.generator.api_key is None


def test_provider_config_rejects_invalid_provider():
    with pytest.raises(ValidationError):
        ProviderConfig(provider="bogus", model="x")


def test_settings_hybrid_defaults():
    settings = Settings(
        generator=ProviderConfig(provider="ollama", model="llama3.1:8b"),
        judge=ProviderConfig(provider="anthropic", model="claude-sonnet-5-5", api_key="x"),
    )
    assert settings.hybrid_alpha == 0.5
    assert settings.rerank_enabled is True
    assert settings.query_rewrite_enabled is True
    assert settings.candidate_pool_size == 50
    assert REFUSAL_MESSAGE in settings.system_prompt_template
    assert settings.cache_url == "redis://localhost:6379/0"
    assert settings.history_max_turns == 6


def test_settings_rejects_hybrid_alpha_out_of_range():
    with pytest.raises(ValidationError):
        Settings(
            generator=ProviderConfig(provider="ollama", model="llama3.1:8b"),
            judge=ProviderConfig(provider="anthropic", model="claude-sonnet-5-5", api_key="x"),
            hybrid_alpha=1.5,
        )


def test_settings_accepts_hybrid_alpha_boundary_values():
    low = Settings(
        generator=ProviderConfig(provider="ollama", model="llama3.1:8b"),
        judge=ProviderConfig(provider="anthropic", model="claude-sonnet-5-5", api_key="x"),
        hybrid_alpha=0.0,
    )
    high = Settings(
        generator=ProviderConfig(provider="ollama", model="llama3.1:8b"),
        judge=ProviderConfig(provider="anthropic", model="claude-sonnet-5-5", api_key="x"),
        hybrid_alpha=1.0,
    )
    assert low.hybrid_alpha == 0.0
    assert high.hybrid_alpha == 1.0


def test_settings_accepts_minimum_candidate_pool_size():
    settings = Settings(
        generator=ProviderConfig(provider="ollama", model="llama3.1:8b"),
        judge=ProviderConfig(provider="anthropic", model="claude-sonnet-5-5", api_key="x"),
        candidate_pool_size=1,
    )
    assert settings.candidate_pool_size == 1


def test_settings_rejects_non_positive_candidate_pool_size():
    with pytest.raises(ValidationError):
        Settings(
            generator=ProviderConfig(provider="ollama", model="llama3.1:8b"),
            judge=ProviderConfig(provider="anthropic", model="claude-sonnet-5-5", api_key="x"),
            candidate_pool_size=0,
        )


def test_settings_accepts_zero_history_max_turns():
    settings = Settings(
        generator=ProviderConfig(provider="ollama", model="llama3.1:8b"),
        judge=ProviderConfig(provider="anthropic", model="claude-sonnet-5-5", api_key="x"),
        history_max_turns=0,
    )
    assert settings.history_max_turns == 0


def test_settings_rejects_negative_history_max_turns():
    with pytest.raises(ValidationError):
        Settings(
            generator=ProviderConfig(provider="ollama", model="llama3.1:8b"),
            judge=ProviderConfig(provider="anthropic", model="claude-sonnet-5-5", api_key="x"),
            history_max_turns=-1,
        )


def test_settings_loads_nested_fields_from_env_vars(monkeypatch):
    """Settings is always configured through real process env vars -- in
    containers, docker-compose's env_file: injects .env.local's contents
    this way before the process starts. This proves the double-underscore
    nested delimiter actually works, not just assumed."""
    monkeypatch.setenv("GENERATOR__PROVIDER", "ollama")
    monkeypatch.setenv("GENERATOR__MODEL", "llama3.1:8b")
    monkeypatch.setenv("GENERATOR__BASE_URL", "http://ollama:11434")
    monkeypatch.setenv("JUDGE__PROVIDER", "anthropic")
    monkeypatch.setenv("JUDGE__MODEL", "claude-sonnet-5-5")
    monkeypatch.setenv("JUDGE__API_KEY", "sk-test")
    monkeypatch.setenv("EMBEDDINGS_BACKEND", "huggingface")

    settings = Settings()

    assert settings.generator.provider == "ollama"
    assert settings.generator.base_url == "http://ollama:11434"
    assert settings.judge.provider == "anthropic"
    assert settings.judge.api_key.get_secret_value() == "sk-test"
    assert settings.embeddings_backend == "huggingface"


def test_settings_graph_defaults():
    settings = Settings(
        generator=ProviderConfig(provider="ollama", model="llama3.1:8b"),
        judge=ProviderConfig(provider="anthropic", model="claude-sonnet-5-5", api_key="x"),
    )
    assert settings.graph_enabled is False
    assert settings.neo4j_uri == "bolt://localhost:7687"
    assert settings.neo4j_user == "neo4j"
    assert settings.neo4j_password.get_secret_value() == "neo4j"
    assert settings.entity_resolution_similarity_threshold == 0.85
    assert settings.graph_traversal_hop_depth == 2
    assert settings.global_search_top_communities == 3


def test_existing_settings_construction_unaffected_by_graph_fields():
    # Every other test file constructs Settings with no graph kwargs; graph_enabled
    # must stay False so retrieve()'s classification branch (Task 13) stays off for them.
    settings = Settings(
        generator=ProviderConfig(provider="ollama", model="llama3.1:8b"),
        judge=ProviderConfig(provider="anthropic", model="claude-sonnet-5-5", api_key="x"),
    )
    assert settings.graph_enabled is False


def _graph_settings(**overrides) -> Settings:
    return Settings(
        generator=ProviderConfig(provider="ollama", model="llama3.1:8b"),
        judge=ProviderConfig(provider="anthropic", model="claude-sonnet-5-5", api_key="x"),
        **overrides,
    )


@pytest.mark.parametrize("password", ["", "neo4j", "ask-the-docs-graph"])
def test_graph_enabled_rejects_blank_or_well_known_neo4j_password(password):
    with pytest.raises(ValidationError, match="NEO4J_PASSWORD"):
        _graph_settings(graph_enabled=True, neo4j_password=password)


def test_graph_enabled_accepts_a_custom_neo4j_password():
    settings = _graph_settings(graph_enabled=True, neo4j_password="a-long-unique-secret")

    assert settings.neo4j_password.get_secret_value() == "a-long-unique-secret"


def test_graph_disabled_ignores_the_neo4j_password():
    assert _graph_settings(graph_enabled=False, neo4j_password="").graph_enabled is False


def test_compose_persists_nats_jetstream_and_points_app_and_worker_at_it():
    compose = yaml.safe_load((Path(__file__).parent.parent / "docker-compose.yml").read_text())
    nats_service = compose["services"]["nats"]
    command = nats_service["command"]
    store_dir = command[command.index("--store_dir") + 1]

    volume = next(v for v in nats_service["volumes"] if v.split(":")[1] == store_dir)
    assert volume.split(":")[0] in compose["volumes"]
    for name in ("app", "worker"):
        assert "NATS_URL=nats://nats:4222" in compose["services"][name]["environment"]


def test_defaults_use_anthropic_for_generator_and_judge(monkeypatch):
    for name in ("GENERATOR__PROVIDER", "GENERATOR__MODEL", "JUDGE__PROVIDER", "JUDGE__MODEL"):
        monkeypatch.delenv(name, raising=False)
    settings = Settings(_env_file=None)
    assert (settings.generator.provider, settings.generator.model) == ("anthropic", "claude-haiku-5-5")
    assert (settings.judge.provider, settings.judge.model) == ("anthropic", "claude-sonnet-5-5")


def test_environment_switches_generator_back_to_ollama(monkeypatch):
    monkeypatch.setenv("GENERATOR__PROVIDER", "ollama")
    monkeypatch.setenv("GENERATOR__MODEL", "llama3.1:8b")
    settings = Settings(_env_file=None)
    assert (settings.generator.provider, settings.generator.model) == ("ollama", "llama3.1:8b")
    assert settings.judge.provider == "anthropic"
