from pathlib import Path

import pytest
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
    assert settings.otel_exporter_endpoint == "http://tempo:4317"
    assert settings.metrics_port == 9100


def test_settings_rejects_missing_api_key_for_anthropic():
    with pytest.raises(ValidationError):
        Settings(
            generator=ProviderConfig(provider="ollama", model="llama3.1:8b"),
            judge=ProviderConfig(provider="anthropic", model="claude-sonnet-5-5"),
        )


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
