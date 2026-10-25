from pathlib import Path

import pytest
from pydantic import ValidationError

from research_assistant.config import ProviderConfig, Settings


def test_settings_defaults():
    settings = Settings(
        generator=ProviderConfig(provider="ollama", model="llama3.1:8b"),
        judge=ProviderConfig(provider="anthropic", model="claude-sonnet-5-5", api_key="x"),
    )
    assert settings.embeddings_backend == "ollama"
    assert settings.index_dir == Path("data/index")
    assert settings.docs_dir == Path("data/documents")


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
