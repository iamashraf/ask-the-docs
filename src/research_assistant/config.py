"""Application configuration, loaded once from the process environment.

Settings never reads a file itself -- docker-compose's `env_file:`
directive injects .env.local's contents as real environment variables
before the container starts, so this is the only mechanism that matters."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROVIDERS_REQUIRING_API_KEY = {"anthropic", "openai"}


class ProviderConfig(BaseModel):
    """Connection details for one LLM role: which provider/model, and its credentials."""

    provider: Literal["ollama", "anthropic", "openai"]
    model: str
    api_key: SecretStr | None = None
    base_url: str | None = None

    @model_validator(mode="after")
    def check_api_key_present_for_hosted_providers(self) -> "ProviderConfig":
        """Hosted providers need a key; ollama is local and doesn't."""
        if self.provider in PROVIDERS_REQUIRING_API_KEY and self.api_key is None:
            raise ValueError(f"provider {self.provider!r} requires an api_key")
        return self


class Settings(BaseSettings):
    """App-wide config. `generator` and `judge` are independently-configured LLM roles."""

    generator: ProviderConfig
    judge: ProviderConfig

    embeddings_backend: Literal["ollama", "huggingface"] = "ollama"
    ollama_embedding_model: str = "nomic-embed-text"
    ollama_base_url: str = "http://localhost:11434"
    huggingface_embedding_model: str = "BAAI/bge-small-en-v1.5"

    index_dir: Path = Path("data/index")
    docs_dir: Path = Path("data/documents")
    max_file_size_mb: int = 25
    max_url_fetch_size_mb: int = 25
    url_fetch_timeout_seconds: int = 10

    model_config = SettingsConfigDict(env_nested_delimiter="__", extra="ignore")


@lru_cache
def get_settings() -> Settings:
    """Cached so the environment is parsed once per process, not on every call."""
    return Settings()
