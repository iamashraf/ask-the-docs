"""Application configuration, loaded once from the environment/.env file."""

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
    huggingface_embedding_model: str = "BAAI/bge-small-en-v1.5"

    index_dir: Path = Path("data/index")
    docs_dir: Path = Path("data/documents")

    model_config = SettingsConfigDict(
        env_file=".env", env_nested_delimiter="__", extra="ignore"
    )


@lru_cache
def get_settings() -> Settings:
    """Cached so env/.env is parsed once per process, not on every call."""
    return Settings()
