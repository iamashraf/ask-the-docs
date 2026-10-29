"""Application configuration, loaded once from the process environment.

Settings never reads a file itself -- docker-compose's `env_file:`
directive injects .env.local's contents as real environment variables
before the container starts, so this is the only mechanism that matters."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROVIDERS_REQUIRING_API_KEY = {"anthropic", "openai"}

REFUSAL_MESSAGE = "I don't have enough information in the available documents to answer that."

DEFAULT_SYSTEM_PROMPT_TEMPLATE = (
    "Answer the user's question using only the context they provide. "
    "Cite every claim with its [n] marker from the context.\n\n"
    "If the question is hypothetical, asks you to speculate, or asks about "
    "something not stated in the context, or if the context does not "
    f'contain the answer, respond with exactly: "{REFUSAL_MESSAGE}"'
)


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

    hybrid_alpha: float = Field(default=0.5, ge=0.0, le=1.0)
    rerank_enabled: bool = True
    query_rewrite_enabled: bool = True
    candidate_pool_size: int = Field(default=50, ge=1)
    system_prompt_template: str = DEFAULT_SYSTEM_PROMPT_TEMPLATE
    cache_url: str = "redis://localhost:6379/0"
    nats_url: str = "nats://localhost:4222"
    otel_exporter_endpoint: str = "http://jaeger:4317"

    graph_enabled: bool = False
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: SecretStr = SecretStr("neo4j")
    entity_resolution_similarity_threshold: float = Field(default=0.85, ge=0.0, le=1.0)
    graph_traversal_hop_depth: int = Field(default=2, ge=1)
    global_search_top_communities: int = Field(default=3, ge=1)

    model_config = SettingsConfigDict(env_nested_delimiter="__", extra="ignore")


@lru_cache
def get_settings() -> Settings:
    """Cached so the environment is parsed once per process, not on every call."""
    return Settings()
