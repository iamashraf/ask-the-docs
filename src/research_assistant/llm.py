"""LLMClient abstraction: one interface, swappable providers.

Used twice with independent config -- once as the generator (answer.py)
and once as the judge (evaluate.py).
"""

from typing import Protocol

import requests
from anthropic import Anthropic
from openai import OpenAI

from research_assistant.config import ProviderConfig

OLLAMA_CHAT_TIMEOUT_SECONDS = 60


class OllamaUnavailableError(RuntimeError):
    pass


class LLMClient(Protocol):
    """Interface every provider-specific client below implements."""

    def generate(self, prompt: str) -> str: ...


class OllamaLLMClient:
    """Talks to a local Ollama server over HTTP."""

    def __init__(self, model: str, base_url: str):
        self.model = model
        self.base_url = base_url

    def generate(self, prompt: str) -> str:
        try:
            response = requests.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                },
                timeout=OLLAMA_CHAT_TIMEOUT_SECONDS,
            )
        except (requests.ConnectionError, requests.Timeout) as exc:
            raise OllamaUnavailableError(
                f"Could not reach Ollama at {self.base_url}. "
                "Is it running? Check OLLAMA_BASE_URL / GENERATOR__BASE_URL."
            ) from exc

        response.raise_for_status()
        return response.json()["message"]["content"]


class AnthropicLLMClient:
    """Wraps the Anthropic SDK."""

    def __init__(self, model: str, api_key: str):
        self.model = model
        self._client = Anthropic(api_key=api_key)

    def generate(self, prompt: str) -> str:
        response = self._client.messages.create(
            model=self.model,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text


class OpenAILLMClient:
    """Wraps the OpenAI SDK."""

    def __init__(self, model: str, api_key: str):
        self.model = model
        self._client = OpenAI(api_key=api_key)

    def generate(self, prompt: str) -> str:
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content


def get_llm_client(config: ProviderConfig) -> LLMClient:
    """Factory: builds the client matching config.provider."""
    if config.provider == "ollama":
        return OllamaLLMClient(model=config.model, base_url=config.base_url or "http://localhost:11434")
    if config.provider == "anthropic":
        return AnthropicLLMClient(model=config.model, api_key=config.api_key.get_secret_value())
    if config.provider == "openai":
        return OpenAILLMClient(model=config.model, api_key=config.api_key.get_secret_value())
    raise ValueError(f"Unknown provider: {config.provider!r}")
