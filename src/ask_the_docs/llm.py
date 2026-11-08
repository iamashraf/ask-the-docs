"""LLMClient abstraction: one interface, swappable providers.

Used twice with independent config -- once as the generator (answer.py)
and once as the judge (evaluate.py).
"""

import dataclasses
import json
import os
import re
import subprocess
from typing import Protocol

import requests
import anthropic
from anthropic import Anthropic
from openai import OpenAI

from ask_the_docs.config import ProviderConfig

FENCED_BLOCK_PATTERN = re.compile(r"^```(?:json)?\s*\n?(.*?)\n?```$", re.DOTALL)


def strip_code_fence(raw: str) -> str:
    """Small/local models often wrap JSON responses in markdown code fences; non-str input is left to json.loads to reject."""
    if not isinstance(raw, str):
        return raw
    match = FENCED_BLOCK_PATTERN.match(raw.strip())
    return match.group(1) if match else raw


def fence_untrusted(tag: str, text: str) -> str:
    """Wraps untrusted text in <tag> and escapes any open/close tag inside it so it cannot end the block early."""
    escaped = re.sub(rf"<(\s*/?\s*{re.escape(tag)}\s*)>", r"&lt;\1&gt;", text, flags=re.IGNORECASE)
    return f"<{tag}>\n{escaped}\n</{tag}>"


OLLAMA_CHAT_TIMEOUT_SECONDS = 60
# Ollama's own default idle-unload is 5m -- without an explicit keep_alive,
# a gap between requests longer than that forces the next call to pay a
# full model-load cost on top of generation.
OLLAMA_KEEP_ALIVE = "30m"
# Decode time scales with output length, and CPU-only decode measured at
# ~10-26 tok/s depending on context size -- an uncapped generation can run
# to 600+ tokens for a question that needed a short answer. 500 is generous
# for a cited, grounded explanation while bounding the worst-case tail.
OLLAMA_MAX_OUTPUT_TOKENS = 500


class LLMUnavailableError(RuntimeError):
    """A provider could not be reached or refused the call; the message is written for users."""


class OllamaUnavailableError(LLMUnavailableError):
    pass


class StructuredOutputError(RuntimeError):
    """Ollama's response didn't parse as the requested JSON schema -- distinct from OllamaUnavailableError."""


@dataclasses.dataclass
class GenerationResult:
    text: str
    input_tokens: int | None
    output_tokens: int | None
    response_model: str | None = None


@dataclasses.dataclass
class StructuredJSONResult:
    """Parsed JSON object from Ollama's schema-constrained output."""

    data: dict
    input_tokens: int | None
    output_tokens: int | None
    response_model: str | None = None


class LLMClient(Protocol):
    """Interface every provider-specific client below implements."""

    def generate(self, prompt: str, system: str | None = None) -> str: ...
    def generate_with_usage(self, prompt: str, system: str | None = None) -> GenerationResult: ...


class OllamaLLMClient:
    """Talks to a local Ollama server over HTTP."""

    def __init__(self, model: str, base_url: str, think: bool | None = None):
        self.model = model
        self.base_url = base_url
        self.think = think

    def generate(self, prompt: str, system: str | None = None) -> str:
        return self.generate_with_usage(prompt, system).text

    def generate_with_usage(self, prompt: str, system: str | None = None) -> GenerationResult:
        messages = []
        if system is not None:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        body = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "keep_alive": OLLAMA_KEEP_ALIVE,
            "options": {"num_predict": OLLAMA_MAX_OUTPUT_TOKENS},
        }
        if self.think is not None:
            body["think"] = self.think

        try:
            response = requests.post(
                f"{self.base_url}/api/chat",
                json=body,
                timeout=OLLAMA_CHAT_TIMEOUT_SECONDS,
            )
        except (requests.ConnectionError, requests.Timeout) as exc:
            raise OllamaUnavailableError(
                f"Could not reach Ollama at {self.base_url}. "
                "Is it running? Check OLLAMA_BASE_URL / GENERATOR__BASE_URL."
            ) from exc

        response.raise_for_status()
        body = response.json()
        return GenerationResult(
            text=body["message"]["content"],
            input_tokens=body.get("prompt_eval_count"),
            output_tokens=body.get("eval_count"),
            response_model=body.get("model"),
        )

    def generate_structured(
        self, prompt: str, schema: dict, system: str | None = None
    ) -> StructuredJSONResult:
        """Like generate_with_usage, but constrains Ollama's output to `schema` via the `format` field."""
        messages = []
        if system is not None:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        body = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "format": schema,
            "keep_alive": OLLAMA_KEEP_ALIVE,
            "options": {"num_predict": OLLAMA_MAX_OUTPUT_TOKENS},
        }
        if self.think is not None:
            body["think"] = self.think

        try:
            response = requests.post(
                f"{self.base_url}/api/chat",
                json=body,
                timeout=OLLAMA_CHAT_TIMEOUT_SECONDS,
            )
        except (requests.ConnectionError, requests.Timeout) as exc:
            raise OllamaUnavailableError(
                f"Could not reach Ollama at {self.base_url}. "
                "Is it running? Check OLLAMA_BASE_URL / GENERATOR__BASE_URL."
            ) from exc

        response.raise_for_status()
        body = response.json()
        raw_content = body["message"]["content"]
        try:
            data = json.loads(raw_content)
        except json.JSONDecodeError:
            data = None
        if not isinstance(data, dict):
            raise StructuredOutputError(f"Ollama's response wasn't a JSON object: {raw_content!r}")
        return StructuredJSONResult(
            data=data,
            input_tokens=body.get("prompt_eval_count"),
            output_tokens=body.get("eval_count"),
            response_model=body.get("model"),
        )


ANTHROPIC_DEFAULT_MAX_TOKENS = 4096


class AnthropicLLMClient:
    """Wraps the Anthropic SDK; takes an API key or an OAuth bearer token (the `ant` CLI login)."""

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        auth_token: str | None = None,
        max_tokens: int = ANTHROPIC_DEFAULT_MAX_TOKENS,
    ):
        if api_key is None and auth_token is None:
            raise ValueError("AnthropicLLMClient needs an api_key or auth_token")
        self.model = model
        self.max_tokens = max_tokens
        self._client = Anthropic(api_key=api_key) if api_key is not None else Anthropic(auth_token=auth_token)

    def generate(self, prompt: str, system: str | None = None) -> str:
        return self.generate_with_usage(prompt, system).text

    def generate_with_usage(self, prompt: str, system: str | None = None) -> GenerationResult:
        # No temperature: several current Claude models reject it with HTTP 400.
        kwargs = {"model": self.model, "max_tokens": self.max_tokens, "messages": [{"role": "user", "content": prompt}]}
        if system is not None:
            # Anthropic requires roughly 1024+ tokens (model-dependent) before a
            # cache_control block actually caches anything; below that it's a
            # silent no-op (cache_creation_input_tokens stays 0), not an error.
            # This project's static system prompts are currently all smaller
            # than that, so this is dormant until one of them grows past it.
            kwargs["system"] = [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]
        try:
            response = self._client.messages.create(**kwargs)
        except (anthropic.AuthenticationError, anthropic.PermissionDeniedError) as exc:
            raise LLMUnavailableError(
                "Anthropic rejected the credentials. Check ANTHROPIC_API_KEY or run `ant auth login`."
            ) from exc
        except anthropic.RateLimitError as exc:
            raise LLMUnavailableError("Anthropic hit a rate limit. Try again shortly.") from exc
        except anthropic.APIConnectionError as exc:
            raise LLMUnavailableError("Could not reach the Anthropic API. Check the network connection.") from exc
        except anthropic.InternalServerError as exc:
            raise LLMUnavailableError("The Anthropic API is unavailable or overloaded. Try again shortly.") from exc
        usage = getattr(response, "usage", None)
        return GenerationResult(
            text=response.content[0].text,
            input_tokens=getattr(usage, "input_tokens", None),
            output_tokens=getattr(usage, "output_tokens", None),
            response_model=getattr(response, "model", None),
        )


class OpenAILLMClient:
    """Wraps the OpenAI SDK."""

    def __init__(self, model: str, api_key: str, base_url: str | None = None):
        self.model = model
        kwargs = {"api_key": api_key}
        if base_url is not None:
            kwargs["base_url"] = base_url
        self._client = OpenAI(**kwargs)

    def generate(self, prompt: str, system: str | None = None) -> str:
        return self.generate_with_usage(prompt, system).text

    def generate_with_usage(self, prompt: str, system: str | None = None) -> GenerationResult:
        messages = []
        if system is not None:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = self._client.chat.completions.create(model=self.model, messages=messages)
        usage = getattr(response, "usage", None)
        return GenerationResult(
            text=response.choices[0].message.content,
            input_tokens=getattr(usage, "prompt_tokens", None),
            output_tokens=getattr(usage, "completion_tokens", None),
            response_model=getattr(response, "model", None),
        )


def _ant_cli_token() -> str | None:
    """The `ant auth login` OAuth access token, or None; it lasts hours, so callers read it per client."""
    try:
        out = subprocess.run(["ant", "auth", "print-credentials"], capture_output=True, text=True, timeout=20)
        return json.loads(out.stdout).get("access_token") or None
    except (OSError, subprocess.SubprocessError, ValueError, AttributeError, TypeError):
        return None


def resolve_anthropic_credential(config: ProviderConfig) -> dict[str, str]:
    """Config key, then ANTHROPIC_API_KEY, then the `ant` CLI login; the `ant` login only works outside containers."""
    if config.api_key is not None and config.api_key.get_secret_value().strip():
        return {"api_key": config.api_key.get_secret_value()}
    if env_key := os.environ.get("ANTHROPIC_API_KEY", "").strip():
        return {"api_key": env_key}
    if token := _ant_cli_token():
        return {"auth_token": token}
    raise LLMUnavailableError(
        "No Anthropic credential found. Set ANTHROPIC_API_KEY (or GENERATOR__API_KEY / JUDGE__API_KEY), "
        "run `ant auth login` on the host, or switch the provider to ollama."
    )


def get_llm_client(config: ProviderConfig) -> LLMClient:
    """Factory: builds the client matching config.provider."""
    if config.provider == "ollama":
        return OllamaLLMClient(
            model=config.model, base_url=config.base_url or "http://localhost:11434", think=config.think
        )
    if config.provider == "anthropic":
        extra = {"max_tokens": config.max_tokens} if config.max_tokens else {}
        return AnthropicLLMClient(model=config.model, **resolve_anthropic_credential(config), **extra)
    if config.provider == "openai":
        return OpenAILLMClient(
            model=config.model, api_key=config.api_key.get_secret_value(), base_url=config.base_url
        )
    raise ValueError(f"Unknown provider: {config.provider!r}")
