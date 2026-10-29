from types import SimpleNamespace

import pytest
import requests

from ask_the_docs.config import ProviderConfig
from ask_the_docs.llm import (
    AnthropicLLMClient,
    GenerationResult,
    OllamaLLMClient,
    OllamaUnavailableError,
    OpenAILLMClient,
    get_llm_client,
)


def test_get_llm_client_returns_ollama_client():
    client = get_llm_client(
        ProviderConfig(provider="ollama", model="llama3.1:8b", base_url="http://localhost:11434")
    )
    assert isinstance(client, OllamaLLMClient)


def test_get_llm_client_returns_anthropic_client():
    client = get_llm_client(
        ProviderConfig(provider="anthropic", model="claude-sonnet-5-5", api_key="x")
    )
    assert isinstance(client, AnthropicLLMClient)


def test_get_llm_client_returns_openai_client():
    client = get_llm_client(ProviderConfig(provider="openai", model="gpt-4o-mini", api_key="x"))
    assert isinstance(client, OpenAILLMClient)


def test_ollama_client_generate_posts_to_chat_endpoint(monkeypatch):
    captured = {}

    def fake_post(url, json, timeout):
        captured["url"] = url
        captured["json"] = json
        return SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: {"message": {"content": "hello world"}},
        )

    monkeypatch.setattr(requests, "post", fake_post)

    client = OllamaLLMClient(model="llama3.1:8b", base_url="http://localhost:11434")
    result = client.generate("hi")

    assert result == "hello world"
    assert captured["url"] == "http://localhost:11434/api/chat"
    assert captured["json"]["model"] == "llama3.1:8b"
    assert captured["json"]["messages"] == [{"role": "user", "content": "hi"}]


def test_ollama_client_generate_raises_clear_error_on_connection_failure(monkeypatch):
    def fake_post(url, json, timeout):
        raise requests.ConnectionError("boom")

    monkeypatch.setattr(requests, "post", fake_post)

    client = OllamaLLMClient(model="llama3.1:8b", base_url="http://localhost:11434")
    with pytest.raises(OllamaUnavailableError, match="OLLAMA_BASE_URL"):
        client.generate("hi")


def test_ollama_client_generate_prepends_system_message_when_given(monkeypatch):
    captured = {}

    def fake_post(url, json, timeout):
        captured["json"] = json
        return SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: {"message": {"content": "hello world"}},
        )

    monkeypatch.setattr(requests, "post", fake_post)

    client = OllamaLLMClient(model="llama3.1:8b", base_url="http://localhost:11434")
    client.generate("hi", system="be terse")

    assert captured["json"]["messages"][0] == {"role": "system", "content": "be terse"}
    assert captured["json"]["messages"][1] == {"role": "user", "content": "hi"}


def test_anthropic_client_generate_calls_sdk(monkeypatch):
    captured = {}

    class FakeMessages:
        def create(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(content=[SimpleNamespace(text="hello from claude")])

    class FakeAnthropic:
        def __init__(self, api_key):
            self.messages = FakeMessages()

    monkeypatch.setattr("ask_the_docs.llm.Anthropic", FakeAnthropic)

    client = AnthropicLLMClient(model="claude-sonnet-5-5", api_key="x")
    assert client.generate("hi") == "hello from claude"
    assert "system" not in captured


def test_anthropic_client_generate_passes_system_kwarg_when_given(monkeypatch):
    captured = {}

    class FakeMessages:
        def create(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(content=[SimpleNamespace(text="hello from claude")])

    class FakeAnthropic:
        def __init__(self, api_key):
            self.messages = FakeMessages()

    monkeypatch.setattr("ask_the_docs.llm.Anthropic", FakeAnthropic)

    client = AnthropicLLMClient(model="claude-sonnet-5-5", api_key="x")
    client.generate("hi", system="be terse")

    assert captured["system"] == "be terse"


def test_openai_client_generate_calls_sdk(monkeypatch):
    captured = {}

    class FakeCompletions:
        def create(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="hello from gpt"))]
            )

    class FakeChat:
        def __init__(self):
            self.completions = FakeCompletions()

    class FakeOpenAI:
        def __init__(self, api_key):
            self.chat = FakeChat()

    monkeypatch.setattr("ask_the_docs.llm.OpenAI", FakeOpenAI)

    client = OpenAILLMClient(model="gpt-4o-mini", api_key="x")
    assert client.generate("hi") == "hello from gpt"
    assert captured["messages"] == [{"role": "user", "content": "hi"}]


def test_ollama_generate_with_usage_reads_token_counts(monkeypatch):
    def fake_post(url, json, timeout):
        return SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: {
                "message": {"content": "answer"},
                "prompt_eval_count": 12,
                "eval_count": 34,
                "model": "llama3.1:8b-instruct-q4_K_M",
            },
        )

    monkeypatch.setattr(requests, "post", fake_post)

    client = OllamaLLMClient(model="llama3.1:8b", base_url="http://localhost:11434")
    result = client.generate_with_usage("prompt")

    assert result == GenerationResult(
        text="answer",
        input_tokens=12,
        output_tokens=34,
        response_model="llama3.1:8b-instruct-q4_K_M",
    )


def test_ollama_generate_with_usage_defaults_to_none_when_counts_absent(monkeypatch):
    def fake_post(url, json, timeout):
        return SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: {"message": {"content": "answer"}},
        )

    monkeypatch.setattr(requests, "post", fake_post)

    client = OllamaLLMClient(model="llama3.1:8b", base_url="http://localhost:11434")
    result = client.generate_with_usage("prompt")

    assert result.input_tokens is None and result.output_tokens is None
    assert result.response_model is None


def test_anthropic_generate_with_usage_reads_token_counts(monkeypatch):
    class FakeMessages:
        def create(self, **kwargs):
            return SimpleNamespace(
                content=[SimpleNamespace(text="hello from claude")],
                usage=SimpleNamespace(input_tokens=10, output_tokens=20),
                model="claude-sonnet-5-5-20260101",
            )

    class FakeAnthropic:
        def __init__(self, api_key):
            self.messages = FakeMessages()

    monkeypatch.setattr("ask_the_docs.llm.Anthropic", FakeAnthropic)

    client = AnthropicLLMClient(model="claude-sonnet-5-5", api_key="x")
    result = client.generate_with_usage("hi")

    assert result == GenerationResult(
        text="hello from claude",
        input_tokens=10,
        output_tokens=20,
        response_model="claude-sonnet-5-5-20260101",
    )


def test_anthropic_generate_with_usage_missing_usage_field_returns_none(monkeypatch):
    class FakeMessages:
        def create(self, **kwargs):
            return SimpleNamespace(content=[SimpleNamespace(text="hello from claude")])

    class FakeAnthropic:
        def __init__(self, api_key):
            self.messages = FakeMessages()

    monkeypatch.setattr("ask_the_docs.llm.Anthropic", FakeAnthropic)

    client = AnthropicLLMClient(model="claude-sonnet-5-5", api_key="x")
    result = client.generate_with_usage("hi")

    assert result.input_tokens is None and result.output_tokens is None
    assert result.response_model is None


def test_openai_generate_with_usage_reads_token_counts(monkeypatch):
    class FakeCompletions:
        def create(self, **kwargs):
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="hello from gpt"))],
                usage=SimpleNamespace(prompt_tokens=5, completion_tokens=7),
                model="gpt-4o-mini-2024-07-18",
            )

    class FakeChat:
        def __init__(self):
            self.completions = FakeCompletions()

    class FakeOpenAI:
        def __init__(self, api_key):
            self.chat = FakeChat()

    monkeypatch.setattr("ask_the_docs.llm.OpenAI", FakeOpenAI)

    client = OpenAILLMClient(model="gpt-4o-mini", api_key="x")
    result = client.generate_with_usage("hi")

    assert result == GenerationResult(
        text="hello from gpt",
        input_tokens=5,
        output_tokens=7,
        response_model="gpt-4o-mini-2024-07-18",
    )


def test_openai_generate_with_usage_missing_usage_field_returns_none(monkeypatch):
    class FakeCompletions:
        def create(self, **kwargs):
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="hello from gpt"))]
            )

    class FakeChat:
        def __init__(self):
            self.completions = FakeCompletions()

    class FakeOpenAI:
        def __init__(self, api_key):
            self.chat = FakeChat()

    monkeypatch.setattr("ask_the_docs.llm.OpenAI", FakeOpenAI)

    client = OpenAILLMClient(model="gpt-4o-mini", api_key="x")
    result = client.generate_with_usage("hi")

    assert result.input_tokens is None and result.output_tokens is None
    assert result.response_model is None


def test_openai_client_generate_prepends_system_message_when_given(monkeypatch):
    captured = {}

    class FakeCompletions:
        def create(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="hello from gpt"))]
            )

    class FakeChat:
        def __init__(self):
            self.completions = FakeCompletions()

    class FakeOpenAI:
        def __init__(self, api_key):
            self.chat = FakeChat()

    monkeypatch.setattr("ask_the_docs.llm.OpenAI", FakeOpenAI)

    client = OpenAILLMClient(model="gpt-4o-mini", api_key="x")
    client.generate("hi", system="be terse")

    assert captured["messages"][0] == {"role": "system", "content": "be terse"}
    assert captured["messages"][1] == {"role": "user", "content": "hi"}
