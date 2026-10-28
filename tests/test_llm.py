from types import SimpleNamespace

import pytest
import requests

from ask_the_docs.config import ProviderConfig
from ask_the_docs.llm import (
    AnthropicLLMClient,
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
