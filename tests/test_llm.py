import json
from types import SimpleNamespace

import pytest
import requests

from ask_the_docs.config import ProviderConfig
from ask_the_docs.llm import (
    AnthropicLLMClient,
    GenerationResult,
    LLMUnavailableError,
    OLLAMA_KEEP_ALIVE,
    OLLAMA_MAX_OUTPUT_TOKENS,
    OllamaLLMClient,
    OllamaUnavailableError,
    OpenAILLMClient,
    StructuredJSONResult,
    StructuredOutputError,
    get_llm_client,
)

STRUCTURED_SCHEMA = {
    "type": "object",
    "properties": {
        "grounded": {"type": "boolean"},
        "answer": {"type": "string"},
        "citations": {"type": "array", "items": {"type": "integer"}},
    },
    "required": ["grounded", "answer", "citations"],
}


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


def test_ollama_client_omits_think_field_by_default(monkeypatch):
    captured = {}

    def fake_post(url, json, timeout):
        captured["json"] = json
        return SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: {"message": {"content": "hello world"}},
        )

    monkeypatch.setattr(requests, "post", fake_post)

    client = OllamaLLMClient(model="llama3.1:8b", base_url="http://localhost:11434")
    client.generate("hi")

    assert "think" not in captured["json"]


def test_ollama_client_sends_think_false_when_configured(monkeypatch):
    captured = {}

    def fake_post(url, json, timeout):
        captured["json"] = json
        return SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: {"message": {"content": "hello world"}},
        )

    monkeypatch.setattr(requests, "post", fake_post)

    client = OllamaLLMClient(model="qwen3:4b", base_url="http://localhost:11434", think=False)
    client.generate("hi")

    assert captured["json"]["think"] is False


def test_get_llm_client_passes_think_to_ollama_client():
    client = get_llm_client(
        ProviderConfig(provider="ollama", model="qwen3:4b", base_url="http://localhost:11434", think=False)
    )
    assert client.think is False


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


def test_ollama_client_sends_keep_alive_to_avoid_idle_model_eviction(monkeypatch):
    """Without this, Ollama unloads the model after its default 5-minute
    idle timeout, so the next request pays a full model-load cost on top
    of generation -- this is the single biggest source of the "it was
    fast a minute ago" variance observed in production traces."""
    captured = {}

    def fake_post(url, json, timeout):
        captured["json"] = json
        return SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: {"message": {"content": "hello world"}},
        )

    monkeypatch.setattr(requests, "post", fake_post)

    client = OllamaLLMClient(model="llama3.1:8b", base_url="http://localhost:11434")
    client.generate("hi")

    assert captured["json"]["keep_alive"] == OLLAMA_KEEP_ALIVE


def test_ollama_client_caps_output_tokens_to_bound_decode_time(monkeypatch):
    """Decode time scales with output length (measured: ~10 tok/s at RAG-sized
    context vs ~26 tok/s on a short prompt), and nothing previously stopped a
    model from generating hundreds of tokens for a question that needed a
    short answer -- this caps the worst-case tail."""
    captured = {}

    def fake_post(url, json, timeout):
        captured["json"] = json
        return SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: {"message": {"content": "hello world"}},
        )

    monkeypatch.setattr(requests, "post", fake_post)

    client = OllamaLLMClient(model="llama3.1:8b", base_url="http://localhost:11434")
    client.generate("hi")

    assert captured["json"]["options"]["num_predict"] == OLLAMA_MAX_OUTPUT_TOKENS


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

    assert captured["system"] == [{"type": "text", "text": "be terse", "cache_control": {"type": "ephemeral"}}]


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


def test_openai_client_forwards_base_url_when_set(monkeypatch):
    captured = {}

    class FakeOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr("ask_the_docs.llm.OpenAI", FakeOpenAI)

    OpenAILLMClient(model="gpt-4o-mini", api_key="x", base_url="https://custom-gateway.example.com/v1")

    assert captured == {"api_key": "x", "base_url": "https://custom-gateway.example.com/v1"}


def test_get_llm_client_forwards_base_url_for_openai(monkeypatch):
    captured = {}

    class FakeOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr("ask_the_docs.llm.OpenAI", FakeOpenAI)

    get_llm_client(
        ProviderConfig(
            provider="openai",
            model="gpt-4o-mini",
            api_key="x",
            base_url="https://custom-gateway.example.com/v1",
        )
    )

    assert captured["base_url"] == "https://custom-gateway.example.com/v1"


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


def test_ollama_generate_structured_sends_format_schema_and_parses_response(monkeypatch):
    captured = {}
    content = json.dumps({"grounded": True, "answer": "the answer [1]", "citations": [1]})

    def fake_post(url, json, timeout):
        captured["url"] = url
        captured["json"] = json
        return SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: {
                "message": {"content": content},
                "prompt_eval_count": 12,
                "eval_count": 34,
                "model": "llama3.1:8b-instruct-q4_K_M",
            },
        )

    monkeypatch.setattr(requests, "post", fake_post)

    client = OllamaLLMClient(model="llama3.1:8b", base_url="http://localhost:11434")
    result = client.generate_structured("prompt", STRUCTURED_SCHEMA, system="be grounded")

    assert captured["url"] == "http://localhost:11434/api/chat"
    assert captured["json"]["format"] == STRUCTURED_SCHEMA
    assert captured["json"]["messages"][0] == {"role": "system", "content": "be grounded"}
    assert captured["json"]["messages"][1] == {"role": "user", "content": "prompt"}
    assert result == StructuredJSONResult(
        data={"grounded": True, "answer": "the answer [1]", "citations": [1]},
        input_tokens=12,
        output_tokens=34,
        response_model="llama3.1:8b-instruct-q4_K_M",
    )


def test_ollama_generate_structured_omits_system_message_when_not_given(monkeypatch):
    captured = {}
    content = json.dumps({"grounded": False, "answer": "", "citations": []})

    def fake_post(url, json, timeout):
        captured["json"] = json
        return SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: {"message": {"content": content}},
        )

    monkeypatch.setattr(requests, "post", fake_post)

    client = OllamaLLMClient(model="llama3.1:8b", base_url="http://localhost:11434")
    client.generate_structured("prompt", STRUCTURED_SCHEMA)

    assert captured["json"]["messages"] == [{"role": "user", "content": "prompt"}]


def test_ollama_generate_structured_raises_on_malformed_json(monkeypatch):
    def fake_post(url, json, timeout):
        return SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: {"message": {"content": "not json at all"}},
        )

    monkeypatch.setattr(requests, "post", fake_post)

    client = OllamaLLMClient(model="llama3.1:8b", base_url="http://localhost:11434")
    with pytest.raises(StructuredOutputError):
        client.generate_structured("prompt", STRUCTURED_SCHEMA)


def test_ollama_generate_structured_raises_when_content_is_not_an_object(monkeypatch):
    def fake_post(url, json, timeout):
        return SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: {"message": {"content": "[1, 2]"}},
        )

    monkeypatch.setattr(requests, "post", fake_post)

    client = OllamaLLMClient(model="llama3.1:8b", base_url="http://localhost:11434")
    with pytest.raises(StructuredOutputError):
        client.generate_structured("prompt", STRUCTURED_SCHEMA)


def test_ollama_generate_structured_returns_arbitrary_schema_objects(monkeypatch):
    content = json.dumps({"intent": "doc", "standalone_query": "q", "reply": ""})

    def fake_post(url, json, timeout):
        return SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: {"message": {"content": content}},
        )

    monkeypatch.setattr(requests, "post", fake_post)

    client = OllamaLLMClient(model="llama3.1:8b", base_url="http://localhost:11434")
    result = client.generate_structured("prompt", {"type": "object"})

    assert result.data == {"intent": "doc", "standalone_query": "q", "reply": ""}


def test_ollama_generate_structured_raises_clear_error_on_connection_failure(monkeypatch):
    def fake_post(url, json, timeout):
        raise requests.ConnectionError("boom")

    monkeypatch.setattr(requests, "post", fake_post)

    client = OllamaLLMClient(model="llama3.1:8b", base_url="http://localhost:11434")
    with pytest.raises(OllamaUnavailableError, match="OLLAMA_BASE_URL"):
        client.generate_structured("prompt", STRUCTURED_SCHEMA)


def test_ollama_generate_structured_sends_keep_alive_to_avoid_idle_model_eviction(monkeypatch):
    """Same reasoning as the generate_with_usage keep_alive test above --
    this is the call path answer.py actually uses for every Ollama
    generation, so a missing keep_alive here is the one that bites."""
    captured = {}
    content = json.dumps({"grounded": True, "answer": "a", "citations": [1]})

    def fake_post(url, json, timeout):
        captured["json"] = json
        return SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: {"message": {"content": content}},
        )

    monkeypatch.setattr(requests, "post", fake_post)

    client = OllamaLLMClient(model="llama3.1:8b", base_url="http://localhost:11434")
    client.generate_structured("prompt", STRUCTURED_SCHEMA)

    assert captured["json"]["keep_alive"] == OLLAMA_KEEP_ALIVE


def test_ollama_generate_structured_caps_output_tokens_to_bound_decode_time(monkeypatch):
    """Same reasoning as the generate_with_usage cap above -- this is the
    call path answer.py actually uses for every Ollama generation."""
    captured = {}
    content = json.dumps({"grounded": True, "answer": "a", "citations": [1]})

    def fake_post(url, json, timeout):
        captured["json"] = json
        return SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: {"message": {"content": content}},
        )

    monkeypatch.setattr(requests, "post", fake_post)

    client = OllamaLLMClient(model="llama3.1:8b", base_url="http://localhost:11434")
    client.generate_structured("prompt", STRUCTURED_SCHEMA)

    assert captured["json"]["options"]["num_predict"] == OLLAMA_MAX_OUTPUT_TOKENS


def test_ollama_generate_structured_defaults_to_none_when_counts_absent(monkeypatch):
    content = json.dumps({"grounded": True, "answer": "a", "citations": [1]})

    def fake_post(url, json, timeout):
        return SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: {"message": {"content": content}},
        )

    monkeypatch.setattr(requests, "post", fake_post)

    client = OllamaLLMClient(model="llama3.1:8b", base_url="http://localhost:11434")
    result = client.generate_structured("prompt", STRUCTURED_SCHEMA)

    assert result.input_tokens is None and result.output_tokens is None
    assert result.response_model is None


def test_strip_code_fence_returns_non_str_input_unchanged():
    from ask_the_docs.llm import strip_code_fence

    assert strip_code_fence(None) is None


def test_strip_code_fence_strips_json_tagged_and_bare_fences():
    from ask_the_docs.llm import strip_code_fence

    assert strip_code_fence('```json\n{"a": 1}\n```') == '{"a": 1}'
    assert strip_code_fence('```\n[1]\n```') == "[1]"


def test_strip_code_fence_leaves_unfenced_text_unchanged():
    from ask_the_docs.llm import strip_code_fence

    assert strip_code_fence('{"a": 1}') == '{"a": 1}'


def test_ollama_unavailable_is_an_llm_unavailable_error():
    assert issubclass(OllamaUnavailableError, LLMUnavailableError)
