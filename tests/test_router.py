import logging
from unittest.mock import Mock

import pytest
from pydantic import ValidationError

from ask_the_docs.config import ProviderConfig, Settings
from ask_the_docs.llm import GenerationResult, StructuredJSONResult, StructuredOutputError
from ask_the_docs.router import (
    ANALYZE_SCHEMA,
    ANALYZE_SYSTEM_PROMPT,
    GOODBYE_REPLY,
    GREETING_REPLY,
    META_CAPABILITY_SENTENCE,
    NO_DOCUMENTS_SENTENCE,
    THANKS_REPLY,
    Route,
    Turn,
    analyze,
    build_analyze_prompt,
    build_meta_reply,
    match_smalltalk,
    route,
    trim_history,
)
from tests.test_metrics import find_data_point, find_metric, get_metrics_data


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("hi", GREETING_REPLY),
        ("Hi!!", GREETING_REPLY),
        ("hello", GREETING_REPLY),
        ("Hello.", GREETING_REPLY),
        ("hey", GREETING_REPLY),
        ("hey there", GREETING_REPLY),
        ("hi there", GREETING_REPLY),
        ("hello there", GREETING_REPLY),
        ("good morning", GREETING_REPLY),
        ("Good evening!", GREETING_REPLY),
        ("yo", GREETING_REPLY),
        ("hiya", GREETING_REPLY),
        ("\U0001f44b", GREETING_REPLY),
        ("  hi \n", GREETING_REPLY),
        ("thanks", THANKS_REPLY),
        ("Thank you!", THANKS_REPLY),
        ("thank you so much", THANKS_REPLY),
        ("thx", THANKS_REPLY),
        ("ty", THANKS_REPLY),
        ("thanks a lot", THANKS_REPLY),
        ("thanks \U0001f64f", THANKS_REPLY),
        ("\U0001f64f", THANKS_REPLY),
        ("bye", GOODBYE_REPLY),
        ("Goodbye!", GOODBYE_REPLY),
        ("see you", GOODBYE_REPLY),
        ("cya", GOODBYE_REPLY),
    ],
)
def test_match_smalltalk_matches(query, expected):
    assert match_smalltalk(query) == expected


@pytest.mark.parametrize(
    "query",
    [
        "hi, what is LoRA?",
        "hello, explain attention",
        "thanks, and what about RLHF?",
        "hey can you summarize the GPT-3 paper",
        "?",
        "",
        "   ",
        "hi " + "please tell me about transformers " * 3,
        "history of attention",
        "high-rank adaptation",
        "hello world program in python",
        "good morning, what is chain of thought?",
    ],
)
def test_match_smalltalk_rejects(query):
    assert match_smalltalk(query) is None


def _make_turns(n: int) -> list[Turn]:
    return [Turn(role="user" if i % 2 == 0 else "assistant", content=f"turn {i}") for i in range(n)]


def test_trim_history_none_returns_empty():
    assert trim_history(None, 6) == []


def test_trim_history_empty_returns_empty():
    assert trim_history([], 6) == []


def test_trim_history_under_limit_returns_all():
    turns = _make_turns(6)
    assert trim_history(turns, 6) == turns


def test_trim_history_over_limit_returns_last_n_in_order():
    turns = _make_turns(7)
    assert trim_history(turns, 6) == turns[-6:]


def test_trim_history_zero_max_turns_returns_empty():
    turns = _make_turns(3)
    assert trim_history(turns, 0) == []


def test_turn_rejects_empty_and_oversized_content():
    with pytest.raises(ValidationError):
        Turn(role="user", content="")
    with pytest.raises(ValidationError):
        Turn(role="user", content="x" * 4001)


def test_turn_rejects_unknown_role():
    with pytest.raises(ValidationError):
        Turn(role="system", content="hi")


def test_build_meta_reply_empty():
    assert build_meta_reply([]) == f"{META_CAPABILITY_SENTENCE} {NO_DOCUMENTS_SENTENCE}"


def test_build_meta_reply_lists_sources():
    reply = build_meta_reply(["b.pdf", "a.pdf"])
    assert reply.startswith(META_CAPABILITY_SENTENCE)
    assert "2 documents" in reply
    assert reply.splitlines()[1:] == ["- b.pdf", "- a.pdf"]


def test_build_meta_reply_singular_for_one_source():
    reply = build_meta_reply(["a.pdf"])
    assert "1 document" in reply
    assert "1 documents" not in reply


def make_settings(provider="ollama", model="llama3.1:8b", **overrides) -> Settings:
    return Settings(
        generator=ProviderConfig(provider=provider, model=model, api_key=None if provider == "ollama" else "x"),
        judge=ProviderConfig(provider="anthropic", model="claude-sonnet-5-5", api_key="x"),
        **overrides,
    )


class FakeStructuredClient:
    """Simulates OllamaLLMClient.generate_structured, including raising on request."""

    def __init__(self, data=None, error=None):
        self.data = data
        self.error = error
        self.captured_prompt = None
        self.captured_schema = None
        self.captured_system = None

    def generate_structured(self, prompt, schema, system=None):
        self.captured_prompt = prompt
        self.captured_schema = schema
        self.captured_system = system
        if self.error is not None:
            raise self.error
        return StructuredJSONResult(data=self.data, input_tokens=5, output_tokens=7, response_model="fake-model")


class FakeTextClient:
    """Simulates a hosted provider's generate_with_usage, returning free text."""

    def __init__(self, text):
        self.text = text
        self.captured_prompt = None
        self.captured_system = None

    def generate_with_usage(self, prompt, system=None):
        self.captured_prompt = prompt
        self.captured_system = system
        return GenerationResult(text=self.text, input_tokens=5, output_tokens=7, response_model="fake-model")


def _counter_value(name, **attributes):
    try:
        metric = find_metric(get_metrics_data(), name)
        return find_data_point(metric, **attributes).value
    except AssertionError:
        return 0


def _histogram_count(name, **attributes):
    """Reads the running count for a histogram data point; cumulative across the test session, so
    callers compare a before/after delta (mirrors the helper in tests/test_query_analyzer.py)."""
    try:
        metric = find_metric(get_metrics_data(), name)
        return find_data_point(metric, **attributes).count
    except AssertionError:
        return 0


def test_build_analyze_prompt_without_history():
    assert build_analyze_prompt("what is LoRA?", []) == "Latest user message: what is LoRA?"


def test_build_analyze_prompt_with_history():
    history = [
        Turn(role="user", content="what is attention?"),
        Turn(role="assistant", content="Attention weighs tokens [1]."),
    ]
    expected = (
        "Conversation so far:\n"
        "user: what is attention?\n"
        "assistant: Attention weighs tokens [1].\n"
        "\n"
        "Latest user message: explain that more simply"
    )
    assert build_analyze_prompt("explain that more simply", history) == expected


def test_build_analyze_prompt_caps_long_history_turn_content():
    long_content = "x" * 650
    history = [Turn(role="user", content=long_content)]

    prompt = build_analyze_prompt("latest message", history)

    assert (long_content[:600] + "...") in prompt
    assert long_content not in prompt


def test_build_analyze_prompt_leaves_short_history_turn_content_unchanged():
    short_content = "a short turn"
    history = [Turn(role="user", content=short_content)]

    prompt = build_analyze_prompt("latest message", history)

    assert f"user: {short_content}" in prompt
    assert "..." not in prompt


def test_analyze_ollama_uses_structured_output(monkeypatch):
    fake_client = FakeStructuredClient(
        data={"intent": "doc", "standalone_query": "LoRA low-rank adaptation", "reply": ""}
    )
    monkeypatch.setattr("ask_the_docs.router.get_llm_client", lambda _: fake_client)

    result = analyze("what is LoRA?", [], make_settings())

    assert result == Route("doc", "LoRA low-rank adaptation", "", "llm")
    assert fake_client.captured_schema == ANALYZE_SCHEMA
    assert fake_client.captured_system == ANALYZE_SYSTEM_PROMPT


def test_analyze_hosted_provider_parses_fenced_json(monkeypatch):
    text = '```json\n{"intent": "doc", "standalone_query": "LoRA low-rank adaptation", "reply": ""}\n```'
    fake_client = FakeTextClient(text)
    monkeypatch.setattr("ask_the_docs.router.get_llm_client", lambda _: fake_client)

    result = analyze("what is LoRA?", [], make_settings(provider="anthropic", model="claude-sonnet-5-5"))

    assert result == Route("doc", "LoRA low-rank adaptation", "", "llm")
    assert fake_client.captured_system == ANALYZE_SYSTEM_PROMPT


def test_analyze_parses_distinct_search_query(monkeypatch):
    fake_client = FakeStructuredClient(
        data={
            "intent": "doc",
            "standalone_query": "what is LoRA?",
            "search_query": "LoRA low-rank adaptation fine-tuning PEFT",
            "reply": "",
        }
    )
    monkeypatch.setattr("ask_the_docs.router.get_llm_client", lambda _: fake_client)

    result = analyze("what is LoRA?", [], make_settings())

    assert result.search_query == "LoRA low-rank adaptation fine-tuning PEFT"
    assert result.standalone_query == "what is LoRA?"


def test_analyze_hosted_provider_parses_distinct_search_query(monkeypatch):
    text = (
        '```json\n{"intent": "doc", "standalone_query": "what is LoRA?", '
        '"search_query": "LoRA low-rank adaptation fine-tuning PEFT", "reply": ""}\n```'
    )
    fake_client = FakeTextClient(text)
    monkeypatch.setattr("ask_the_docs.router.get_llm_client", lambda _: fake_client)

    result = analyze("what is LoRA?", [], make_settings(provider="anthropic", model="claude-sonnet-5-5"))

    assert result.search_query == "LoRA low-rank adaptation fine-tuning PEFT"


def test_analyze_normalizes_intent_casing(monkeypatch):
    fake_client = FakeStructuredClient(data={"intent": " DOC ", "standalone_query": "q", "reply": ""})
    monkeypatch.setattr("ask_the_docs.router.get_llm_client", lambda _: fake_client)

    result = analyze("q", [], make_settings())

    assert result.intent == "doc"


def test_route_rules_hit_skips_llm(monkeypatch):
    mock_get_client = Mock()
    monkeypatch.setattr("ask_the_docs.router.get_llm_client", mock_get_client)

    result = route("hi", None, make_settings())

    assert result == Route("chitchat", "", GREETING_REPLY, "rules")
    mock_get_client.assert_not_called()


def test_route_disabled_skips_llm(monkeypatch):
    mock_get_client = Mock()
    monkeypatch.setattr("ask_the_docs.router.get_llm_client", mock_get_client)

    result = route("what is LoRA?", None, make_settings(query_rewrite_enabled=False))

    assert result == Route("doc", "what is LoRA?", "", "disabled")
    mock_get_client.assert_not_called()


@pytest.mark.parametrize(
    ("data", "error"),
    [
        (None, RuntimeError("boom")),
        (None, StructuredOutputError("bad json")),
        ({"intent": "weather", "standalone_query": "q", "reply": ""}, None),
        ({"intent": "doc", "standalone_query": 123, "reply": ""}, None),
    ],
)
def test_route_falls_back_to_doc(monkeypatch, caplog, data, error):
    fake_client = FakeStructuredClient(data=data, error=error)
    monkeypatch.setattr("ask_the_docs.router.get_llm_client", lambda _: fake_client)

    with caplog.at_level(logging.WARNING, logger="ask_the_docs.router"):
        result = route("what is LoRA?", None, make_settings())

    assert result == Route("doc", "what is LoRA?", "", "fallback")
    assert any(record.levelname == "WARNING" for record in caplog.records)


def test_route_chitchat_with_empty_reply_uses_canned_reply(monkeypatch):
    fake_client = FakeStructuredClient(data={"intent": "chitchat", "standalone_query": "", "reply": ""})
    monkeypatch.setattr("ask_the_docs.router.get_llm_client", lambda _: fake_client)

    result = route("tell me a joke", None, make_settings())

    assert result == Route("chitchat", "", GREETING_REPLY, "llm")


def test_route_doc_with_empty_standalone_query_uses_raw_query(monkeypatch):
    fake_client = FakeStructuredClient(data={"intent": "doc", "standalone_query": "", "reply": ""})
    monkeypatch.setattr("ask_the_docs.router.get_llm_client", lambda _: fake_client)

    result = route("explain that more simply", None, make_settings())

    assert result == Route("doc", "explain that more simply", "", "llm")


def test_route_trims_history_before_analyze(monkeypatch):
    fake_client = FakeStructuredClient(data={"intent": "doc", "standalone_query": "q", "reply": ""})
    monkeypatch.setattr("ask_the_docs.router.get_llm_client", lambda _: fake_client)
    history = _make_turns(8)

    route("explain that more simply", history, make_settings(history_max_turns=6))

    assert "turn 0" not in fake_client.captured_prompt
    assert "turn 1" not in fake_client.captured_prompt
    assert "turn 2" in fake_client.captured_prompt
    assert "turn 7" in fake_client.captured_prompt


def test_route_records_gen_ai_metrics_on_hosted_provider_malformed_json(monkeypatch):
    """Even when the hosted provider's text isn't parseable JSON, the LLM call still happened and consumed
    tokens -- analyze must record gen_ai.* telemetry before attempting json.loads, not lose it to the parse error."""
    fake_client = FakeTextClient("this is not json")
    monkeypatch.setattr("ask_the_docs.router.get_llm_client", lambda _: fake_client)
    settings = make_settings(provider="anthropic", model="claude-sonnet-5-5")
    attrs = {
        "gen_ai.operation.name": "chat",
        "gen_ai.provider.name": "anthropic",
        "gen_ai.request.model": "claude-sonnet-5-5",
    }

    before = _histogram_count("gen_ai.client.inference.duration", **attrs)
    result = route("what is LoRA?", None, settings)
    after = _histogram_count("gen_ai.client.inference.duration", **attrs)

    assert result == Route("doc", "what is LoRA?", "", "fallback")
    assert after - before == 1


def test_route_increments_route_counter(monkeypatch):
    before = _counter_value("route", intent="chitchat", source="rules")

    route("hi", None, make_settings())

    after = _counter_value("route", intent="chitchat", source="rules")
    assert after - before == 1
