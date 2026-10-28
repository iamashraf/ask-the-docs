import dataclasses
import json
import logging
from unittest.mock import Mock

import pytest
from langchain_core.documents import Document

from ask_the_docs.answer import (
    MAX_GENERATION_ATTEMPTS,
    AnswerResult,
    answer,
    build_prompt,
    citation_guardrail_passes,
    format_context,
)
from ask_the_docs.cache import make_cache_key
from ask_the_docs.config import REFUSAL_MESSAGE, ProviderConfig, Settings


def make_settings() -> Settings:
    return Settings(
        generator=ProviderConfig(provider="ollama", model="llama3.1:8b"),
        judge=ProviderConfig(provider="anthropic", model="claude-sonnet-5-5", api_key="x"),
    )


def test_format_context_includes_source_citation_markers():
    chunks = [
        Document(page_content="chunk one", metadata={"source": "a.pdf"}),
        Document(page_content="chunk two", metadata={"source": "b.pdf"}),
    ]

    result = format_context(chunks)

    assert "[1]" in result and "a.pdf" in result
    assert "[2]" in result and "b.pdf" in result
    assert "chunk one" in result
    assert "chunk two" in result


def test_build_prompt_includes_query_and_context():
    chunks = [Document(page_content="chunk one", metadata={"source": "a.pdf"})]

    prompt = build_prompt("What is X?", format_context(chunks))

    assert "What is X?" in prompt
    assert "chunk one" in prompt


class FakeCacheClient:
    def __init__(self):
        self._store: dict[str, bytes] = {}

    def get(self, key):
        return self._store.get(key)

    def set(self, key, value, ex=None):
        self._store[key] = value


@pytest.fixture(autouse=True)
def _no_real_cache_or_index(monkeypatch):
    """answer() always computes a cache key (needs get_chunk_count) and
    checks a cache client before doing anything else -- without this,
    every test in this file would hit a real Chroma dir and a real Valkey
    connection. A fresh, empty FakeCacheClient per test means every call
    here is a cache miss by default unless a test wires up its own."""
    monkeypatch.setattr("ask_the_docs.answer.get_chunk_count", lambda settings: 0)
    monkeypatch.setattr("ask_the_docs.answer.get_cache_client", lambda settings: FakeCacheClient())


def test_citation_guardrail_fails_on_zero_citations():
    assert citation_guardrail_passes("just an unsupported claim", num_chunks=3) is False


def test_citation_guardrail_fails_on_out_of_range_citation():
    assert citation_guardrail_passes("per [6], ...", num_chunks=5) is False


def test_citation_guardrail_passes_on_valid_citation():
    assert citation_guardrail_passes("per [2], ...", num_chunks=5) is True


def test_citation_guardrail_fails_gracefully_on_absurdly_large_citation_number():
    # Python's int() refuses strings over 4300 digits (its own DoS guard) --
    # the guardrail must treat that as "invalid citation", not crash.
    huge_citation = f"per [{'9' * 5000}], ..."
    assert citation_guardrail_passes(huge_citation, num_chunks=5) is False


def test_answer_returns_refusal_message_without_calling_llm(monkeypatch):
    monkeypatch.setattr("ask_the_docs.answer.retrieve", lambda query, settings, k: [])
    mock_get_llm_client = Mock()
    monkeypatch.setattr("ask_the_docs.answer.get_llm_client", mock_get_llm_client)

    result = answer("anything", make_settings())

    assert result == AnswerResult(text=REFUSAL_MESSAGE, citations=[])
    mock_get_llm_client.assert_not_called()


def test_answer_calls_generator_with_context_and_returns_citations(monkeypatch):
    chunks = [
        Document(page_content="chunk one", metadata={"source": "a.pdf"}),
        Document(page_content="chunk two", metadata={"source": "b.pdf"}),
    ]
    monkeypatch.setattr("ask_the_docs.answer.retrieve", lambda query, settings, k: chunks)

    captured_prompts = []

    class FakeClient:
        def generate(self, prompt, system=None):
            captured_prompts.append(prompt)
            return "the answer [1][2]"

    monkeypatch.setattr("ask_the_docs.answer.get_llm_client", lambda config: FakeClient())

    result = answer("What is X?", make_settings())

    assert result.text == "the answer [1][2]"
    assert result.citations == ["a.pdf", "b.pdf"]
    assert "chunk one" in result.context
    assert "What is X?" in captured_prompts[0]
    assert "chunk one" in captured_prompts[0]


def test_answer_dedupes_citations_preserving_order(monkeypatch):
    chunks = [
        Document(page_content="chunk one", metadata={"source": "a.pdf"}),
        Document(page_content="chunk two", metadata={"source": "b.pdf"}),
        Document(page_content="chunk three", metadata={"source": "a.pdf"}),
    ]
    monkeypatch.setattr("ask_the_docs.answer.retrieve", lambda query, settings, k: chunks)
    monkeypatch.setattr(
        "ask_the_docs.answer.get_llm_client",
        lambda config: Mock(generate=Mock(return_value="the answer [1][2][3]")),
    )

    result = answer("What is X?", make_settings())

    assert result.citations == ["a.pdf", "b.pdf"]


def test_answer_returns_refusal_message_when_guardrail_fails(monkeypatch):
    chunks = [Document(page_content="chunk one", metadata={"source": "a.pdf"})]
    monkeypatch.setattr("ask_the_docs.answer.retrieve", lambda query, settings, k: chunks)
    fake_generate = Mock(return_value="an answer with no citation markers")
    monkeypatch.setattr(
        "ask_the_docs.answer.get_llm_client",
        lambda config: Mock(generate=fake_generate),
    )

    result = answer("some question", make_settings())

    assert result == AnswerResult(text=REFUSAL_MESSAGE, citations=[])
    # Every attempt failed the guardrail, so the retry loop must have
    # exhausted MAX_GENERATION_ATTEMPTS before falling back to refusal --
    # not just tried once.
    assert fake_generate.call_count == MAX_GENERATION_ATTEMPTS


def test_answer_retries_once_when_first_attempt_fails_the_guardrail(monkeypatch):
    """A local LLM's citation formatting is non-deterministic -- a single
    malformed/uncited response must not immediately refuse a genuinely
    answerable question; one retry should recover it."""
    chunks = [Document(page_content="chunk one", metadata={"source": "a.pdf"})]
    monkeypatch.setattr("ask_the_docs.answer.retrieve", lambda query, settings, k: chunks)
    fake_generate = Mock(
        side_effect=["an answer with no citation markers", "a grounded answer [1]"]
    )
    monkeypatch.setattr(
        "ask_the_docs.answer.get_llm_client",
        lambda config: Mock(generate=fake_generate),
    )

    result = answer("some question", make_settings())

    assert result.text == "a grounded answer [1]"
    assert result.citations == ["a.pdf"]
    assert fake_generate.call_count == 2


def test_answer_logs_prompt_at_debug_level(monkeypatch, caplog):
    chunks = [Document(page_content="chunk one", metadata={"source": "a.pdf"})]
    monkeypatch.setattr("ask_the_docs.answer.retrieve", lambda query, settings, k: chunks)
    monkeypatch.setattr(
        "ask_the_docs.answer.get_llm_client",
        lambda config: Mock(generate=Mock(return_value="the answer [1]")),
    )

    with caplog.at_level(logging.DEBUG, logger="ask_the_docs.answer"):
        answer("What is X?", make_settings())

    assert any("What is X?" in record.message for record in caplog.records)


def test_answer_returns_cached_result_without_calling_generator(monkeypatch):
    monkeypatch.setattr("ask_the_docs.answer.get_chunk_count", lambda settings: 1)
    fake_cache = FakeCacheClient()
    monkeypatch.setattr("ask_the_docs.answer.get_cache_client", lambda settings: fake_cache)
    settings = make_settings()

    cached_result = AnswerResult(text="cached answer [1]", citations=["a.pdf"], context="ctx")
    key = make_cache_key(
        "what is x?",
        5,
        settings.embeddings_backend,
        settings.generator.provider,
        settings.generator.model,
        settings.hybrid_alpha,
        settings.rerank_enabled,
        settings.query_rewrite_enabled,
        settings.system_prompt_template,
        settings.candidate_pool_size,
        settings.graph_enabled,
        settings.graph_traversal_hop_depth,
        settings.global_search_top_communities,
        1,
    )
    fake_cache.set(key, json.dumps(dataclasses.asdict(cached_result)).encode())

    mock_retrieve = Mock()
    monkeypatch.setattr("ask_the_docs.answer.retrieve", mock_retrieve)
    mock_get_llm_client = Mock()
    monkeypatch.setattr("ask_the_docs.answer.get_llm_client", mock_get_llm_client)

    result = answer("What is X?", settings, k=5)

    assert result == cached_result
    mock_retrieve.assert_not_called()
    mock_get_llm_client.assert_not_called()


def test_answer_recomputes_when_cached_value_is_corrupted(monkeypatch):
    chunks = [Document(page_content="chunk one", metadata={"source": "a.pdf"})]
    monkeypatch.setattr("ask_the_docs.answer.retrieve", lambda query, settings, k: chunks)
    monkeypatch.setattr(
        "ask_the_docs.answer.get_llm_client",
        lambda config: Mock(generate=Mock(return_value="the answer [1]")),
    )

    fake_cache = FakeCacheClient()
    # Prime the cache with a value that would crash a naive json.loads/AnswerResult(**...).
    key = make_cache_key(
        "what is x?",
        5,
        make_settings().embeddings_backend,
        make_settings().generator.provider,
        make_settings().generator.model,
        make_settings().hybrid_alpha,
        make_settings().rerank_enabled,
        make_settings().query_rewrite_enabled,
        make_settings().system_prompt_template,
        make_settings().candidate_pool_size,
        make_settings().graph_enabled,
        make_settings().graph_traversal_hop_depth,
        make_settings().global_search_top_communities,
        0,
    )
    fake_cache.set(key, b"not valid json")
    monkeypatch.setattr("ask_the_docs.answer.get_cache_client", lambda settings: fake_cache)

    result = answer("What is X?", make_settings(), k=5)

    assert result.text == "the answer [1]"


def test_answer_calls_generator_once_for_repeated_identical_calls(monkeypatch):
    chunks = [Document(page_content="chunk one", metadata={"source": "a.pdf"})]
    monkeypatch.setattr("ask_the_docs.answer.retrieve", lambda query, settings, k: chunks)
    monkeypatch.setattr("ask_the_docs.answer.get_chunk_count", lambda settings: 1)
    shared_cache = FakeCacheClient()
    monkeypatch.setattr("ask_the_docs.answer.get_cache_client", lambda settings: shared_cache)

    calls = {"count": 0}

    class FakeClient:
        def generate(self, prompt, system=None):
            calls["count"] += 1
            return "the answer [1]"

    monkeypatch.setattr("ask_the_docs.answer.get_llm_client", lambda config: FakeClient())
    settings = make_settings()

    answer("What is X?", settings, k=5)
    answer("What is X?", settings, k=5)

    assert calls["count"] == 1


def test_answer_cache_key_changes_when_system_prompt_template_changes(monkeypatch):
    chunks = [Document(page_content="chunk one", metadata={"source": "a.pdf"})]
    monkeypatch.setattr("ask_the_docs.answer.retrieve", lambda query, settings, k: chunks)
    monkeypatch.setattr("ask_the_docs.answer.get_chunk_count", lambda settings: 1)
    shared_cache = FakeCacheClient()
    monkeypatch.setattr("ask_the_docs.answer.get_cache_client", lambda settings: shared_cache)

    calls = {"count": 0}

    class FakeClient:
        def generate(self, prompt, system=None):
            calls["count"] += 1
            return "the answer [1]"

    monkeypatch.setattr("ask_the_docs.answer.get_llm_client", lambda config: FakeClient())

    settings_a = make_settings().model_copy(update={"system_prompt_template": "be terse"})
    settings_b = make_settings().model_copy(update={"system_prompt_template": "be verbose"})

    answer("What is X?", settings_a, k=5)
    answer("What is X?", settings_b, k=5)

    assert calls["count"] == 2


def test_answer_cache_key_changes_when_candidate_pool_size_changes(monkeypatch):
    chunks = [Document(page_content="chunk one", metadata={"source": "a.pdf"})]
    monkeypatch.setattr("ask_the_docs.answer.retrieve", lambda query, settings, k: chunks)
    monkeypatch.setattr("ask_the_docs.answer.get_chunk_count", lambda settings: 1)
    shared_cache = FakeCacheClient()
    monkeypatch.setattr("ask_the_docs.answer.get_cache_client", lambda settings: shared_cache)

    calls = {"count": 0}

    class FakeClient:
        def generate(self, prompt, system=None):
            calls["count"] += 1
            return "the answer [1]"

    monkeypatch.setattr("ask_the_docs.answer.get_llm_client", lambda config: FakeClient())

    settings_a = make_settings().model_copy(update={"candidate_pool_size": 20})
    settings_b = make_settings().model_copy(update={"candidate_pool_size": 50})

    answer("What is X?", settings_a, k=5)
    answer("What is X?", settings_b, k=5)

    assert calls["count"] == 2


@pytest.mark.parametrize(
    ("field", "value_a", "value_b"),
    [
        ("graph_enabled", False, True),
        ("graph_traversal_hop_depth", 1, 2),
        ("global_search_top_communities", 3, 5),
    ],
)
def test_answer_cache_key_changes_when_graph_setting_changes(monkeypatch, field, value_a, value_b):
    chunks = [Document(page_content="chunk one", metadata={"source": "a.pdf"})]
    monkeypatch.setattr("ask_the_docs.answer.retrieve", lambda query, settings, k: chunks)
    monkeypatch.setattr("ask_the_docs.answer.get_chunk_count", lambda settings: 1)
    shared_cache = FakeCacheClient()
    monkeypatch.setattr("ask_the_docs.answer.get_cache_client", lambda settings: shared_cache)

    calls = {"count": 0}

    class FakeClient:
        def generate(self, prompt, system=None):
            calls["count"] += 1
            return "the answer [1]"

    monkeypatch.setattr("ask_the_docs.answer.get_llm_client", lambda config: FakeClient())

    settings_a = make_settings().model_copy(update={field: value_a})
    settings_b = make_settings().model_copy(update={field: value_b})

    answer("What is X?", settings_a, k=5)
    answer("What is X?", settings_b, k=5)

    assert calls["count"] == 2


def test_answer_cache_key_changes_when_chunk_count_changes(monkeypatch):
    chunks = [Document(page_content="chunk one", metadata={"source": "a.pdf"})]
    monkeypatch.setattr("ask_the_docs.answer.retrieve", lambda query, settings, k: chunks)

    counts = [2, 3]
    monkeypatch.setattr("ask_the_docs.answer.get_chunk_count", lambda settings: counts.pop(0))
    shared_cache = FakeCacheClient()
    monkeypatch.setattr("ask_the_docs.answer.get_cache_client", lambda settings: shared_cache)

    calls = {"count": 0}

    class FakeClient:
        def generate(self, prompt, system=None):
            calls["count"] += 1
            return "the answer [1]"

    monkeypatch.setattr("ask_the_docs.answer.get_llm_client", lambda config: FakeClient())
    settings = make_settings()

    answer("What is X?", settings, k=5)
    answer("What is X?", settings, k=5)

    assert calls["count"] == 2


def test_answer_passes_system_prompt_to_generate(monkeypatch):
    chunks = [Document(page_content="chunk one", metadata={"source": "a.pdf"})]
    monkeypatch.setattr("ask_the_docs.answer.retrieve", lambda query, settings, k: chunks)

    captured = {}

    class FakeClient:
        def generate(self, prompt, system=None):
            captured["system"] = system
            return "the answer [1]"

    monkeypatch.setattr("ask_the_docs.answer.get_llm_client", lambda config: FakeClient())
    settings = make_settings()

    answer("q", settings)

    assert captured["system"] == settings.system_prompt_template
