import dataclasses
import json
import logging
from unittest.mock import Mock

import pytest
from langchain_core.documents import Document

from ask_the_docs import metrics
from ask_the_docs.answer import (
    MAX_GENERATION_ATTEMPTS,
    AnswerResult,
    _compute_answer,
    answer,
    build_prompt,
    citation_guardrail_passes,
    format_context,
)
from ask_the_docs.cache import make_cache_key
from ask_the_docs.config import REFUSAL_MESSAGE, ProviderConfig, Settings
from ask_the_docs.llm import GenerationResult
from tests.test_metrics import find_data_point, find_metric, get_metrics_data


def make_settings() -> Settings:
    return Settings(
        generator=ProviderConfig(provider="ollama", model="llama3.1:8b"),
        judge=ProviderConfig(provider="anthropic", model="claude-sonnet-5-5", api_key="x"),
    )


def _find_point_or_none(name, **attributes):
    """Looks up a recorded data point, tolerating one that doesn't exist yet."""
    try:
        metric = find_metric(get_metrics_data(), name)
    except AssertionError:
        return None
    try:
        return find_data_point(metric, **attributes)
    except AssertionError:
        return None


def _counter_value(name, **attributes):
    point = _find_point_or_none(name, **attributes)
    return point.value if point is not None else 0


def _histogram_count(name, **attributes):
    point = _find_point_or_none(name, **attributes)
    return point.count if point is not None else 0


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
    """answer() always computes a cache key (needs get_index_generation)
    and checks a cache client before doing anything else -- without this,
    every test in this file would hit a real Chroma dir and a real Valkey
    connection. A fresh, empty FakeCacheClient per test means every call
    here is a cache miss by default unless a test wires up its own."""
    monkeypatch.setattr("ask_the_docs.answer.get_index_generation", lambda settings: 0)
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


def test_answer_calls_generator_with_context_and_returns_citations(monkeypatch, caplog):
    chunks = [
        Document(page_content="chunk one", metadata={"source": "a.pdf"}),
        Document(page_content="chunk two", metadata={"source": "b.pdf"}),
    ]
    monkeypatch.setattr("ask_the_docs.answer.retrieve", lambda query, settings, k: chunks)

    captured_prompts = []

    class FakeClient:
        def generate_with_usage(self, prompt, system=None):
            captured_prompts.append(prompt)
            return GenerationResult(text="the answer [1][2]", input_tokens=None, output_tokens=None)

    monkeypatch.setattr("ask_the_docs.answer.get_llm_client", lambda config: FakeClient())

    caplog.set_level(logging.INFO, logger="ask_the_docs.answer")

    result = answer("What is X?", make_settings())

    assert result.text == "the answer [1][2]"
    assert result.citations == ["a.pdf", "b.pdf"]
    assert "chunk one" in result.context
    assert "What is X?" in captured_prompts[0]
    assert "chunk one" in captured_prompts[0]
    assert any("Answer generated" in record.message for record in caplog.records)


def test_answer_dedupes_citations_preserving_order(monkeypatch):
    chunks = [
        Document(page_content="chunk one", metadata={"source": "a.pdf"}),
        Document(page_content="chunk two", metadata={"source": "b.pdf"}),
        Document(page_content="chunk three", metadata={"source": "a.pdf"}),
    ]
    monkeypatch.setattr("ask_the_docs.answer.retrieve", lambda query, settings, k: chunks)
    monkeypatch.setattr(
        "ask_the_docs.answer.get_llm_client",
        lambda config: Mock(
            generate_with_usage=Mock(
                return_value=GenerationResult(
                    text="the answer [1][2][3]", input_tokens=None, output_tokens=None
                )
            )
        ),
    )

    result = answer("What is X?", make_settings())

    assert result.citations == ["a.pdf", "b.pdf"]


def test_answer_returns_refusal_message_when_guardrail_fails(monkeypatch):
    chunks = [Document(page_content="chunk one", metadata={"source": "a.pdf"})]
    monkeypatch.setattr("ask_the_docs.answer.retrieve", lambda query, settings, k: chunks)
    fake_generate = Mock(
        return_value=GenerationResult(
            text="an answer with no citation markers", input_tokens=None, output_tokens=None
        )
    )
    monkeypatch.setattr(
        "ask_the_docs.answer.get_llm_client",
        lambda config: Mock(generate_with_usage=fake_generate),
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
        side_effect=[
            GenerationResult(
                text="an answer with no citation markers", input_tokens=None, output_tokens=None
            ),
            GenerationResult(text="a grounded answer [1]", input_tokens=None, output_tokens=None),
        ]
    )
    monkeypatch.setattr(
        "ask_the_docs.answer.get_llm_client",
        lambda config: Mock(generate_with_usage=fake_generate),
    )

    result = answer("some question", make_settings())

    assert result.text == "a grounded answer [1]"
    assert result.citations == ["a.pdf"]
    assert fake_generate.call_count == 2


def test_compute_answer_records_the_guardrail_result_on_the_generate_span(monkeypatch):
    class FakeSpan:
        def __init__(self):
            self.attrs = {}

        def set_attribute(self, k, v):
            self.attrs[k] = v

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    class FakeTracer:
        def __init__(self):
            self.spans: list[FakeSpan] = []

        def start_as_current_span(self, name, **kwargs):
            span = FakeSpan()
            self.spans.append(span)
            return span

    fake_tracer = FakeTracer()
    monkeypatch.setattr("ask_the_docs.answer.tracer", fake_tracer)
    chunks = [Document(page_content="chunk one", metadata={"source": "a.pdf"})]
    monkeypatch.setattr("ask_the_docs.answer.retrieve", lambda query, settings, k: chunks)
    monkeypatch.setattr(
        "ask_the_docs.answer.get_llm_client",
        lambda config: Mock(
            generate_with_usage=Mock(
                return_value=GenerationResult(text="the answer [1]", input_tokens=None, output_tokens=None)
            )
        ),
    )

    _compute_answer("What is X?", make_settings(), k=5)

    assert fake_tracer.spans[0].attrs["citation_guardrail_passed"] is True


def test_compute_answer_sets_gen_ai_span_attributes(monkeypatch):
    class FakeSpan:
        def __init__(self):
            self.attrs = {}

        def set_attribute(self, k, v):
            self.attrs[k] = v

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    class FakeTracer:
        def __init__(self):
            self.spans: list[FakeSpan] = []

        def start_as_current_span(self, name, **kwargs):
            span = FakeSpan()
            self.spans.append(span)
            return span

    fake_tracer = FakeTracer()
    monkeypatch.setattr("ask_the_docs.answer.tracer", fake_tracer)
    chunks = [Document(page_content="chunk one", metadata={"source": "a.pdf"})]
    monkeypatch.setattr("ask_the_docs.answer.retrieve", lambda query, settings, k: chunks)
    monkeypatch.setattr(
        "ask_the_docs.answer.get_llm_client",
        lambda config: Mock(
            generate_with_usage=Mock(
                return_value=GenerationResult(
                    text="the answer [1]",
                    input_tokens=12,
                    output_tokens=34,
                    response_model="llama3.1:8b-instruct-q4_K_M",
                )
            )
        ),
    )
    settings = make_settings()

    _compute_answer("What is X?", settings, k=5)

    attrs = fake_tracer.spans[0].attrs
    assert attrs["gen_ai.request.model"] == settings.generator.model
    assert attrs["gen_ai.response.model"] == "llama3.1:8b-instruct-q4_K_M"
    assert attrs["gen_ai.provider.name"] == settings.generator.provider
    assert attrs["gen_ai.usage.input_tokens"] == 12
    assert attrs["gen_ai.usage.output_tokens"] == 34


def test_compute_answer_omits_usage_attributes_when_token_counts_are_none(monkeypatch):
    """Ollama omits prompt_eval_count/eval_count when a prompt is served
    from its KV cache -- must not crash or fabricate a count."""

    class FakeSpan:
        def __init__(self):
            self.attrs = {}

        def set_attribute(self, k, v):
            self.attrs[k] = v

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    class FakeTracer:
        def __init__(self):
            self.spans: list[FakeSpan] = []

        def start_as_current_span(self, name, **kwargs):
            span = FakeSpan()
            self.spans.append(span)
            return span

    fake_tracer = FakeTracer()
    monkeypatch.setattr("ask_the_docs.answer.tracer", fake_tracer)
    chunks = [Document(page_content="chunk one", metadata={"source": "a.pdf"})]
    monkeypatch.setattr("ask_the_docs.answer.retrieve", lambda query, settings, k: chunks)
    monkeypatch.setattr(
        "ask_the_docs.answer.get_llm_client",
        lambda config: Mock(
            generate_with_usage=Mock(
                return_value=GenerationResult(text="the answer [1]", input_tokens=None, output_tokens=None)
            )
        ),
    )

    settings = make_settings()
    result = _compute_answer("What is X?", settings, k=5)

    assert result.text == "the answer [1]"
    attrs = fake_tracer.spans[0].attrs
    assert "gen_ai.usage.input_tokens" not in attrs
    assert "gen_ai.usage.output_tokens" not in attrs
    # response_model absent on the GenerationResult -- falls back to the requested model.
    assert attrs["gen_ai.response.model"] == settings.generator.model


def test_compute_answer_increments_citation_guardrail_counter(monkeypatch):
    chunks = [Document(page_content="chunk one", metadata={"source": "a.pdf"})]
    monkeypatch.setattr("ask_the_docs.answer.retrieve", lambda query, settings, k: chunks)
    monkeypatch.setattr(
        "ask_the_docs.answer.get_llm_client",
        lambda config: Mock(
            generate_with_usage=Mock(
                return_value=GenerationResult(text="the answer [1]", input_tokens=None, output_tokens=None)
            )
        ),
    )
    before = _counter_value("citation_guardrail", result="pass")

    _compute_answer("What is X?", make_settings(), k=5)

    after = _counter_value("citation_guardrail", result="pass")
    assert after - before == 1.0


def test_compute_answer_increments_refusal_counter_when_no_chunks_retrieved(monkeypatch):
    monkeypatch.setattr("ask_the_docs.answer.retrieve", lambda query, settings, k: [])
    before = _counter_value("refusal")

    _compute_answer("anything", make_settings(), k=5)

    after = _counter_value("refusal")
    assert after - before == 1.0


def test_compute_answer_increments_refusal_counter_when_guardrail_fails_after_retries(monkeypatch):
    chunks = [Document(page_content="chunk one", metadata={"source": "a.pdf"})]
    monkeypatch.setattr("ask_the_docs.answer.retrieve", lambda query, settings, k: chunks)
    monkeypatch.setattr(
        "ask_the_docs.answer.get_llm_client",
        lambda config: Mock(
            generate_with_usage=Mock(
                return_value=GenerationResult(
                    text="an answer with no citation markers", input_tokens=None, output_tokens=None
                )
            )
        ),
    )
    before = _counter_value("refusal")

    _compute_answer("some question", make_settings(), k=5)

    after = _counter_value("refusal")
    assert after - before == 1.0


def test_compute_answer_observes_gen_ai_token_usage_and_duration_histograms(monkeypatch):
    chunks = [Document(page_content="chunk one", metadata={"source": "a.pdf"})]
    monkeypatch.setattr("ask_the_docs.answer.retrieve", lambda query, settings, k: chunks)
    settings = make_settings()
    monkeypatch.setattr(
        "ask_the_docs.answer.get_llm_client",
        lambda config: Mock(
            generate_with_usage=Mock(
                return_value=GenerationResult(text="the answer [1]", input_tokens=12, output_tokens=34)
            )
        ),
    )
    duration_attrs = {
        "gen_ai.operation.name": "chat",
        "gen_ai.provider.name": settings.generator.provider,
        "gen_ai.request.model": settings.generator.model,
    }
    usage_attrs = {**duration_attrs, "gen_ai.token.modality": "text"}
    before_input = _counter_value("gen_ai.client.inference.usage.input_tokens", **usage_attrs)
    before_output = _counter_value("gen_ai.client.inference.usage.output_tokens", **usage_attrs)
    before_duration = _histogram_count("gen_ai.client.inference.duration", **duration_attrs)
    before_generate = _histogram_count("generate_latency")

    _compute_answer("What is X?", settings, k=5)

    assert _counter_value("gen_ai.client.inference.usage.input_tokens", **usage_attrs) - before_input == 12
    assert _counter_value("gen_ai.client.inference.usage.output_tokens", **usage_attrs) - before_output == 34
    assert _histogram_count("gen_ai.client.inference.duration", **duration_attrs) - before_duration == 1
    assert _histogram_count("generate_latency") - before_generate == 1


def test_compute_answer_skips_token_usage_histogram_when_tokens_are_none(monkeypatch):
    chunks = [Document(page_content="chunk one", metadata={"source": "a.pdf"})]
    monkeypatch.setattr("ask_the_docs.answer.retrieve", lambda query, settings, k: chunks)
    settings = make_settings()
    monkeypatch.setattr(
        "ask_the_docs.answer.get_llm_client",
        lambda config: Mock(
            generate_with_usage=Mock(
                return_value=GenerationResult(text="the answer [1]", input_tokens=None, output_tokens=None)
            )
        ),
    )
    usage_attrs = {
        "gen_ai.operation.name": "chat",
        "gen_ai.provider.name": settings.generator.provider,
        "gen_ai.request.model": settings.generator.model,
        "gen_ai.token.modality": "text",
    }
    before_input = _counter_value("gen_ai.client.inference.usage.input_tokens", **usage_attrs)

    result = _compute_answer("What is X?", settings, k=5)

    assert result.text == "the answer [1]"
    assert _counter_value("gen_ai.client.inference.usage.input_tokens", **usage_attrs) == before_input


def test_answer_logs_prompt_at_debug_level(monkeypatch, caplog):
    chunks = [Document(page_content="chunk one", metadata={"source": "a.pdf"})]
    monkeypatch.setattr("ask_the_docs.answer.retrieve", lambda query, settings, k: chunks)
    monkeypatch.setattr(
        "ask_the_docs.answer.get_llm_client",
        lambda config: Mock(
            generate_with_usage=Mock(
                return_value=GenerationResult(text="the answer [1]", input_tokens=None, output_tokens=None)
            )
        ),
    )

    with caplog.at_level(logging.DEBUG, logger="ask_the_docs.answer"):
        answer("What is X?", make_settings())

    assert any("What is X?" in record.message for record in caplog.records)


def test_answer_returns_cached_result_without_calling_generator(monkeypatch):
    monkeypatch.setattr("ask_the_docs.answer.get_index_generation", lambda settings: 1)
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

    before_hit = _counter_value("cache_result", result="hit")

    result = answer("What is X?", settings, k=5)

    assert result == cached_result
    mock_retrieve.assert_not_called()
    mock_get_llm_client.assert_not_called()
    assert _counter_value("cache_result", result="hit") - before_hit == 1.0


def test_answer_increments_cache_miss_counter_on_cache_miss(monkeypatch):
    chunks = [Document(page_content="chunk one", metadata={"source": "a.pdf"})]
    monkeypatch.setattr("ask_the_docs.answer.retrieve", lambda query, settings, k: chunks)
    monkeypatch.setattr(
        "ask_the_docs.answer.get_llm_client",
        lambda config: Mock(
            generate_with_usage=Mock(
                return_value=GenerationResult(text="the answer [1]", input_tokens=None, output_tokens=None)
            )
        ),
    )
    before_miss = _counter_value("cache_result", result="miss")

    answer("a fresh question", make_settings(), k=5)

    assert _counter_value("cache_result", result="miss") - before_miss == 1.0


def test_answer_recomputes_when_cached_value_is_corrupted(monkeypatch):
    chunks = [Document(page_content="chunk one", metadata={"source": "a.pdf"})]
    monkeypatch.setattr("ask_the_docs.answer.retrieve", lambda query, settings, k: chunks)
    monkeypatch.setattr(
        "ask_the_docs.answer.get_llm_client",
        lambda config: Mock(
            generate_with_usage=Mock(
                return_value=GenerationResult(text="the answer [1]", input_tokens=None, output_tokens=None)
            )
        ),
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
    before_hit = _counter_value("cache_result", result="hit")
    before_miss = _counter_value("cache_result", result="miss")

    result = answer("What is X?", make_settings(), k=5)

    assert result.text == "the answer [1]"
    # A corrupted cache value is treated as a miss, not a hit -- it was
    # never a usable AnswerResult to begin with.
    assert _counter_value("cache_result", result="hit") - before_hit == 0
    assert _counter_value("cache_result", result="miss") - before_miss == 1.0


def test_answer_calls_generator_once_for_repeated_identical_calls(monkeypatch):
    chunks = [Document(page_content="chunk one", metadata={"source": "a.pdf"})]
    monkeypatch.setattr("ask_the_docs.answer.retrieve", lambda query, settings, k: chunks)
    monkeypatch.setattr("ask_the_docs.answer.get_index_generation", lambda settings: 1)
    shared_cache = FakeCacheClient()
    monkeypatch.setattr("ask_the_docs.answer.get_cache_client", lambda settings: shared_cache)

    calls = {"count": 0}

    class FakeClient:
        def generate_with_usage(self, prompt, system=None):
            calls["count"] += 1
            return GenerationResult(text="the answer [1]", input_tokens=None, output_tokens=None)

    monkeypatch.setattr("ask_the_docs.answer.get_llm_client", lambda config: FakeClient())
    settings = make_settings()

    answer("What is X?", settings, k=5)
    answer("What is X?", settings, k=5)

    assert calls["count"] == 1


def test_answer_cache_key_changes_when_system_prompt_template_changes(monkeypatch):
    chunks = [Document(page_content="chunk one", metadata={"source": "a.pdf"})]
    monkeypatch.setattr("ask_the_docs.answer.retrieve", lambda query, settings, k: chunks)
    monkeypatch.setattr("ask_the_docs.answer.get_index_generation", lambda settings: 1)
    shared_cache = FakeCacheClient()
    monkeypatch.setattr("ask_the_docs.answer.get_cache_client", lambda settings: shared_cache)

    calls = {"count": 0}

    class FakeClient:
        def generate_with_usage(self, prompt, system=None):
            calls["count"] += 1
            return GenerationResult(text="the answer [1]", input_tokens=None, output_tokens=None)

    monkeypatch.setattr("ask_the_docs.answer.get_llm_client", lambda config: FakeClient())

    settings_a = make_settings().model_copy(update={"system_prompt_template": "be terse"})
    settings_b = make_settings().model_copy(update={"system_prompt_template": "be verbose"})

    answer("What is X?", settings_a, k=5)
    answer("What is X?", settings_b, k=5)

    assert calls["count"] == 2


def test_answer_cache_key_changes_when_candidate_pool_size_changes(monkeypatch):
    chunks = [Document(page_content="chunk one", metadata={"source": "a.pdf"})]
    monkeypatch.setattr("ask_the_docs.answer.retrieve", lambda query, settings, k: chunks)
    monkeypatch.setattr("ask_the_docs.answer.get_index_generation", lambda settings: 1)
    shared_cache = FakeCacheClient()
    monkeypatch.setattr("ask_the_docs.answer.get_cache_client", lambda settings: shared_cache)

    calls = {"count": 0}

    class FakeClient:
        def generate_with_usage(self, prompt, system=None):
            calls["count"] += 1
            return GenerationResult(text="the answer [1]", input_tokens=None, output_tokens=None)

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
    monkeypatch.setattr("ask_the_docs.answer.get_index_generation", lambda settings: 1)
    shared_cache = FakeCacheClient()
    monkeypatch.setattr("ask_the_docs.answer.get_cache_client", lambda settings: shared_cache)

    calls = {"count": 0}

    class FakeClient:
        def generate_with_usage(self, prompt, system=None):
            calls["count"] += 1
            return GenerationResult(text="the answer [1]", input_tokens=None, output_tokens=None)

    monkeypatch.setattr("ask_the_docs.answer.get_llm_client", lambda config: FakeClient())

    settings_a = make_settings().model_copy(update={field: value_a})
    settings_b = make_settings().model_copy(update={field: value_b})

    answer("What is X?", settings_a, k=5)
    answer("What is X?", settings_b, k=5)

    assert calls["count"] == 2


def test_answer_cache_key_changes_when_index_generation_changes(monkeypatch):
    chunks = [Document(page_content="chunk one", metadata={"source": "a.pdf"})]
    monkeypatch.setattr("ask_the_docs.answer.retrieve", lambda query, settings, k: chunks)

    generations = [2, 3]
    monkeypatch.setattr("ask_the_docs.answer.get_index_generation", lambda settings: generations.pop(0))
    shared_cache = FakeCacheClient()
    monkeypatch.setattr("ask_the_docs.answer.get_cache_client", lambda settings: shared_cache)

    calls = {"count": 0}

    class FakeClient:
        def generate_with_usage(self, prompt, system=None):
            calls["count"] += 1
            return GenerationResult(text="the answer [1]", input_tokens=None, output_tokens=None)

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
        def generate_with_usage(self, prompt, system=None):
            captured["system"] = system
            return GenerationResult(text="the answer [1]", input_tokens=None, output_tokens=None)

    monkeypatch.setattr("ask_the_docs.answer.get_llm_client", lambda config: FakeClient())
    settings = make_settings()

    answer("q", settings)

    assert captured["system"] == settings.system_prompt_template
