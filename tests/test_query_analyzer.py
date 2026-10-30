from ask_the_docs.config import ProviderConfig, Settings
from ask_the_docs.llm import GenerationResult
from ask_the_docs.query_analyzer import (
    QueryClassification,
    _strip_code_fence,
    classify_query,
    rewrite_query,
)
from tests.test_metrics import find_data_point, find_metric, get_metrics_data


def make_settings() -> Settings:
    return Settings(
        generator=ProviderConfig(provider="ollama", model="llama3.1:8b"),
        judge=ProviderConfig(provider="anthropic", model="claude-sonnet-5-5", api_key="x"),
    )


def _histogram_count(name, **attributes):
    """Reads the running count for one data point; the shared InMemoryMetricReader is cumulative across the whole test session, so callers compare a before/after delta rather than an absolute value."""
    try:
        metric = find_metric(get_metrics_data(), name)
        return find_data_point(metric, **attributes).count
    except AssertionError:
        return 0


class FakeLLMClient:
    def __init__(self, response: str = "  rewritten search query  \n"):
        self.response = response
        self.captured_prompt = None

    def generate(self, prompt: str, system: str | None = None) -> str:
        self.captured_prompt = prompt
        return self.response

    def generate_with_usage(self, prompt: str, system: str | None = None) -> GenerationResult:
        self.captured_prompt = prompt
        return GenerationResult(text=self.response, input_tokens=5, output_tokens=7, response_model="fake-model")


def test_rewrite_query_returns_stripped_result(monkeypatch):
    fake_client = FakeLLMClient()
    monkeypatch.setattr("ask_the_docs.query_analyzer.get_llm_client", lambda _: fake_client)

    result = rewrite_query("what is attention?", make_settings())

    assert result == "rewritten search query"


def test_rewrite_query_includes_original_query_in_prompt(monkeypatch):
    fake_client = FakeLLMClient()
    monkeypatch.setattr("ask_the_docs.query_analyzer.get_llm_client", lambda _: fake_client)

    rewrite_query("what is attention?", make_settings())

    assert "what is attention?" in fake_client.captured_prompt


def test_classify_query_parses_local_with_entities(monkeypatch):
    fake_client = FakeLLMClient(response='{"label": "local", "entities": ["BM25", "vector search"]}')
    monkeypatch.setattr("ask_the_docs.query_analyzer.get_llm_client", lambda cfg: fake_client)

    result = classify_query("how does BM25 relate to vector search?", make_settings())

    assert result == QueryClassification(label="local", entities=["BM25", "vector search"])


def test_classify_query_parses_global_with_no_entities(monkeypatch):
    fake_client = FakeLLMClient(response='{"label": "global", "entities": []}')
    monkeypatch.setattr("ask_the_docs.query_analyzer.get_llm_client", lambda cfg: fake_client)

    result = classify_query("what are the main themes of this corpus?", make_settings())

    assert result == QueryClassification(label="global", entities=[])


def test_classify_query_parses_fenced_json(monkeypatch):
    fake_client = FakeLLMClient(response='```json\n{"label": "global", "entities": []}\n```')
    monkeypatch.setattr("ask_the_docs.query_analyzer.get_llm_client", lambda cfg: fake_client)

    result = classify_query("what are the main themes of this corpus?", make_settings())

    assert result == QueryClassification(label="global", entities=[])


def test_classify_query_parses_bare_fenced_json(monkeypatch):
    fake_client = FakeLLMClient(response='```\n{"label": "global", "entities": []}\n```')
    monkeypatch.setattr("ask_the_docs.query_analyzer.get_llm_client", lambda cfg: fake_client)

    result = classify_query("what are the main themes of this corpus?", make_settings())

    assert result == QueryClassification(label="global", entities=[])


def test_classify_query_defaults_to_local_on_malformed_output(monkeypatch):
    fake_client = FakeLLMClient(response="not json")
    monkeypatch.setattr("ask_the_docs.query_analyzer.get_llm_client", lambda cfg: fake_client)

    result = classify_query("anything", make_settings())

    assert result == QueryClassification(label="local", entities=[])


def test_classify_query_defaults_to_local_on_invalid_label(monkeypatch):
    fake_client = FakeLLMClient(response='{"label": "sideways", "entities": []}')
    monkeypatch.setattr("ask_the_docs.query_analyzer.get_llm_client", lambda cfg: fake_client)

    result = classify_query("anything", make_settings())

    assert result == QueryClassification(label="local", entities=[])


def test_classify_query_defaults_to_local_when_entities_not_a_list(monkeypatch):
    fake_client = FakeLLMClient(response='{"label": "local", "entities": "BM25"}')
    monkeypatch.setattr("ask_the_docs.query_analyzer.get_llm_client", lambda cfg: fake_client)

    result = classify_query("anything", make_settings())

    assert result == QueryClassification(label="local", entities=[])


def test_strip_code_fence_returns_non_str_input_unchanged():
    assert _strip_code_fence(None) is None


def test_strip_code_fence_strips_json_tagged_fence():
    assert _strip_code_fence('```json\n{"a": 1}\n```') == '{"a": 1}'


def test_strip_code_fence_strips_bare_fence():
    assert _strip_code_fence('```\n{"a": 1}\n```') == '{"a": 1}'


def test_strip_code_fence_leaves_unfenced_text_unchanged():
    assert _strip_code_fence('{"a": 1}') == '{"a": 1}'


def test_classify_query_defaults_to_local_on_non_str_output(monkeypatch):
    fake_client = FakeLLMClient(response=None)
    monkeypatch.setattr("ask_the_docs.query_analyzer.get_llm_client", lambda cfg: fake_client)

    result = classify_query("anything", make_settings())

    assert result == QueryClassification(label="local", entities=[])


def test_classify_query_includes_query_in_prompt(monkeypatch):
    fake_client = FakeLLMClient(response='{"label": "local", "entities": []}')
    monkeypatch.setattr("ask_the_docs.query_analyzer.get_llm_client", lambda cfg: fake_client)

    classify_query("how does BM25 relate to vector search?", make_settings())

    assert "how does BM25 relate to vector search?" in fake_client.captured_prompt


GEN_AI_DURATION_ATTRS = {
    "gen_ai.operation.name": "chat",
    "gen_ai.provider.name": "ollama",
    "gen_ai.request.model": "llama3.1:8b",
}


def test_rewrite_query_records_gen_ai_inference_duration(monkeypatch):
    fake_client = FakeLLMClient()
    monkeypatch.setattr("ask_the_docs.query_analyzer.get_llm_client", lambda _: fake_client)

    before = _histogram_count("gen_ai.client.inference.duration", **GEN_AI_DURATION_ATTRS)
    rewrite_query("what is attention?", make_settings())

    assert _histogram_count("gen_ai.client.inference.duration", **GEN_AI_DURATION_ATTRS) - before == 1


def test_classify_query_records_gen_ai_inference_duration(monkeypatch):
    fake_client = FakeLLMClient(response='{"label": "local", "entities": []}')
    monkeypatch.setattr("ask_the_docs.query_analyzer.get_llm_client", lambda cfg: fake_client)

    before = _histogram_count("gen_ai.client.inference.duration", **GEN_AI_DURATION_ATTRS)
    classify_query("anything", make_settings())

    assert _histogram_count("gen_ai.client.inference.duration", **GEN_AI_DURATION_ATTRS) - before == 1


def test_classify_query_records_gen_ai_inference_duration_on_fallback_path(monkeypatch):
    """The span/metrics record even when parsing fails and classify_query falls back to label="local"."""
    fake_client = FakeLLMClient(response="not json")
    monkeypatch.setattr("ask_the_docs.query_analyzer.get_llm_client", lambda cfg: fake_client)

    before = _histogram_count("gen_ai.client.inference.duration", **GEN_AI_DURATION_ATTRS)
    classify_query("anything", make_settings())

    assert _histogram_count("gen_ai.client.inference.duration", **GEN_AI_DURATION_ATTRS) - before == 1
