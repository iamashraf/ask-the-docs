from ask_the_docs.config import ProviderConfig, Settings
from ask_the_docs.query_analyzer import (
    QueryClassification,
    _strip_code_fence,
    classify_query,
    rewrite_query,
)


def make_settings() -> Settings:
    return Settings(
        generator=ProviderConfig(provider="ollama", model="llama3.1:8b"),
        judge=ProviderConfig(provider="anthropic", model="claude-sonnet-5-5", api_key="x"),
    )


class FakeLLMClient:
    def __init__(self):
        self.captured_prompt = None

    def generate(self, prompt: str) -> str:
        self.captured_prompt = prompt
        return "  rewritten search query  \n"


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
    class FakeClient:
        def generate(self, prompt, system=None):
            return '{"label": "local", "entities": ["BM25", "vector search"]}'

    monkeypatch.setattr("ask_the_docs.query_analyzer.get_llm_client", lambda cfg: FakeClient())

    result = classify_query("how does BM25 relate to vector search?", make_settings())

    assert result == QueryClassification(label="local", entities=["BM25", "vector search"])


def test_classify_query_parses_global_with_no_entities(monkeypatch):
    class FakeClient:
        def generate(self, prompt, system=None):
            return '{"label": "global", "entities": []}'

    monkeypatch.setattr("ask_the_docs.query_analyzer.get_llm_client", lambda cfg: FakeClient())

    result = classify_query("what are the main themes of this corpus?", make_settings())

    assert result == QueryClassification(label="global", entities=[])


def test_classify_query_parses_fenced_json(monkeypatch):
    class FakeClient:
        def generate(self, prompt, system=None):
            return '```json\n{"label": "global", "entities": []}\n```'

    monkeypatch.setattr("ask_the_docs.query_analyzer.get_llm_client", lambda cfg: FakeClient())

    result = classify_query("what are the main themes of this corpus?", make_settings())

    assert result == QueryClassification(label="global", entities=[])


def test_classify_query_parses_bare_fenced_json(monkeypatch):
    class FakeClient:
        def generate(self, prompt, system=None):
            return '```\n{"label": "global", "entities": []}\n```'

    monkeypatch.setattr("ask_the_docs.query_analyzer.get_llm_client", lambda cfg: FakeClient())

    result = classify_query("what are the main themes of this corpus?", make_settings())

    assert result == QueryClassification(label="global", entities=[])


def test_classify_query_defaults_to_local_on_malformed_output(monkeypatch):
    class FakeClient:
        def generate(self, prompt, system=None):
            return "not json"

    monkeypatch.setattr("ask_the_docs.query_analyzer.get_llm_client", lambda cfg: FakeClient())

    result = classify_query("anything", make_settings())

    assert result == QueryClassification(label="local", entities=[])


def test_classify_query_defaults_to_local_on_invalid_label(monkeypatch):
    class FakeClient:
        def generate(self, prompt, system=None):
            return '{"label": "sideways", "entities": []}'

    monkeypatch.setattr("ask_the_docs.query_analyzer.get_llm_client", lambda cfg: FakeClient())

    result = classify_query("anything", make_settings())

    assert result == QueryClassification(label="local", entities=[])


def test_classify_query_defaults_to_local_when_entities_not_a_list(monkeypatch):
    class FakeClient:
        def generate(self, prompt, system=None):
            return '{"label": "local", "entities": "BM25"}'

    monkeypatch.setattr("ask_the_docs.query_analyzer.get_llm_client", lambda cfg: FakeClient())

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
    class FakeClient:
        def generate(self, prompt, system=None):
            return None

    monkeypatch.setattr("ask_the_docs.query_analyzer.get_llm_client", lambda cfg: FakeClient())

    result = classify_query("anything", make_settings())

    assert result == QueryClassification(label="local", entities=[])


def test_classify_query_includes_query_in_prompt(monkeypatch):
    captured = {}

    class FakeClient:
        def generate(self, prompt, system=None):
            captured["prompt"] = prompt
            return '{"label": "local", "entities": []}'

    monkeypatch.setattr("ask_the_docs.query_analyzer.get_llm_client", lambda cfg: FakeClient())

    classify_query("how does BM25 relate to vector search?", make_settings())

    assert "how does BM25 relate to vector search?" in captured["prompt"]
