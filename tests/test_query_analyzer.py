from research_assistant.config import ProviderConfig, Settings
from research_assistant.query_analyzer import QueryClassification, classify_query, rewrite_query


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
    monkeypatch.setattr("research_assistant.query_analyzer.get_llm_client", lambda _: fake_client)

    result = rewrite_query("what is attention?", make_settings())

    assert result == "rewritten search query"


def test_rewrite_query_includes_original_query_in_prompt(monkeypatch):
    fake_client = FakeLLMClient()
    monkeypatch.setattr("research_assistant.query_analyzer.get_llm_client", lambda _: fake_client)

    rewrite_query("what is attention?", make_settings())

    assert "what is attention?" in fake_client.captured_prompt


def test_classify_query_parses_local_with_entities(monkeypatch):
    class FakeClient:
        def generate(self, prompt, system=None):
            return '{"label": "local", "entities": ["BM25", "vector search"]}'

    monkeypatch.setattr("research_assistant.query_analyzer.get_llm_client", lambda cfg: FakeClient())

    result = classify_query("how does BM25 relate to vector search?", make_settings())

    assert result == QueryClassification(label="local", entities=["BM25", "vector search"])


def test_classify_query_parses_global_with_no_entities(monkeypatch):
    class FakeClient:
        def generate(self, prompt, system=None):
            return '{"label": "global", "entities": []}'

    monkeypatch.setattr("research_assistant.query_analyzer.get_llm_client", lambda cfg: FakeClient())

    result = classify_query("what are the main themes of this corpus?", make_settings())

    assert result == QueryClassification(label="global", entities=[])


def test_classify_query_parses_fenced_json(monkeypatch):
    class FakeClient:
        def generate(self, prompt, system=None):
            return '```json\n{"label": "global", "entities": []}\n```'

    monkeypatch.setattr("research_assistant.query_analyzer.get_llm_client", lambda cfg: FakeClient())

    result = classify_query("what are the main themes of this corpus?", make_settings())

    assert result == QueryClassification(label="global", entities=[])


def test_classify_query_parses_bare_fenced_json(monkeypatch):
    class FakeClient:
        def generate(self, prompt, system=None):
            return '```\n{"label": "global", "entities": []}\n```'

    monkeypatch.setattr("research_assistant.query_analyzer.get_llm_client", lambda cfg: FakeClient())

    result = classify_query("what are the main themes of this corpus?", make_settings())

    assert result == QueryClassification(label="global", entities=[])


def test_classify_query_defaults_to_local_on_malformed_output(monkeypatch):
    class FakeClient:
        def generate(self, prompt, system=None):
            return "not json"

    monkeypatch.setattr("research_assistant.query_analyzer.get_llm_client", lambda cfg: FakeClient())

    result = classify_query("anything", make_settings())

    assert result == QueryClassification(label="local", entities=[])


def test_classify_query_defaults_to_local_on_invalid_label(monkeypatch):
    class FakeClient:
        def generate(self, prompt, system=None):
            return '{"label": "sideways", "entities": []}'

    monkeypatch.setattr("research_assistant.query_analyzer.get_llm_client", lambda cfg: FakeClient())

    result = classify_query("anything", make_settings())

    assert result == QueryClassification(label="local", entities=[])


def test_classify_query_defaults_to_local_when_entities_not_a_list(monkeypatch):
    class FakeClient:
        def generate(self, prompt, system=None):
            return '{"label": "local", "entities": "BM25"}'

    monkeypatch.setattr("research_assistant.query_analyzer.get_llm_client", lambda cfg: FakeClient())

    result = classify_query("anything", make_settings())

    assert result == QueryClassification(label="local", entities=[])


def test_classify_query_includes_query_in_prompt(monkeypatch):
    captured = {}

    class FakeClient:
        def generate(self, prompt, system=None):
            captured["prompt"] = prompt
            return '{"label": "local", "entities": []}'

    monkeypatch.setattr("research_assistant.query_analyzer.get_llm_client", lambda cfg: FakeClient())

    classify_query("how does BM25 relate to vector search?", make_settings())

    assert "how does BM25 relate to vector search?" in captured["prompt"]
