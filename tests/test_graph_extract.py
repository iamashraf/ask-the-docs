import logging

from research_assistant.config import ProviderConfig, Settings
from research_assistant.graph_extract import Triplet, extract_triplets


def make_settings() -> Settings:
    return Settings(
        generator=ProviderConfig(provider="ollama", model="llama3.1:8b"),
        judge=ProviderConfig(provider="anthropic", model="claude-sonnet-5-5", api_key="x"),
    )


def test_extract_triplets_parses_llm_json(monkeypatch):
    class FakeClient:
        def generate(self, prompt, system=None):
            return '[{"subject": "Transformer", "relation": "introduced-in", "object": "Attention Is All You Need"}]'

    monkeypatch.setattr("research_assistant.graph_extract.get_llm_client", lambda cfg: FakeClient())

    result = extract_triplets("some chunk text", make_settings())

    assert result == [Triplet(subject="Transformer", relation="introduced-in", obj="Attention Is All You Need")]


def test_extract_triplets_returns_empty_list_on_malformed_json(monkeypatch):
    class FakeClient:
        def generate(self, prompt, system=None):
            return "not json"

    monkeypatch.setattr("research_assistant.graph_extract.get_llm_client", lambda cfg: FakeClient())

    assert extract_triplets("some chunk text", make_settings()) == []


def test_extract_triplets_returns_empty_list_when_items_missing_required_field(monkeypatch):
    class FakeClient:
        def generate(self, prompt, system=None):
            return '[{"subject": "Transformer"}]'

    monkeypatch.setattr("research_assistant.graph_extract.get_llm_client", lambda cfg: FakeClient())

    assert extract_triplets("some chunk text", make_settings()) == []


def test_extract_triplets_logs_warning_on_malformed_json(monkeypatch, caplog):
    class FakeClient:
        def generate(self, prompt, system=None):
            return "not json"

    monkeypatch.setattr("research_assistant.graph_extract.get_llm_client", lambda cfg: FakeClient())

    with caplog.at_level(logging.WARNING, logger="research_assistant.graph_extract"):
        extract_triplets("some chunk text", make_settings())

    assert any("invalid JSON" in record.message for record in caplog.records)


def test_extract_triplets_returns_empty_list_when_response_is_dict_not_array(monkeypatch):
    class FakeClient:
        def generate(self, prompt, system=None):
            return '{"triplets": [{"subject": "A", "relation": "uses", "object": "B"}]}'

    monkeypatch.setattr("research_assistant.graph_extract.get_llm_client", lambda cfg: FakeClient())

    assert extract_triplets("some chunk text", make_settings()) == []


def test_extract_triplets_includes_chunk_text_in_prompt(monkeypatch):
    captured = {}

    class FakeClient:
        def generate(self, prompt, system=None):
            captured["prompt"] = prompt
            return "[]"

    monkeypatch.setattr("research_assistant.graph_extract.get_llm_client", lambda cfg: FakeClient())

    extract_triplets("some chunk text", make_settings())

    assert "some chunk text" in captured["prompt"]
