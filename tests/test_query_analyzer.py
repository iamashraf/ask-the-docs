from research_assistant.config import ProviderConfig, Settings
from research_assistant.query_analyzer import rewrite_query


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
