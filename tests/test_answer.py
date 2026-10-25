from unittest.mock import Mock

from langchain_core.documents import Document

from research_assistant.answer import NO_CONTEXT_MESSAGE, AnswerResult, answer, build_prompt, format_context
from research_assistant.config import ProviderConfig, Settings


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

    prompt = build_prompt("What is X?", chunks)

    assert "What is X?" in prompt
    assert "chunk one" in prompt


def test_answer_returns_no_context_message_without_calling_llm(monkeypatch):
    monkeypatch.setattr("research_assistant.answer.retrieve", lambda query, settings, k: [])
    mock_get_llm_client = Mock()
    monkeypatch.setattr("research_assistant.answer.get_llm_client", mock_get_llm_client)

    result = answer("anything", make_settings())

    assert result == AnswerResult(text=NO_CONTEXT_MESSAGE, citations=[])
    mock_get_llm_client.assert_not_called()


def test_answer_calls_generator_with_context_and_returns_citations(monkeypatch):
    chunks = [
        Document(page_content="chunk one", metadata={"source": "a.pdf"}),
        Document(page_content="chunk two", metadata={"source": "b.pdf"}),
    ]
    monkeypatch.setattr("research_assistant.answer.retrieve", lambda query, settings, k: chunks)

    captured_prompts = []

    class FakeClient:
        def generate(self, prompt):
            captured_prompts.append(prompt)
            return "the answer"

    monkeypatch.setattr("research_assistant.answer.get_llm_client", lambda config: FakeClient())

    result = answer("What is X?", make_settings())

    assert result.text == "the answer"
    assert result.citations == ["a.pdf", "b.pdf"]
    assert "What is X?" in captured_prompts[0]
    assert "chunk one" in captured_prompts[0]
