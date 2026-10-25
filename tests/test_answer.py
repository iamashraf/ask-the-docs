import logging
from unittest.mock import Mock

from langchain_core.documents import Document

from research_assistant.answer import (
    AnswerResult,
    answer,
    build_prompt,
    citation_guardrail_passes,
    format_context,
)
from research_assistant.config import REFUSAL_MESSAGE, ProviderConfig, Settings


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
    monkeypatch.setattr("research_assistant.answer.retrieve", lambda query, settings, k: [])
    mock_get_llm_client = Mock()
    monkeypatch.setattr("research_assistant.answer.get_llm_client", mock_get_llm_client)

    result = answer("anything", make_settings())

    assert result == AnswerResult(text=REFUSAL_MESSAGE, citations=[])
    mock_get_llm_client.assert_not_called()


def test_answer_calls_generator_with_context_and_returns_citations(monkeypatch):
    chunks = [
        Document(page_content="chunk one", metadata={"source": "a.pdf"}),
        Document(page_content="chunk two", metadata={"source": "b.pdf"}),
    ]
    monkeypatch.setattr("research_assistant.answer.retrieve", lambda query, settings, k: chunks)

    captured_prompts = []

    class FakeClient:
        def generate(self, prompt, system=None):
            captured_prompts.append(prompt)
            return "the answer [1][2]"

    monkeypatch.setattr("research_assistant.answer.get_llm_client", lambda config: FakeClient())

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
    monkeypatch.setattr("research_assistant.answer.retrieve", lambda query, settings, k: chunks)
    monkeypatch.setattr(
        "research_assistant.answer.get_llm_client",
        lambda config: Mock(generate=Mock(return_value="the answer [1][2][3]")),
    )

    result = answer("What is X?", make_settings())

    assert result.citations == ["a.pdf", "b.pdf"]


def test_answer_returns_refusal_message_when_guardrail_fails(monkeypatch):
    chunks = [Document(page_content="chunk one", metadata={"source": "a.pdf"})]
    monkeypatch.setattr("research_assistant.answer.retrieve", lambda query, settings, k: chunks)
    monkeypatch.setattr(
        "research_assistant.answer.get_llm_client",
        lambda config: Mock(generate=Mock(return_value="an answer with no citation markers")),
    )

    result = answer("some question", make_settings())

    assert result == AnswerResult(text=REFUSAL_MESSAGE, citations=[])


def test_answer_logs_prompt_at_debug_level(monkeypatch, caplog):
    chunks = [Document(page_content="chunk one", metadata={"source": "a.pdf"})]
    monkeypatch.setattr("research_assistant.answer.retrieve", lambda query, settings, k: chunks)
    monkeypatch.setattr(
        "research_assistant.answer.get_llm_client",
        lambda config: Mock(generate=Mock(return_value="the answer [1]")),
    )

    with caplog.at_level(logging.DEBUG, logger="research_assistant.answer"):
        answer("What is X?", make_settings())

    assert any("What is X?" in record.message for record in caplog.records)


def test_answer_passes_system_prompt_to_generate(monkeypatch):
    chunks = [Document(page_content="chunk one", metadata={"source": "a.pdf"})]
    monkeypatch.setattr("research_assistant.answer.retrieve", lambda query, settings, k: chunks)

    captured = {}

    class FakeClient:
        def generate(self, prompt, system=None):
            captured["system"] = system
            return "the answer [1]"

    monkeypatch.setattr("research_assistant.answer.get_llm_client", lambda config: FakeClient())
    settings = make_settings()

    answer("q", settings)

    assert captured["system"] == settings.system_prompt_template
