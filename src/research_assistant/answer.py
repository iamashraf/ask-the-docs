"""Builds a prompt from retrieved context and generates a cited answer."""

from dataclasses import dataclass

from langchain_core.documents import Document

from research_assistant.config import Settings
from research_assistant.llm import get_llm_client
from research_assistant.retrieve import retrieve

NO_CONTEXT_MESSAGE = "no relevant context found"


@dataclass
class AnswerResult:
    text: str
    citations: list[str]
    context: str = ""


def format_context(chunks: list[Document]) -> str:
    """Numbers each chunk with its source; [n] markers double as citation anchors."""
    return "\n\n".join(
        f"[{i + 1}] (source: {chunk.metadata['source']})\n{chunk.page_content}"
        for i, chunk in enumerate(chunks)
    )


def build_prompt(query: str, context: str) -> str:
    return (
        "Answer the question using only the context below. "
        "Cite sources using their [n] marker.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}"
    )


def answer(query: str, settings: Settings, k: int = 5) -> AnswerResult:
    """Retrieves context and generates an answer; skips the LLM call entirely
    if nothing relevant was found."""
    chunks = retrieve(query, settings, k=k)
    if not chunks:
        return AnswerResult(text=NO_CONTEXT_MESSAGE, citations=[])

    context = format_context(chunks)
    client = get_llm_client(settings.generator)
    text = client.generate(build_prompt(query, context))
    citations = list(dict.fromkeys(chunk.metadata["source"] for chunk in chunks))
    return AnswerResult(text=text, citations=citations, context=context)
