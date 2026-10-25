"""Builds a prompt from retrieved context and generates a cited, grounded
answer."""

import dataclasses
import json
import logging
import re

from langchain_core.documents import Document

from research_assistant.cache import cache_get, cache_set, get_cache_client, make_cache_key
from research_assistant.config import REFUSAL_MESSAGE, Settings
from research_assistant.index import get_chunk_count
from research_assistant.llm import get_llm_client
from research_assistant.retrieve import retrieve

logger = logging.getLogger(__name__)

CITATION_PATTERN = re.compile(r"\[(\d+)\]")
ANSWER_CACHE_TTL_SECONDS = 86400


@dataclasses.dataclass
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
    """User-turn content only -- grounding/citation/refusal instructions
    live in the system prompt (settings.system_prompt_template) instead."""
    return f"Context:\n{context}\n\nQuestion: {query}"


def citation_guardrail_passes(answer_text: str, num_chunks: int) -> bool:
    """Deterministic backstop: fails an answer with no [n] markers at all,
    or one that cites a chunk number that doesn't exist -- no retry, no
    extra LLM call, just a hard pass/fail on the model's own citations."""
    citations = CITATION_PATTERN.findall(answer_text)
    if not citations:
        return False
    try:
        return all(1 <= int(n) <= num_chunks for n in citations)
    except ValueError:
        # int() refuses absurdly long digit strings (its own guard against
        # quadratic-time conversion) -- treat that as an invalid citation,
        # not a crash.
        return False


def _compute_answer(query: str, settings: Settings, k: int) -> AnswerResult:
    """Retrieves context and generates a grounded answer; skips the LLM
    call entirely if nothing relevant was found, and refuses if the model's
    answer isn't actually grounded in what was retrieved."""
    chunks = retrieve(query, settings, k=k)
    if not chunks:
        return AnswerResult(text=REFUSAL_MESSAGE, citations=[])

    context = format_context(chunks)
    client = get_llm_client(settings.generator)
    prompt = build_prompt(query, context)
    logger.debug("LLM call -- system=%r, prompt=%r", settings.system_prompt_template, prompt)
    text = client.generate(prompt, system=settings.system_prompt_template)

    if not citation_guardrail_passes(text, num_chunks=len(chunks)):
        return AnswerResult(text=REFUSAL_MESSAGE, citations=[])

    citations = list(dict.fromkeys(chunk.metadata["source"] for chunk in chunks))
    return AnswerResult(text=text, citations=citations, context=context)


def answer(query: str, settings: Settings, k: int = 5) -> AnswerResult:
    """Exact-match cached wrapper around _compute_answer -- a hit skips
    retrieval, reranking, and the LLM call entirely. The corpus chunk count
    is part of the cache key, so a new ingest produces a new key rather
    than needing explicit cache invalidation."""
    cache_client = get_cache_client(settings)
    cache_key = make_cache_key(
        query.strip().lower(),
        k,
        settings.embeddings_backend,
        settings.generator.provider,
        settings.generator.model,
        settings.hybrid_alpha,
        settings.rerank_enabled,
        settings.query_rewrite_enabled,
        get_chunk_count(settings),
    )

    cached = cache_get(cache_client, cache_key)
    if cached is not None:
        try:
            return AnswerResult(**json.loads(cached))
        except (json.JSONDecodeError, TypeError):
            # A corrupted value or a schema mismatch from an older cache
            # entry -- treat it as a miss and recompute, rather than
            # letting a cache problem break the request.
            logger.warning("Cached answer value was unusable; recomputing.", exc_info=True)

    result = _compute_answer(query, settings, k)
    cache_set(cache_client, cache_key, json.dumps(dataclasses.asdict(result)).encode(), ttl_seconds=ANSWER_CACHE_TTL_SECONDS)
    return result
