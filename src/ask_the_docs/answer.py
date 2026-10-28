"""Builds a prompt from retrieved context and generates a cited, grounded
answer."""

import dataclasses
import json
import logging
import re

from langchain_core.documents import Document

from ask_the_docs.cache import cache_get, cache_set, get_cache_client, make_cache_key
from ask_the_docs.config import REFUSAL_MESSAGE, Settings
from ask_the_docs.index import get_index_generation
from ask_the_docs.llm import get_llm_client
from ask_the_docs.retrieve import retrieve

logger = logging.getLogger(__name__)

CITATION_PATTERN = re.compile(r"\[(\d+)\]")
ANSWER_CACHE_TTL_SECONDS = 86400
# A local LLM's citation formatting is non-deterministic -- it occasionally
# fails to emit [n] markers (or cites an invalid one) on a question it could
# otherwise answer correctly. One retry resolves the large majority of these
# without materially changing worst-case latency.
MAX_GENERATION_ATTEMPTS = 2


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
    or one that cites a chunk number that doesn't exist. Pure pass/fail on
    one piece of text -- the retry-on-failure policy lives in the caller,
    not here."""
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

    for attempt in range(1, MAX_GENERATION_ATTEMPTS + 1):
        text = client.generate(prompt, system=settings.system_prompt_template)
        if citation_guardrail_passes(text, num_chunks=len(chunks)):
            citations = list(dict.fromkeys(chunk.metadata["source"] for chunk in chunks))
            return AnswerResult(text=text, citations=citations, context=context)
        logger.warning(
            "Generated answer failed the citation guardrail (attempt %d/%d).",
            attempt,
            MAX_GENERATION_ATTEMPTS,
        )

    return AnswerResult(text=REFUSAL_MESSAGE, citations=[])


def answer(query: str, settings: Settings, k: int = 5) -> AnswerResult:
    """Exact-match cached wrapper around _compute_answer -- a hit skips
    retrieval, reranking, and the LLM call entirely. The index generation
    is part of the cache key, so any real ingest write (including a
    same-chunk-count replace) produces a new key rather than needing
    explicit cache invalidation; reading it never touches Chroma, so
    computing the key doesn't depend on the index being reachable."""
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
        settings.system_prompt_template,
        settings.candidate_pool_size,
        settings.graph_enabled,
        settings.graph_traversal_hop_depth,
        settings.global_search_top_communities,
        get_index_generation(settings),
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
