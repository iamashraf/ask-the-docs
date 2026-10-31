"""Builds a prompt from retrieved context and generates a cited, grounded
answer."""

import dataclasses
import json
import logging
import re
import time

from langchain_core.documents import Document
from opentelemetry import trace

from ask_the_docs import metrics
from ask_the_docs.cache import cache_get, cache_set, get_cache_client, make_cache_key
from ask_the_docs.config import REFUSAL_MESSAGE, Settings
from ask_the_docs.index import get_index_generation
from ask_the_docs.llm import (
    StructuredGenerationResult,
    StructuredOutputError,
    get_llm_client,
)
from ask_the_docs.retrieve import retrieve

logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)

CITATION_PATTERN = re.compile(r"\[(\d+)\]")
ANSWER_CACHE_TTL_SECONDS = 86400
# A local LLM's citation formatting is non-deterministic -- it occasionally
# fails to emit [n] markers (or cites an invalid one) on a question it could
# otherwise answer correctly. One retry resolves the large majority of these
# without materially changing worst-case latency.
MAX_GENERATION_ATTEMPTS = 2

# Ollama-only: constrains the model's own output to this schema (via Ollama's
# `format` field) instead of generating free text and regexing it for [n]
# markers afterwards. Anthropic/OpenAI structured output is out of scope --
# they keep using generate_with_usage + citation_guardrail_passes below.
STRUCTURED_ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "grounded": {"type": "boolean"},
        "answer": {"type": "string"},
        "citations": {"type": "array", "items": {"type": "integer"}},
    },
    "required": ["grounded", "answer", "citations"],
}

# Separate from config.DEFAULT_SYSTEM_PROMPT_TEMPLATE -- that one is being
# edited in parallel on another branch for the free-text path. This one
# explains the structured schema's fields instead of [n]-marker conventions
# in prose.
STRUCTURED_SYSTEM_PROMPT_TEMPLATE = (
    "You answer questions using only the numbered context chunks the user provides. "
    "Respond with JSON matching the required schema.\n\n"
    "Fields:\n"
    "- grounded: true only if the context chunks actually contain the information "
    "needed to answer the question; false if they do not.\n"
    "- answer: when grounded is true, a complete, substantive explanation of the "
    "answer in your own words, written as if explaining it to the user -- not a "
    "restatement of the question, not a bare label or one-line fragment, and not "
    "copied verbatim from the context. Mark every claim with the [n] marker of the "
    'context chunk it came from (e.g. "...uses a reservoir [1]."), the same way '
    "footnotes work. Each marker holds exactly one number -- write [1][2] for two "
    "sources, never [1, 2]. When grounded is false, a short sentence saying the "
    "context does not contain the answer, with no [n] markers.\n"
    "- citations: the integer numbers (e.g. 1, 2) of every context chunk whose [n] "
    "marker you used in answer. Leave this empty when grounded is false.\n\n"
    "If the question is hypothetical, asks you to speculate, or asks about something "
    "not stated in the context, set grounded to false."
)


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


def structured_answer_passes(result: StructuredGenerationResult, num_chunks: int) -> bool:
    """Structured-output counterpart to citation_guardrail_passes: trusts the
    model's own grounded/citations fields instead of regexing its prose."""
    if not result.grounded or not result.citations:
        return False
    return all(1 <= c <= num_chunks for c in result.citations)


def _structured_retry_feedback(result: StructuredGenerationResult | None) -> str:
    """Corrective, not blind: tells the model what was specifically wrong
    with its last attempt instead of resending the identical prompt."""
    if result is None:
        return "Your previous reply was not valid JSON. Reply again as JSON matching the schema exactly."
    if not result.grounded:
        return (
            "Your previous reply said the context did not contain the answer. "
            "Re-read the context chunks below carefully before answering again. "
            "If they genuinely do not answer the question, set grounded to false "
            "again -- do not guess."
        )
    return (
        "Your previous reply did not include valid citations. Answer again, "
        "marking every claim with the [n] marker of the context chunk it came "
        "from, and list those same numbers in citations."
    )


def _compute_answer(query: str, settings: Settings, k: int) -> AnswerResult:
    """Retrieves context and generates a grounded answer; skips the LLM
    call entirely if nothing relevant was found, and refuses if the model's
    answer isn't actually grounded in what was retrieved."""
    chunks = retrieve(query, settings, k=k)
    if not chunks:
        metrics.refusal_total.add(1)
        return AnswerResult(text=REFUSAL_MESSAGE, citations=[])

    context = format_context(chunks)
    client = get_llm_client(settings.generator)
    # retrieve() cleans the query up (typos, grammar) for its own search --
    # reuse that for generation too, so a malformed input doesn't throw off
    # the model's citation formatting even though the right context was
    # found. Falls back to the raw query if retrieve() is mocked/old and
    # returns a plain list with no .search_query.
    generation_query = getattr(chunks, "search_query", query)
    prompt = build_prompt(generation_query, context)
    logger.debug("LLM call -- system=%r, prompt=%r", settings.system_prompt_template, prompt)

    # Ollama gets schema-constrained structured output (grounded/answer/citations
    # read directly); other providers keep the free-text + regex guardrail, since
    # Anthropic/OpenAI structured output is out of scope here.
    use_structured = settings.generator.provider == "ollama"
    structured: StructuredGenerationResult | None = None
    feedback: str | None = None

    for attempt in range(1, MAX_GENERATION_ATTEMPTS + 1):
        attempt_prompt = prompt if feedback is None else f"{prompt}\n\n(Note: {feedback})"
        with tracer.start_as_current_span("generate") as span:
            generate_start = time.monotonic()

            if use_structured:
                try:
                    structured = client.generate_structured(
                        attempt_prompt, STRUCTURED_ANSWER_SCHEMA, system=STRUCTURED_SYSTEM_PROMPT_TEMPLATE
                    )
                except StructuredOutputError:
                    logger.warning(
                        "Ollama's structured response didn't parse (attempt %d/%d).",
                        attempt,
                        MAX_GENERATION_ATTEMPTS,
                    )
                    structured = None
                passed = structured is not None and structured_answer_passes(structured, num_chunks=len(chunks))
                text = structured.answer if structured is not None else ""
                input_tokens = structured.input_tokens if structured is not None else None
                output_tokens = structured.output_tokens if structured is not None else None
                response_model = structured.response_model if structured is not None else None
            else:
                result = client.generate_with_usage(attempt_prompt, system=settings.system_prompt_template)
                text = result.text
                passed = citation_guardrail_passes(text, num_chunks=len(chunks))
                input_tokens = result.input_tokens
                output_tokens = result.output_tokens
                response_model = result.response_model

            duration = time.monotonic() - generate_start

            span.set_attribute("gen_ai.request.model", settings.generator.model)
            span.set_attribute("gen_ai.response.model", response_model or settings.generator.model)
            span.set_attribute("gen_ai.provider.name", settings.generator.provider)
            attrs = {
                "gen_ai.operation.name": "chat",
                "gen_ai.provider.name": settings.generator.provider,
                "gen_ai.request.model": settings.generator.model,
            }
            if input_tokens is not None:
                span.set_attribute("gen_ai.usage.input_tokens", input_tokens)
                metrics.gen_ai_client_inference_usage_input_tokens.add(
                    input_tokens, {**attrs, "gen_ai.token.modality": "text"}
                )
            if output_tokens is not None:
                span.set_attribute("gen_ai.usage.output_tokens", output_tokens)
                metrics.gen_ai_client_inference_usage_output_tokens.add(
                    output_tokens, {**attrs, "gen_ai.token.modality": "text"}
                )
            metrics.gen_ai_client_inference_duration.record(duration, attrs)
            metrics.generate_latency_seconds.record(duration)

            span.set_attribute("citation_guardrail_passed", passed)
            if passed:
                logger.info(
                    "Answer generated: model=%s, attempt=%d/%d, chunks=%d",
                    settings.generator.model,
                    attempt,
                    MAX_GENERATION_ATTEMPTS,
                    len(chunks),
                )
        metrics.citation_guardrail_total.add(1, {"result": "pass" if passed else "fail"})
        if passed:
            citations = list(dict.fromkeys(chunk.metadata["source"] for chunk in chunks))
            return AnswerResult(text=text, citations=citations, context=context)
        logger.warning(
            "Generated answer failed the citation guardrail (attempt %d/%d).",
            attempt,
            MAX_GENERATION_ATTEMPTS,
        )
        if use_structured:
            feedback = _structured_retry_feedback(structured)

    metrics.refusal_total.add(1)
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
        # Not settings-controlled, but the Ollama path's actual generation
        # behavior changed with its introduction -- included so a pre-existing
        # cached entry (e.g. a stale refusal from the old regex guardrail)
        # never shadows it.
        STRUCTURED_SYSTEM_PROMPT_TEMPLATE,
    )

    cached = cache_get(cache_client, cache_key)
    if cached is not None:
        try:
            result = AnswerResult(**json.loads(cached))
            metrics.cache_result_total.add(1, {"result": "hit"})
            return result
        except (json.JSONDecodeError, TypeError):
            # A corrupted value or a schema mismatch from an older cache
            # entry -- treat it as a miss and recompute, rather than
            # letting a cache problem break the request.
            logger.warning("Cached answer value was unusable; recomputing.", exc_info=True)

    metrics.cache_result_total.add(1, {"result": "miss"})
    result = _compute_answer(query, settings, k)
    cache_set(cache_client, cache_key, json.dumps(dataclasses.asdict(result)).encode(), ttl_seconds=ANSWER_CACHE_TTL_SECONDS)
    return result
