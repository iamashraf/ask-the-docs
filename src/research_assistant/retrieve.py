"""Hybrid retrieval orchestration: optional LLM query rewrite, vector +
BM25 search merged by a weighted blend, optional cross-encoder rerank."""

import logging

from langchain_core.documents import Document

from research_assistant.bm25_index import score as bm25_score
from research_assistant.config import Settings
from research_assistant.index import load_index
from research_assistant.query_analyzer import rewrite_query
from research_assistant.reranker import DEFAULT_RERANKER_MODEL, get_reranker

logger = logging.getLogger(__name__)


def _doc_key(doc: Document) -> tuple[str, int]:
    return (doc.metadata["source_doc_id"], doc.metadata["chunk_index"])


def _normalize(hits: list[tuple[Document, float]]) -> dict[tuple[str, int], float]:
    """Min-max normalize one side's raw scores to [0, 1]; an all-equal pool
    normalizes to 1.0 for everyone rather than dividing by zero."""
    if not hits:
        return {}
    scores = [s for _, s in hits]
    lo, hi = min(scores), max(scores)
    if hi == lo:
        return {_doc_key(doc): 1.0 for doc, _ in hits}
    return {_doc_key(doc): (s - lo) / (hi - lo) for doc, s in hits}


def hybrid_merge(
    vector_hits: list[tuple[Document, float]],
    bm25_hits: list[tuple[Document, float]],
    alpha: float,
) -> list[tuple[Document, float]]:
    """Union of both sides (nothing found by only one side is dropped),
    blended as alpha*vector + (1-alpha)*bm25 after independent normalization,
    sorted descending."""
    vector_norm = _normalize(vector_hits)
    bm25_norm = _normalize(bm25_hits)

    docs_by_key: dict[tuple[str, int], Document] = {}
    for doc, _ in vector_hits:
        docs_by_key[_doc_key(doc)] = doc
    for doc, _ in bm25_hits:
        docs_by_key[_doc_key(doc)] = doc

    merged = [
        (doc, alpha * vector_norm.get(key, 0.0) + (1 - alpha) * bm25_norm.get(key, 0.0))
        for key, doc in docs_by_key.items()
    ]
    merged.sort(key=lambda pair: pair[1], reverse=True)
    return merged


def retrieve(query: str, settings: Settings, k: int = 5) -> list[Document]:
    """Up to k chunks most relevant to query via hybrid (vector + BM25)
    search with optional query rewrite and reranking; [] if the index has
    no chunks yet."""
    search_query = query
    if settings.query_rewrite_enabled:
        try:
            search_query = rewrite_query(query, settings)
        except Exception:
            logger.warning("Query rewrite failed; falling back to the raw query.", exc_info=True)

    index = load_index(settings)
    # Chroma's distance is "lower = more similar"; negate so that, after
    # min-max normalization, the closest match ends up near 1.0 like BM25's
    # "higher = more relevant" scores, instead of inverting the ranking.
    vector_hits = [
        (doc, -distance)
        for doc, distance in index.similarity_search_with_score(search_query, k=settings.candidate_pool_size)
    ]
    # A score of 0 means "no keyword overlap at all" (e.g. an out-of-vocabulary
    # query) -- excluding it keeps that chunk from tying every other
    # zero-score chunk at normalized score 1.0 ("all equal" rule) and
    # crowding out real vector matches.
    bm25_hits = [
        hit
        for hit in sorted(bm25_score(search_query, settings), key=lambda pair: pair[1], reverse=True)
        if hit[1] > 0
    ][: settings.candidate_pool_size]

    candidates = [doc for doc, _ in hybrid_merge(vector_hits, bm25_hits, settings.hybrid_alpha)][
        : settings.candidate_pool_size
    ]

    if settings.rerank_enabled:
        return get_reranker(DEFAULT_RERANKER_MODEL).rerank(query, candidates, k)
    return candidates[:k]
