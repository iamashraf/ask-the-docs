"""Hybrid retrieval orchestration: optional LLM query rewrite, vector +
BM25 search merged by a weighted blend, optional cross-encoder rerank."""

import logging

from langchain_core.documents import Document

from research_assistant.bm25_index import score as bm25_score
from research_assistant.config import Settings
from research_assistant.graph_retrieve import global_search, local_search
from research_assistant.index import load_index
from research_assistant.query_analyzer import QueryClassification, classify_query, rewrite_query
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
    search with optional query rewrite and reranking, with a graph-aware
    "global" short-circuit and "local" candidate merge when graph_enabled;
    [] if the index has no chunks yet."""
    search_query = query
    if settings.query_rewrite_enabled:
        try:
            search_query = rewrite_query(query, settings)
        except Exception:
            logger.warning("Query rewrite failed; falling back to the raw query.", exc_info=True)

    classification = QueryClassification(label="local", entities=[])
    if settings.graph_enabled:
        try:
            classification = classify_query(query, settings)
        except Exception:
            logger.warning("Query classification failed; defaulting to local search.", exc_info=True)

        if classification.label == "global":
            try:
                global_results = global_search(query, settings)
            except Exception:
                logger.warning("Global search failed; falling back to hybrid search.", exc_info=True)
                global_results = []
            if global_results:
                return global_results[:k]

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
    if classification.entities:
        # Appended after the pool-size truncation, not before: vector_hits alone
        # already fills candidate_pool_size on any corpus of that size or larger,
        # so truncating after the union would silently drop every graph-only
        # match almost always. Letting the pool grow slightly past
        # candidate_pool_size here is intended -- rerank()/candidates[:k] below
        # both already handle an arbitrary-length candidate list.
        try:
            local_results = local_search(classification.entities, settings)
        except Exception:
            logger.warning("Local search failed; continuing with hybrid candidates only.", exc_info=True)
            local_results = []
        seen = {_doc_key(doc) for doc in candidates}
        candidates += [doc for doc in local_results if _doc_key(doc) not in seen]

    if settings.rerank_enabled:
        return get_reranker(DEFAULT_RERANKER_MODEL).rerank(query, candidates, k)
    return candidates[:k]
