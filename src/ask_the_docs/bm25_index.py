"""Keyword (BM25) search over the full chunk collection, cached in Valkey
so neither the CLI's one-shot processes nor the long-running API server
rebuild it from scratch on every query."""

import logging
import pickle
import re
import threading

import bm25s
from langchain_core.documents import Document

from ask_the_docs.cache import cache_get, cache_set, get_cache_client, make_cache_key
from ask_the_docs.config import Settings
from ask_the_docs.index import get_all_chunks, get_collection_name, get_index_generation

logger = logging.getLogger(__name__)

BM25_CACHE_TTL_SECONDS = 86400


class ChromaReadError(RuntimeError):
    """Raised by _rebuild_bm25_index when get_all_chunks fails -- distinct
    from a real empty corpus, which legitimately returns (None, [])."""

_WORD_PATTERN = re.compile(r"\w+")


def _tokenize(text: str) -> list[str]:
    # Plain str.split() keeps punctuation attached ("lora:" != "lora",
    # "lora?" != "lora"), which defeats exact keyword/acronym matching --
    # the whole reason BM25 is in the pipeline alongside vector search.
    return _WORD_PATTERN.findall(text.lower())


_bm25_cache: dict[tuple[str, int, str], tuple[bm25s.BM25 | None, list[Document]]] = {}
# 1, not 2: the generation only ever increases and is re-read fresh on
# every call (nothing captures and reuses a stale value across a
# request), so a previous generation's entry is never looked up again the
# instant a new one lands -- keeping it resident doubles memory for
# nothing.
_BM25_CACHE_MAXSIZE = 1
# Serializes the miss path only (a cache hit is a plain dict read, done
# before acquiring this). Without it, every request in flight right after
# a generation bump misses in parallel and independently pays the full
# Chroma-read + bm25s-index cost at once -- N rebuilds and N times the
# memory instead of one rebuild the rest wait on.
_bm25_cache_lock = threading.Lock()


def get_bm25_index(
    settings: Settings, raise_on_read_error: bool = False
) -> tuple[bm25s.BM25 | None, list[Document]]:
    """Resident per (collection_name, generation, index_dir) within this
    process, on top of the cross-process Valkey cache _rebuild_bm25_index
    still uses -- avoids a network round trip + pickle.loads() on every
    call, not just avoiding a rebuild from Chroma. index_dir is part of the
    key for the same collision reason load_index's cache includes it."""
    generation = get_index_generation(settings)
    key = (get_collection_name(settings), generation, str(settings.index_dir))
    if key in _bm25_cache:
        return _bm25_cache[key]
    with _bm25_cache_lock:
        if key not in _bm25_cache:
            if len(_bm25_cache) >= _BM25_CACHE_MAXSIZE:
                _bm25_cache.pop(next(iter(_bm25_cache)))
            try:
                rebuilt = _rebuild_bm25_index(settings, generation)
            except ChromaReadError:
                # A transient failure must not be cached -- it's not real
                # data for this generation, just degraded service for
                # this one call. Caching it would turn one Chroma hiccup
                # into "BM25 off for the rest of this process's life at
                # this generation."
                if raise_on_read_error:
                    raise
                return None, []
            _bm25_cache[key] = rebuilt
        return _bm25_cache[key]


def _rebuild_bm25_index(settings: Settings, generation: int) -> tuple[bm25s.BM25 | None, list[Document]]:
    """Cache key embeds the index generation -- a write from this process
    or another (including a same-chunk-count replace, which chunk count
    alone wouldn't catch) changes the key, so a stale entry is simply
    never looked up again, with no explicit invalidation step. generation
    is passed in rather than re-read here, so one get_bm25_index() call
    reads it exactly once, not twice."""
    client = get_cache_client(settings)
    key = make_cache_key("bm25", get_collection_name(settings), generation)

    # pickle is safe here: this cache is exclusively written and read by
    # this service itself, never by an untrusted party.
    cached = cache_get(client, key)
    if cached is not None:
        try:
            return pickle.loads(cached)
        except Exception:
            # A corrupted entry, an incompatible bm25s version, or --
            # observed live when this branch swapped rank_bm25 for bm25s --
            # a pickle referencing a module no longer installed at all
            # (ModuleNotFoundError, not UnpicklingError/EOFError/
            # AttributeError). Any deserialization failure degrades to a
            # rebuild rather than crashing the caller; caching is an
            # optimization here, not a hard dependency.
            logger.warning("Cached BM25 index was unusable; rebuilding.", exc_info=True)

    try:
        chunks = get_all_chunks(settings)
    except Exception as exc:
        logger.warning("Reading chunks from Chroma for the BM25 index failed; degrading to no BM25 results.", exc_info=True)
        raise ChromaReadError from exc
    # bm25s.BM25().index([]) raises ValueError on an empty corpus (nothing
    # ingested yet) -- unlike rank_bm25, which only raised on scoring one.
    if chunks:
        bm25 = bm25s.BM25()
        bm25.index([_tokenize(chunk.page_content) for chunk in chunks], show_progress=False)
    else:
        bm25 = None
    cache_set(client, key, pickle.dumps((bm25, chunks)), ttl_seconds=BM25_CACHE_TTL_SECONDS)
    return bm25, chunks


def score(query: str, settings: Settings, k: int) -> list[tuple[Document, float]]:
    bm25, chunks = get_bm25_index(settings, raise_on_read_error=True)
    if bm25 is None or not chunks:
        return []
    # bm25s.BM25.retrieve() raises ValueError if k exceeds the corpus size.
    k = min(k, len(chunks))
    doc_indices, scores = bm25.retrieve([_tokenize(query)], k=k, show_progress=False)
    return [(chunks[idx], float(s)) for idx, s in zip(doc_indices[0], scores[0])]
