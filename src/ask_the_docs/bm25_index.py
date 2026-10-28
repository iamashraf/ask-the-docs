"""Keyword (BM25) search over the full chunk collection, cached in Valkey
so neither the CLI's one-shot processes nor the long-running API server
rebuild it from scratch on every query."""

import logging
import pickle
import re

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from ask_the_docs.cache import cache_get, cache_set, get_cache_client, make_cache_key
from ask_the_docs.config import Settings
from ask_the_docs.index import get_all_chunks, get_chunk_count, get_collection_name

logger = logging.getLogger(__name__)

BM25_CACHE_TTL_SECONDS = 86400

_WORD_PATTERN = re.compile(r"\w+")


def _tokenize(text: str) -> list[str]:
    # Plain str.split() keeps punctuation attached ("lora:" != "lora",
    # "lora?" != "lora"), which defeats exact keyword/acronym matching --
    # the whole reason BM25 is in the pipeline alongside vector search.
    return _WORD_PATTERN.findall(text.lower())


def get_bm25_index(settings: Settings) -> tuple[BM25Okapi, list[Document]]:
    """Cache key embeds the live chunk count -- a new ingest (from this
    process or another) changes the key, so a stale entry is simply never
    looked up again, with no explicit invalidation step."""
    client = get_cache_client(settings)
    key = make_cache_key("bm25", get_collection_name(settings), get_chunk_count(settings))

    # pickle is safe here: this cache is exclusively written and read by
    # this service itself, never by an untrusted party.
    cached = cache_get(client, key)
    if cached is not None:
        try:
            return pickle.loads(cached)
        except (pickle.UnpicklingError, EOFError, AttributeError):
            # A corrupted entry, or one from an incompatible rank_bm25
            # version -- treat it as a miss and rebuild, same as the
            # answer cache's corruption handling.
            pass

    try:
        chunks = get_all_chunks(settings)
    except Exception:
        logger.warning("Reading chunks from Chroma for the BM25 index failed; degrading to no BM25 results.", exc_info=True)
        return None, []
    # BM25Okapi divides by corpus size internally -- an empty corpus (nothing
    # ingested yet) would otherwise raise ZeroDivisionError.
    bm25 = BM25Okapi([_tokenize(chunk.page_content) for chunk in chunks]) if chunks else None
    cache_set(client, key, pickle.dumps((bm25, chunks)), ttl_seconds=BM25_CACHE_TTL_SECONDS)
    return bm25, chunks


def score(query: str, settings: Settings) -> list[tuple[Document, float]]:
    bm25, chunks = get_bm25_index(settings)
    if bm25 is None:
        return []
    return list(zip(chunks, bm25.get_scores(_tokenize(query))))
