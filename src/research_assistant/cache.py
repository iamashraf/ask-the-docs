"""Thin wrapper over a Valkey/Redis-protocol client, shared by the BM25
index cache and the answer cache -- the only two cache types in this
project, so this stays a plain module, not a Strategy/Factory pair."""

import hashlib
import json
import logging

import redis

from research_assistant.config import Settings

logger = logging.getLogger(__name__)


def get_cache_client(settings: Settings) -> redis.Redis:
    return redis.Redis.from_url(settings.cache_url)


def make_cache_key(*parts: object) -> str:
    """Content-addressed key: encoding the corpus chunk count as one of
    `parts` means a new ingest produces a new key automatically, instead
    of needing explicit cache-invalidation code."""
    canonical = json.dumps(parts, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()


def cache_get(client, key: str) -> bytes | None:
    """Caching is an optimization, not a hard dependency -- same reasoning
    the pipeline already applies to query rewrite (falls back on failure
    rather than taking the whole request down with it)."""
    try:
        return client.get(key)
    except redis.RedisError:
        logger.warning("Cache read failed; treating as a miss.", exc_info=True)
        return None


def cache_set(client, key: str, value: bytes, ttl_seconds: int) -> None:
    try:
        client.set(key, value, ex=ttl_seconds)
    except redis.RedisError:
        logger.warning("Cache write failed; continuing without caching this result.", exc_info=True)
