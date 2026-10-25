"""Thin wrapper over a Valkey/Redis-protocol client, shared by the BM25
index cache and the answer cache -- the only two cache types in this
project, so this stays a plain module, not a Strategy/Factory pair."""

import hashlib
import json

import redis

from research_assistant.config import Settings


def get_cache_client(settings: Settings) -> redis.Redis:
    return redis.Redis.from_url(settings.cache_url)


def make_cache_key(*parts: object) -> str:
    """Content-addressed key: encoding the corpus chunk count as one of
    `parts` means a new ingest produces a new key automatically, instead
    of needing explicit cache-invalidation code."""
    canonical = json.dumps(parts, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()


def cache_get(client, key: str) -> bytes | None:
    return client.get(key)


def cache_set(client, key: str, value: bytes, ttl_seconds: int) -> None:
    client.set(key, value, ex=ttl_seconds)
