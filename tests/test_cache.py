import redis

from ask_the_docs.cache import cache_get, cache_set, get_cache_client, make_cache_key
from ask_the_docs.config import ProviderConfig, Settings


class FakeRedis:
    def __init__(self):
        self._store: dict[str, bytes] = {}

    def set(self, key, value, ex=None):
        self._store[key] = value

    def get(self, key):
        return self._store.get(key)


def test_make_cache_key_is_order_independent_and_deterministic():
    a = make_cache_key("q", 5, 0.5)
    b = make_cache_key("q", 5, 0.5)
    assert a == b


def test_make_cache_key_differs_for_different_input():
    assert make_cache_key("q", 5) != make_cache_key("q", 6)


def test_cache_get_set_roundtrip():
    fake_client = FakeRedis()
    cache_set(fake_client, "k", b"v", ttl_seconds=60)
    assert cache_get(fake_client, "k") == b"v"


def test_cache_get_returns_none_on_miss():
    fake_client = FakeRedis()
    assert cache_get(fake_client, "missing") is None


def test_cache_get_returns_none_when_client_raises_redis_error():
    class BrokenClient:
        def get(self, key):
            raise redis.RedisError("connection refused")

    assert cache_get(BrokenClient(), "k") is None


def test_cache_set_is_a_noop_when_client_raises_redis_error():
    class BrokenClient:
        def set(self, key, value, ex=None):
            raise redis.RedisError("connection refused")

    cache_set(BrokenClient(), "k", b"v", ttl_seconds=60)  # must not raise


def test_cache_set_passes_ttl_seconds_to_client():
    captured = {}

    class CapturingClient:
        def set(self, key, value, ex=None):
            captured["ex"] = ex

    cache_set(CapturingClient(), "k", b"v", ttl_seconds=123)

    assert captured["ex"] == 123


def test_get_cache_client_builds_from_settings_url(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        redis.Redis, "from_url", classmethod(lambda cls, url: captured.setdefault("url", url) or cls())
    )
    settings = Settings(
        generator=ProviderConfig(provider="ollama", model="llama3.1:8b"),
        judge=ProviderConfig(provider="anthropic", model="claude-sonnet-5-5", api_key="x"),
        cache_url="redis://example:6379/0",
    )
    get_cache_client(settings)
    assert captured["url"] == "redis://example:6379/0"
