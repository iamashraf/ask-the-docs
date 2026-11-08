import pytest
import redis

from ask_the_docs import bm25_index, index


@pytest.fixture(autouse=True)
def _clear_process_local_index_caches():
    """Stops cached Chroma/BM25 handles leaking between tests that reuse an index_dir."""

    def clear():
        with index._index_cache_lock:
            index._index_cache.clear()
        with bm25_index._bm25_cache_lock:
            bm25_index._bm25_cache.clear()

    clear()
    yield
    clear()


class FakeCache:
    def __init__(self):
        self.store = {}
        self.fail = False

    def get(self, key):
        if self.fail:
            raise redis.ConnectionError("down")
        return self.store.get(key)

    def set(self, key, value, ex=None):
        if self.fail:
            raise redis.ConnectionError("down")
        self.store[key] = value


@pytest.fixture(autouse=True)
def graph_cache(monkeypatch):
    """Keeps graph triplet/summary caching off the real Valkey and lets tests inspect it."""
    fake = FakeCache()
    for module in ("graph_extract", "community"):
        monkeypatch.setattr(f"ask_the_docs.{module}.get_cache_client", lambda settings: fake, raising=False)
    return fake
