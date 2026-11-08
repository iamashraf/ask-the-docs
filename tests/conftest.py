import pytest

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
