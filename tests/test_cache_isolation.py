from ask_the_docs import bm25_index, index


def test_a_populates_process_local_caches():
    index._index_cache[("leak", 0, "x")] = object()
    bm25_index._bm25_cache[("leak", 0, "x")] = (None, [])


def test_b_sees_empty_process_local_caches():
    """Relies on running after test_a; fails without the conftest clearing fixture."""
    assert index._index_cache == {}
    assert bm25_index._bm25_cache == {}
