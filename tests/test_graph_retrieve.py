from langchain_core.documents import Document
from neo4j.exceptions import ServiceUnavailable

from research_assistant.config import ProviderConfig, Settings
from research_assistant.graph_retrieve import local_search


def make_settings(**overrides) -> Settings:
    return Settings(
        generator=ProviderConfig(provider="ollama", model="llama3.1:8b"),
        judge=ProviderConfig(provider="anthropic", model="claude-sonnet-5-5", api_key="x"),
        **overrides,
    )


def make_chunk(source_doc_id: str, chunk_index: int, text: str) -> Document:
    return Document(page_content=text, metadata={"source_doc_id": source_doc_id, "chunk_index": chunk_index})


class FakeReadResult:
    """Mimics neo4j.Result for a read query: iterable rows, .consume() is a no-op here."""

    def __init__(self, rows):
        self._rows = rows

    def __iter__(self):
        return iter(self._rows)

    def consume(self):
        pass


class FakeSessionReturningChunkRefs:
    def __init__(self, chunk_refs):
        self._chunk_refs = chunk_refs
        self.queries = []

    def run(self, query, **params):
        self.queries.append((query, params))
        return FakeReadResult([{"chunk_refs": self._chunk_refs}])

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class FakeDriverReturningChunkRefs:
    def __init__(self, chunk_refs):
        self.session_obj = FakeSessionReturningChunkRefs(chunk_refs)

    def session(self):
        return self.session_obj


def test_local_search_returns_matching_chunks(monkeypatch):
    driver = FakeDriverReturningChunkRefs(["doc1:0", "doc2:1"])
    monkeypatch.setattr("research_assistant.graph_retrieve.get_graph_client", lambda s: driver)
    monkeypatch.setattr(
        "research_assistant.graph_retrieve.get_all_chunks",
        lambda s: [
            make_chunk("doc1", 0, "about BM25"),
            make_chunk("doc2", 1, "about vectors"),
            make_chunk("doc3", 0, "unrelated"),
        ],
    )

    result = local_search(["BM25"], make_settings())

    assert {(d.metadata["source_doc_id"], d.metadata["chunk_index"]) for d in result} == {
        ("doc1", 0),
        ("doc2", 1),
    }


def test_local_search_returns_empty_list_for_no_entities():
    assert local_search([], make_settings()) == []


def test_local_search_degrades_on_neo4j_unavailable(monkeypatch):
    def boom(s):
        raise ServiceUnavailable("down")

    monkeypatch.setattr("research_assistant.graph_retrieve.get_graph_client", boom)

    assert local_search(["BM25"], make_settings()) == []


def test_local_search_skips_stale_chunk_refs_not_in_index(monkeypatch):
    """Re-ingestion doesn't clear old graph data, so a chunk_ref can point at
    content no longer in the index -- local_search must skip it, not KeyError."""
    driver = FakeDriverReturningChunkRefs(["doc1:0", "doc_deleted:5"])
    monkeypatch.setattr("research_assistant.graph_retrieve.get_graph_client", lambda s: driver)
    monkeypatch.setattr(
        "research_assistant.graph_retrieve.get_all_chunks",
        lambda s: [make_chunk("doc1", 0, "about BM25")],
    )

    result = local_search(["BM25"], make_settings())

    assert {(d.metadata["source_doc_id"], d.metadata["chunk_index"]) for d in result} == {("doc1", 0)}
