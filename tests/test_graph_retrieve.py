import logging

from langchain_core.documents import Document
from neo4j.exceptions import ClientError, ServiceUnavailable, SessionExpired

from research_assistant.config import ProviderConfig, Settings
from research_assistant.graph_retrieve import global_search, local_search


def make_settings(**overrides) -> Settings:
    return Settings(
        generator=ProviderConfig(provider="ollama", model="llama3.1:8b"),
        judge=ProviderConfig(provider="anthropic", model="claude-sonnet-5-5", api_key="x"),
        **overrides,
    )


settings = make_settings()


def make_chunk(source_doc_id: str, chunk_index: int, text: str) -> Document:
    return Document(page_content=text, metadata={"source_doc_id": source_doc_id, "chunk_index": chunk_index})


def fake_chunks_by_refs(chunks: list[Document]):
    """Stands in for index.get_chunks_by_refs: returns only the chunks whose ref was requested."""

    def fetch(settings, refs):
        return [c for c in chunks if f"{c.metadata['source_doc_id']}:{c.metadata['chunk_index']}" in refs]

    return fetch


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


class FakeSessionReturningChunkRefRows:
    """Like FakeSessionReturningChunkRefs but models multiple graph rows, one per connected entity."""

    def __init__(self, rows):
        self._rows = rows
        self.queries = []

    def run(self, query, **params):
        self.queries.append((query, params))
        return FakeReadResult([{"chunk_refs": row} for row in self._rows])

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class FakeDriverReturningChunkRefRows:
    def __init__(self, rows):
        self.session_obj = FakeSessionReturningChunkRefRows(rows)

    def session(self):
        return self.session_obj


def test_local_search_returns_matching_chunks(monkeypatch):
    driver = FakeDriverReturningChunkRefs(["doc1:0", "doc2:1"])
    monkeypatch.setattr("research_assistant.graph_retrieve.get_graph_client", lambda s: driver)
    monkeypatch.setattr(
        "research_assistant.graph_retrieve.get_chunks_by_refs",
        fake_chunks_by_refs(
            [
                make_chunk("doc1", 0, "about BM25"),
                make_chunk("doc2", 1, "about vectors"),
                make_chunk("doc3", 0, "unrelated"),
            ]
        ),
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
    """Stale chunk_ref from re-ingestion (nothing clears old graph data) must be skipped, not KeyError."""
    driver = FakeDriverReturningChunkRefs(["doc1:0", "doc_deleted:5"])
    monkeypatch.setattr("research_assistant.graph_retrieve.get_graph_client", lambda s: driver)
    monkeypatch.setattr(
        "research_assistant.graph_retrieve.get_chunks_by_refs",
        fake_chunks_by_refs([make_chunk("doc1", 0, "about BM25")]),
    )

    result = local_search(["BM25"], make_settings())

    assert {(d.metadata["source_doc_id"], d.metadata["chunk_index"]) for d in result} == {("doc1", 0)}


def test_local_search_dedupes_chunk_refs_across_multiple_rows(monkeypatch):
    """graph_store writes the same chunk_ref onto both triplet endpoints, so overlapping rows are the realistic shape."""
    driver = FakeDriverReturningChunkRefRows([["doc1:0", "doc2:1"], ["doc2:1", "doc1:0"]])
    monkeypatch.setattr("research_assistant.graph_retrieve.get_graph_client", lambda s: driver)
    monkeypatch.setattr(
        "research_assistant.graph_retrieve.get_chunks_by_refs",
        fake_chunks_by_refs([make_chunk("doc1", 0, "about BM25"), make_chunk("doc2", 1, "about vectors")]),
    )

    result = local_search(["BM25"], make_settings())

    assert len(result) == 2
    assert {(d.metadata["source_doc_id"], d.metadata["chunk_index"]) for d in result} == {
        ("doc1", 0),
        ("doc2", 1),
    }


def test_local_search_degrades_when_session_run_raises_service_unavailable(monkeypatch, caplog):
    class RaisingRunSession:
        def run(self, query, **params):
            raise ServiceUnavailable("connection lost mid-transaction")

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    class DriverWithRaisingSession:
        def session(self):
            return RaisingRunSession()

    monkeypatch.setattr(
        "research_assistant.graph_retrieve.get_graph_client", lambda s: DriverWithRaisingSession()
    )

    with caplog.at_level(logging.WARNING):
        result = local_search(["BM25"], make_settings())  # must not raise

    assert result == []
    assert any(record.levelname == "WARNING" for record in caplog.records)


class FakeSessionReturningCommunities:
    def __init__(self, communities):
        self._communities = communities
        self.queries = []

    def run(self, query, **params):
        self.queries.append((query, params))
        return FakeReadResult([{"id": cid, "summary": summary} for cid, summary in self._communities])

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class FakeDriverReturningCommunities:
    def __init__(self, communities):
        self.session_obj = FakeSessionReturningCommunities(communities)

    def session(self):
        return self.session_obj


class FakeClient:
    """Returns one response per call, in order; a response that's an Exception instance is raised instead."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.prompts = []

    def generate(self, prompt, system=None):
        self.prompts.append(prompt)
        response = self._responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def test_global_search_returns_top_scored_community_summaries(monkeypatch):
    driver = FakeDriverReturningCommunities([(0, "summary about RAG"), (1, "summary about transformers")])
    monkeypatch.setattr("research_assistant.graph_retrieve.get_graph_client", lambda s: driver)
    monkeypatch.setattr(
        "research_assistant.graph_retrieve.get_llm_client",
        lambda cfg: FakeClient(responses=['{"relevance": 0.9}', '{"relevance": 0.2}']),
    )
    settings_top_1 = settings.model_copy(update={"global_search_top_communities": 1})

    result = global_search("what is RAG?", settings_top_1)

    assert len(result) == 1
    assert result[0].page_content == "summary about RAG"
    assert result[0].metadata["source"] == "community-0"
    assert result[0].metadata["source_doc_id"] == "community-0"
    assert result[0].metadata["chunk_index"] == 0


def test_global_search_returns_empty_list_when_no_communities(monkeypatch):
    monkeypatch.setattr(
        "research_assistant.graph_retrieve.get_graph_client", lambda s: FakeDriverReturningCommunities([])
    )

    assert global_search("anything", settings) == []


def test_global_search_degrades_on_neo4j_unavailable(monkeypatch):
    def boom(s):
        raise ServiceUnavailable("down")

    monkeypatch.setattr("research_assistant.graph_retrieve.get_graph_client", boom)

    assert global_search("anything", settings) == []


def test_global_search_scores_community_zero_when_llm_raises(monkeypatch):
    """One community's scoring call raising entirely (e.g. backend unreachable) must not abort the others."""
    from research_assistant.llm import OllamaUnavailableError

    driver = FakeDriverReturningCommunities([(0, "summary about RAG"), (1, "summary about transformers")])
    monkeypatch.setattr("research_assistant.graph_retrieve.get_graph_client", lambda s: driver)
    monkeypatch.setattr(
        "research_assistant.graph_retrieve.get_llm_client",
        lambda cfg: FakeClient(responses=[OllamaUnavailableError("down"), '{"relevance": 0.7}']),
    )
    settings_top_1 = settings.model_copy(update={"global_search_top_communities": 1})

    result = global_search("what is RAG?", settings_top_1)

    assert len(result) == 1
    assert result[0].page_content == "summary about transformers"


def test_global_search_parses_relevance_score_wrapped_in_markdown_fence(monkeypatch):
    driver = FakeDriverReturningCommunities([(0, "summary about RAG"), (1, "summary about transformers")])
    monkeypatch.setattr("research_assistant.graph_retrieve.get_graph_client", lambda s: driver)
    monkeypatch.setattr(
        "research_assistant.graph_retrieve.get_llm_client",
        lambda cfg: FakeClient(responses=['```json\n{"relevance": 0.9}\n```', '{"relevance": 0.1}']),
    )
    settings_top_1 = settings.model_copy(update={"global_search_top_communities": 1})

    result = global_search("what is RAG?", settings_top_1)

    assert len(result) == 1
    assert result[0].page_content == "summary about RAG"


def test_local_search_degrades_on_neo4j_client_error(monkeypatch, caplog):
    """Exercises the Neo4jError half of the except tuple."""

    class SessionRaisingClientError:
        def run(self, query, **params):
            raise ClientError("some client error")

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    class DriverRaisingClientError:
        def session(self):
            return SessionRaisingClientError()

    monkeypatch.setattr(
        "research_assistant.graph_retrieve.get_graph_client", lambda s: DriverRaisingClientError()
    )

    with caplog.at_level(logging.WARNING):
        result = local_search(["BM25"], make_settings())  # must not raise

    assert result == []
    assert any(record.levelname == "WARNING" for record in caplog.records)


class SessionRaisingSessionExpired:
    """SessionExpired is a DriverError, not a Neo4jError -- a mid-session connection drop."""

    def run(self, query, **params):
        raise SessionExpired("connection dropped mid-session")

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class DriverRaisingSessionExpired:
    def session(self):
        return SessionRaisingSessionExpired()


def test_local_search_degrades_on_driver_error_that_is_not_neo4j_error(monkeypatch, caplog):
    monkeypatch.setattr(
        "research_assistant.graph_retrieve.get_graph_client", lambda s: DriverRaisingSessionExpired()
    )

    with caplog.at_level(logging.WARNING):
        result = local_search(["BM25"], make_settings())  # must not raise

    assert result == []
    assert any(record.levelname == "WARNING" for record in caplog.records)


def test_global_search_degrades_on_driver_error_that_is_not_neo4j_error(monkeypatch, caplog):
    monkeypatch.setattr(
        "research_assistant.graph_retrieve.get_graph_client", lambda s: DriverRaisingSessionExpired()
    )

    with caplog.at_level(logging.WARNING):
        result = global_search("anything", make_settings())  # must not raise

    assert result == []
    assert any(record.levelname == "WARNING" for record in caplog.records)


def test_local_search_caps_resolved_refs_at_candidate_pool_size(monkeypatch):
    """A hub entity's multi-hop traversal can match far more chunks than the reranker should ever see."""
    refs = [f"doc{i}:0" for i in range(5)]
    driver = FakeDriverReturningChunkRefs(refs)
    monkeypatch.setattr("research_assistant.graph_retrieve.get_graph_client", lambda s: driver)
    requested = []

    def fake_fetch(settings, refs_to_fetch):
        requested.append(list(refs_to_fetch))
        return fake_chunks_by_refs([make_chunk(f"doc{i}", 0, f"chunk {i}") for i in range(5)])(
            settings, refs_to_fetch
        )

    monkeypatch.setattr("research_assistant.graph_retrieve.get_chunks_by_refs", fake_fetch, raising=False)

    result = local_search(["BM25"], make_settings(candidate_pool_size=2))

    assert len(requested) == 1
    assert len(requested[0]) == 2
    assert len(result) == 2


def test_local_search_does_not_fetch_the_whole_corpus(monkeypatch):
    driver = FakeDriverReturningChunkRefs(["doc1:0"])
    monkeypatch.setattr("research_assistant.graph_retrieve.get_graph_client", lambda s: driver)

    def full_fetch(settings):
        raise AssertionError("local_search must not load every chunk on each query")

    monkeypatch.setattr("research_assistant.graph_retrieve.get_all_chunks", full_fetch, raising=False)
    monkeypatch.setattr(
        "research_assistant.graph_retrieve.get_chunks_by_refs",
        fake_chunks_by_refs([make_chunk("doc1", 0, "about BM25")]),
        raising=False,
    )

    result = local_search(["BM25"], make_settings())

    assert [(d.metadata["source_doc_id"], d.metadata["chunk_index"]) for d in result] == [("doc1", 0)]
