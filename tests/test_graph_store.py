import logging
import re

from neo4j.exceptions import ServiceUnavailable

from ask_the_docs.graph_extract import Triplet
from ask_the_docs.graph_store import get_entity_names, graph_supports_incremental, remove_sources, write_graph


class FakeResult:
    """Mimics neo4j.Result: errors from query execution surface on .consume()."""

    def __init__(self, raise_on_consume=None):
        self._raise_on_consume = raise_on_consume

    def consume(self):
        if self._raise_on_consume is not None:
            raise self._raise_on_consume


class FakeSession:
    def __init__(self):
        self.queries = []

    def run(self, query, **params):
        self.queries.append((query, params))
        return FakeResult()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class FakeDriver:
    def __init__(self):
        self.session_obj = FakeSession()

    def session(self):
        return self.session_obj


def test_write_graph_merges_entities_and_relationship():
    driver = FakeDriver()
    write_graph(driver, [(Triplet(subject="A", relation="uses", obj="B"), "doc1:0")])
    merge_queries = [q for q, p in driver.session_obj.queries if "MERGE" in q]
    assert len(merge_queries) >= 1

    query, params = driver.session_obj.queries[-1]
    assert params == {"subject": "A", "relation": "uses", "object": "B", "chunk_ref": "doc1:0"}
    # Catches a param-name/placeholder mismatch (e.g. passing obj= instead of
    # object=) that a real Neo4j driver would reject with a missing-parameter
    # error, but that a substring check on the query text wouldn't catch.
    placeholders = set(re.findall(r"\$(\w+)", query))
    assert placeholders == set(params.keys())


def test_write_graph_writes_each_triplet_with_its_own_distinct_params():
    driver = FakeDriver()
    chunk_triplets = [
        (Triplet(subject="A", relation="uses", obj="B"), "doc1:0"),
        (Triplet(subject="C", relation="implements", obj="D"), "doc2:3"),
    ]
    write_graph(driver, chunk_triplets)

    assert len(driver.session_obj.queries) == 2
    first_params = driver.session_obj.queries[0][1]
    second_params = driver.session_obj.queries[1][1]
    assert first_params == {"subject": "A", "relation": "uses", "object": "B", "chunk_ref": "doc1:0"}
    assert second_params == {
        "subject": "C",
        "relation": "implements",
        "object": "D",
        "chunk_ref": "doc2:3",
    }


def test_write_graph_is_a_noop_for_empty_triplet_list():
    driver = FakeDriver()
    write_graph(driver, [])  # must not raise
    assert driver.session_obj.queries == []


def test_write_graph_degrades_on_neo4j_unavailable(caplog):
    class BrokenDriver:
        def session(self):
            raise ServiceUnavailable("down")

    with caplog.at_level(logging.WARNING):
        write_graph(BrokenDriver(), [(Triplet(subject="A", relation="uses", obj="B"), "doc1:0")])  # must not raise

    assert any(record.levelname == "WARNING" for record in caplog.records)


def test_write_graph_degrades_when_session_run_raises(caplog):
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

    with caplog.at_level(logging.WARNING):
        write_graph(
            DriverWithRaisingSession(), [(Triplet(subject="A", relation="uses", obj="B"), "doc1:0")]
        )  # must not raise

    assert any(record.levelname == "WARNING" for record in caplog.records)


def test_write_graph_warns_when_last_triplets_consume_fails(caplog):
    """Real neo4j driver defers execution errors to .consume(), not to run() itself.
    If write_graph doesn't call .consume() on each result, a failure on the LAST
    triplet in a batch has no later run() call to surface it on, so it would be
    dropped silently -- no exception, no warning. This proves we consume() each
    result so that failure is caught and logged instead."""

    class SessionLastConsumeFails:
        def __init__(self):
            self.queries = []

        def run(self, query, **params):
            self.queries.append((query, params))
            if params.get("subject") == "C":
                return FakeResult(raise_on_consume=ServiceUnavailable("write lost on consume"))
            return FakeResult()

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    class DriverLastConsumeFails:
        def __init__(self):
            self.session_obj = SessionLastConsumeFails()

        def session(self):
            return self.session_obj

    driver = DriverLastConsumeFails()
    chunk_triplets = [
        (Triplet(subject="A", relation="uses", obj="B"), "doc1:0"),
        (Triplet(subject="C", relation="implements", obj="D"), "doc2:3"),
    ]
    with caplog.at_level(logging.WARNING):
        write_graph(driver, chunk_triplets)  # must not raise

    assert len(driver.session_obj.queries) == 2
    assert any(record.levelname == "WARNING" for record in caplog.records)


def test_clear_graph_deletes_all_entity_and_community_nodes_with_relationships():
    from ask_the_docs.graph_store import clear_graph

    driver = FakeDriver()

    clear_graph(driver)

    (query, _), = driver.session_obj.queries
    assert "DETACH DELETE" in query
    assert "Entity" in query and "Community" in query


def test_write_graph_returns_none_on_success_and_the_exception_class_name_on_failure():
    assert write_graph(FakeDriver(), [(Triplet(subject="A", relation="uses", obj="B"), "doc1:0")]) is None

    class BrokenDriver:
        def session(self):
            raise ServiceUnavailable("secret-host:7687 down")

    assert write_graph(BrokenDriver(), [(Triplet(subject="A", relation="uses", obj="B"), "doc1:0")]) == "ServiceUnavailable"


def test_write_graph_records_the_chunk_ref_on_the_relationship_too():
    driver = FakeDriver()
    write_graph(driver, [(Triplet(subject="A", relation="uses", obj="B"), "doc1:0")])
    query, _ = driver.session_obj.queries[-1]
    assert re.search(r"MERGE \(s\)-\[r:RELATED_TO \{relation: \$relation\}\]->\(o\)", query)
    assert "r.chunk_refs" in query


def test_remove_sources_strips_refs_from_edges_then_entities_and_deletes_the_empty_ones():
    driver = FakeDriver()
    remove_sources(driver, {"docA", "docB"})
    queries = driver.session_obj.queries
    assert len(queries) == 2
    edge_query, entity_query = queries[0][0], queries[1][0]
    assert "RELATED_TO" in edge_query and "DELETE r" in edge_query and "size(r.chunk_refs) = 0" in edge_query
    assert "Entity" in entity_query and "DETACH DELETE e" in entity_query
    for _, params in queries:
        assert sorted(params["sources"]) == ["docA", "docB"]


def test_remove_sources_matches_whole_source_ids_not_mere_prefixes():
    driver = FakeDriver()
    remove_sources(driver, {"doc"})
    for query, _ in driver.session_obj.queries:
        assert "STARTS WITH src + ':'" in query and "CONTAINS ':'" in query


def test_remove_sources_is_a_noop_for_no_sources():
    driver = FakeDriver()
    remove_sources(driver, set())
    assert driver.session_obj.queries == []


class FakeReadResult:
    def __init__(self, rows):
        self._rows = rows

    def single(self):
        return self._rows[0]

    def __iter__(self):
        return iter(self._rows)

    def consume(self):
        pass


class FakeReadSession(FakeSession):
    def __init__(self, entities, legacy_edges, names=()):
        super().__init__()
        self._entities, self._legacy, self._names = entities, legacy_edges, names

    def run(self, query, **params):
        self.queries.append((query, params))
        if "r.chunk_refs IS NULL" in query:
            return FakeReadResult([{"n": self._legacy}])
        if "count(e)" in query:
            return FakeReadResult([{"n": self._entities}])
        return FakeReadResult([{"name": n} for n in self._names])


class FakeReadDriver:
    def __init__(self, *args, **kwargs):
        self.session_obj = FakeReadSession(*args, **kwargs)

    def session(self):
        return self.session_obj


def test_graph_supports_incremental_only_for_a_non_empty_graph_with_ref_tagged_edges():
    assert graph_supports_incremental(FakeReadDriver(entities=3, legacy_edges=0)) is True
    assert graph_supports_incremental(FakeReadDriver(entities=0, legacy_edges=0)) is False
    assert graph_supports_incremental(FakeReadDriver(entities=3, legacy_edges=2)) is False


def test_get_entity_names_returns_sorted_names():
    driver = FakeReadDriver(entities=2, legacy_edges=0, names=["b", "a"])
    assert get_entity_names(driver) == ["a", "b"]
