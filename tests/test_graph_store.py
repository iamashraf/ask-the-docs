import logging
import re

from neo4j.exceptions import ServiceUnavailable

from research_assistant.graph_extract import Triplet
from research_assistant.graph_store import write_graph


class FakeSession:
    def __init__(self):
        self.queries = []

    def run(self, query, **params):
        self.queries.append((query, params))

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
