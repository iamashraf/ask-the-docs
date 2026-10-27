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
    assert driver.session_obj.queries[-1][1]["chunk_ref"] == "doc1:0"


def test_write_graph_degrades_on_neo4j_unavailable():
    class BrokenDriver:
        def session(self):
            raise ServiceUnavailable("down")

    write_graph(BrokenDriver(), [(Triplet(subject="A", relation="uses", obj="B"), "doc1:0")])  # must not raise
