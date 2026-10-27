import logging

from neo4j.exceptions import ClientError, ServiceUnavailable

from research_assistant.community import run_community_detection, summarize_communities
from research_assistant.config import ProviderConfig, Settings
from research_assistant.llm import OllamaUnavailableError


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


def test_run_community_detection_projects_runs_leiden_and_drops_projection():
    driver = FakeDriver()
    run_community_detection(driver)
    queries = [q for q, p in driver.session_obj.queries]
    assert any("gds.graph.project" in q for q in queries)
    assert any("gds.leiden.write" in q for q in queries)
    assert any("gds.graph.drop" in q for q in queries)
    # order matters: project, then leiden, then drop
    project_idx = next(i for i, q in enumerate(queries) if "gds.graph.project" in q)
    leiden_idx = next(i for i, q in enumerate(queries) if "gds.leiden.write" in q)
    drop_idx = next(i for i, q in enumerate(queries) if "gds.graph.drop" in q)
    assert project_idx < leiden_idx < drop_idx


def test_run_community_detection_projects_relationships_as_undirected():
    """GDS Leiden refuses directed graphs, so the projection itself must be undirected."""
    driver = FakeDriver()
    run_community_detection(driver)
    project_query = next(q for q, p in driver.session_obj.queries if "gds.graph.project" in q)
    assert "UNDIRECTED" in project_query


def test_run_community_detection_degrades_on_neo4j_unavailable():
    class BrokenDriver:
        def session(self):
            raise ServiceUnavailable("down")

    run_community_detection(BrokenDriver())  # must not raise


def test_run_community_detection_degrades_when_session_run_raises(caplog):
    class RaisingRunSession:
        def run(self, query, **params):
            raise ServiceUnavailable("gds plugin unavailable")

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    class DriverWithRaisingSession:
        def session(self):
            return RaisingRunSession()

    with caplog.at_level(logging.WARNING):
        run_community_detection(DriverWithRaisingSession())  # must not raise

    assert any(record.levelname == "WARNING" for record in caplog.records)


def test_run_community_detection_drops_projection_even_when_leiden_fails():
    """run() itself raising (e.g. connection drop mid-call) must still trigger the drop."""

    class SessionLeidenFails:
        def __init__(self):
            self.queries = []

        def run(self, query, **params):
            self.queries.append((query, params))
            if "gds.leiden.write" in query:
                raise ServiceUnavailable("leiden crashed mid-run")
            return FakeResult()

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    class DriverLeidenFails:
        def __init__(self):
            self.session_obj = SessionLeidenFails()

        def session(self):
            return self.session_obj

    driver = DriverLeidenFails()
    run_community_detection(driver)  # must not raise

    queries = [q for q, p in driver.session_obj.queries]
    assert any("gds.graph.project" in q for q in queries)
    assert any("gds.leiden.write" in q for q in queries)
    assert any("gds.graph.drop" in q for q in queries)


def test_run_community_detection_drops_projection_even_when_leiden_consume_fails(caplog):
    """Real neo4j driver defers execution errors to .consume(), not to run() itself —
    this proves we consume() each result instead of letting the error surface one
    call later (e.g. inside the drop call, or not at all)."""

    class SessionLeidenConsumeFails:
        def __init__(self):
            self.queries = []

        def run(self, query, **params):
            self.queries.append((query, params))
            if "gds.leiden.write" in query:
                return FakeResult(raise_on_consume=ServiceUnavailable("leiden crashed on consume"))
            return FakeResult()

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    class DriverLeidenConsumeFails:
        def __init__(self):
            self.session_obj = SessionLeidenConsumeFails()

        def session(self):
            return self.session_obj

    driver = DriverLeidenConsumeFails()
    with caplog.at_level(logging.WARNING):
        run_community_detection(driver)  # must not raise

    queries = [q for q, p in driver.session_obj.queries]
    assert any("gds.graph.project" in q for q in queries)
    assert any("gds.leiden.write" in q for q in queries)
    assert any("gds.graph.drop" in q for q in queries)
    assert any(record.levelname == "WARNING" for record in caplog.records)


def test_run_community_detection_does_not_drop_when_project_itself_fails():
    class SessionProjectFails:
        def __init__(self):
            self.queries = []

        def run(self, query, **params):
            self.queries.append((query, params))
            if "gds.graph.project" in query:
                raise ServiceUnavailable("project failed")
            return FakeResult()

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    class DriverProjectFails:
        def __init__(self):
            self.session_obj = SessionProjectFails()

        def session(self):
            return self.session_obj

    driver = DriverProjectFails()
    run_community_detection(driver)  # must not raise

    queries = [q for q, p in driver.session_obj.queries]
    assert any("gds.graph.project" in q for q in queries)
    assert not any("gds.leiden.write" in q for q in queries)
    assert not any("gds.graph.drop" in q for q in queries)


def make_settings() -> Settings:
    return Settings(
        generator=ProviderConfig(provider="ollama", model="llama3.1:8b"),
        judge=ProviderConfig(provider="anthropic", model="claude-sonnet-5-5", api_key="x"),
    )


class FakeReadResult:
    """Mimics neo4j.Result for a read query: iterable rows, .consume() possibly raises."""

    def __init__(self, rows, raise_on_consume=None):
        self._rows = rows
        self._raise_on_consume = raise_on_consume

    def __iter__(self):
        return iter(self._rows)

    def consume(self):
        if self._raise_on_consume is not None:
            raise self._raise_on_consume


class FakeClient:
    def __init__(self, response):
        self._response = response
        self.prompts = []

    def generate(self, prompt, system=None):
        self.prompts.append(prompt)
        return self._response


class FakeSessionWithCommunities:
    def __init__(self, communities):
        self.queries = []
        self._communities = communities

    def run(self, query, **params):
        self.queries.append((query, params))
        if "RETURN" in query:
            rows = [{"id": cid, "names": names} for cid, names in self._communities.items()]
            return FakeReadResult(rows)
        return FakeResult()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class FakeDriverWithCommunities:
    def __init__(self, communities):
        self.session_obj = FakeSessionWithCommunities(communities)

    def session(self):
        return self.session_obj


def test_summarize_communities_writes_summary_per_community(monkeypatch):
    driver = FakeDriverWithCommunities({0: ["Transformer", "attention"], 1: ["BM25", "TF-IDF"]})
    fake_client = FakeClient("a summary")
    monkeypatch.setattr("research_assistant.community.get_llm_client", lambda cfg: fake_client)

    summarize_communities(driver, make_settings())

    written = [(q, p) for q, p in driver.session_obj.queries if "Community" in q and "summary" in p]
    assert len(written) == 2
    assert {p["id"] for _, p in written} == {0, 1}
    assert all(p["summary"] == "a summary" for _, p in written)
    assert any("Transformer" in prompt for prompt in fake_client.prompts)
    assert any("BM25" in prompt for prompt in fake_client.prompts)


def test_summarize_communities_degrades_when_read_unreachable(caplog):
    class BrokenDriver:
        def session(self):
            raise ServiceUnavailable("down")

    with caplog.at_level(logging.WARNING):
        summarize_communities(BrokenDriver(), make_settings())  # must not raise

    assert any(record.levelname == "WARNING" for record in caplog.records)


def test_summarize_communities_degrades_when_write_consume_fails(monkeypatch, caplog):
    """Real neo4j driver defers execution errors to .consume(), not to run() itself —
    this proves we consume() the write result instead of letting the error pass unnoticed."""

    class SessionWriteConsumeFails:
        def __init__(self):
            self.queries = []

        def run(self, query, **params):
            self.queries.append((query, params))
            if "RETURN" in query:
                return FakeReadResult([{"id": 0, "names": ["Transformer"]}])
            if "MERGE" in query:
                return FakeResult(raise_on_consume=ServiceUnavailable("write failed on consume"))
            return FakeResult()

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    class DriverWriteConsumeFails:
        def __init__(self):
            self.session_obj = SessionWriteConsumeFails()

        def session(self):
            return self.session_obj

    monkeypatch.setattr("research_assistant.community.get_llm_client", lambda cfg: FakeClient("a summary"))

    driver = DriverWriteConsumeFails()
    with caplog.at_level(logging.WARNING):
        summarize_communities(driver, make_settings())  # must not raise

    queries = [q for q, _ in driver.session_obj.queries]
    assert any("RETURN" in q for q in queries)
    assert any("MERGE" in q and "Community" in q for q in queries)
    assert any(record.levelname == "WARNING" for record in caplog.records)


def test_summarize_communities_continues_past_llm_failure_for_one_community(monkeypatch, caplog):
    """A transient LLM failure on one community must not abort the rest, and must not
    undo summaries already written for communities processed before it."""

    driver = FakeDriverWithCommunities({0: ["Transformer"], 1: ["BM25"], 2: ["Neo4j"]})

    class FlakyClient:
        def generate(self, prompt, system=None):
            if "BM25" in prompt:
                raise OllamaUnavailableError("down")
            return "a summary"

    monkeypatch.setattr("research_assistant.community.get_llm_client", lambda cfg: FlakyClient())

    with caplog.at_level(logging.WARNING):
        summarize_communities(driver, make_settings())  # must not raise

    written = [(q, p) for q, p in driver.session_obj.queries if "Community" in q and "summary" in p]
    assert {p["id"] for _, p in written} == {0, 2}
    assert any(record.levelname == "WARNING" for record in caplog.records)


def test_run_community_detection_degrades_on_neo4j_client_error(caplog):
    """Exercises the Neo4jError half of the except tuple (e.g. missing GDS plugin)."""

    class SessionGdsMissing:
        def run(self, query, **params):
            raise ClientError("There is no procedure with the name `gds.graph.project`")

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    class DriverGdsMissing:
        def session(self):
            return SessionGdsMissing()

    with caplog.at_level(logging.WARNING):
        run_community_detection(DriverGdsMissing())  # must not raise

    assert any(record.levelname == "WARNING" for record in caplog.records)


def test_summarize_communities_deletes_previous_communities_before_writing(monkeypatch):
    """Leiden IDs aren't stable across runs, so a previous run's communities must not linger."""
    driver = FakeDriverWithCommunities({0: ["Transformer"]})
    monkeypatch.setattr("research_assistant.community.get_llm_client", lambda cfg: FakeClient("a summary"))

    summarize_communities(driver, make_settings())

    queries = [q for q, _ in driver.session_obj.queries]
    delete_idx = next(i for i, q in enumerate(queries) if "MATCH (c:Community) DETACH DELETE c" in q)
    first_write_idx = next(i for i, q in enumerate(queries) if "MERGE (c:Community" in q)
    assert delete_idx < first_write_idx
