from types import SimpleNamespace

from langchain_core.documents import Document

from research_assistant.seed import load_seed_sources, seed_if_needed, seed_marker_path


def make_settings(tmp_path):
    return SimpleNamespace(index_dir=tmp_path / "index")


def test_load_seed_sources_parses_yaml(tmp_path):
    path = tmp_path / "seed_sources.yaml"
    path.write_text("- https://example.com/a\n- https://example.com/b\n")

    assert load_seed_sources(path) == ["https://example.com/a", "https://example.com/b"]


def test_seed_if_needed_skips_when_marker_exists(tmp_path, monkeypatch):
    settings = make_settings(tmp_path)
    settings.index_dir.mkdir(parents=True)
    seed_marker_path(settings).write_text("seeded\n")

    called = []
    monkeypatch.setattr("research_assistant.seed.dispatch", lambda source: called.append(source))

    count = seed_if_needed(settings, seed_sources_path=tmp_path / "unused.yaml")

    assert count == 0
    assert called == []


def test_seed_if_needed_ingests_all_sources_and_writes_marker(tmp_path, monkeypatch):
    settings = make_settings(tmp_path)
    seed_path = tmp_path / "seed_sources.yaml"
    seed_path.write_text("- a\n- b\n")

    monkeypatch.setattr(
        "research_assistant.seed.dispatch",
        lambda source: [Document(page_content=source, metadata={"source": source})],
    )
    captured = {}
    monkeypatch.setattr(
        "research_assistant.seed.build_index", lambda docs, s: captured.setdefault("docs", docs)
    )

    count = seed_if_needed(settings, seed_sources_path=seed_path)

    assert count == 2
    assert len(captured["docs"]) == 2
    assert seed_marker_path(settings).exists()


def test_seed_if_needed_continues_past_a_failing_source_and_still_writes_marker(tmp_path, monkeypatch):
    settings = make_settings(tmp_path)
    seed_path = tmp_path / "seed_sources.yaml"
    seed_path.write_text("- good\n- bad\n")

    def fake_dispatch(source):
        if source == "bad":
            raise ValueError("network error")
        return [Document(page_content=source, metadata={"source": source})]

    monkeypatch.setattr("research_assistant.seed.dispatch", fake_dispatch)
    monkeypatch.setattr("research_assistant.seed.build_index", lambda docs, s: None)

    count = seed_if_needed(settings, seed_sources_path=seed_path)

    assert count == 1
    assert seed_marker_path(settings).exists()
