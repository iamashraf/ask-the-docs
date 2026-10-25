from research_assistant.answer import AnswerResult
from research_assistant.cli import build_parser
from research_assistant.config import ProviderConfig, Settings
from research_assistant.evaluate import EvalResult, JudgeScore


def make_settings(**overrides) -> Settings:
    fields = dict(
        generator=ProviderConfig(provider="ollama", model="llama3.1:8b"),
        judge=ProviderConfig(provider="anthropic", model="claude-sonnet-5-5", api_key="x"),
    )
    fields.update(overrides)
    return Settings(**fields)


def test_ingest_dispatches_each_source_and_builds_index(monkeypatch):
    docs_by_source = {"a.pdf": ["doc-a"], "b.pdf": ["doc-b"]}
    monkeypatch.setattr("research_assistant.cli.dispatch", lambda source: docs_by_source[source])
    captured = {}
    monkeypatch.setattr(
        "research_assistant.cli.build_index", lambda docs, settings: captured.setdefault("docs", docs)
    )

    parser = build_parser()
    args = parser.parse_args(["ingest", "a.pdf", "b.pdf"])
    args.func(args, make_settings())

    assert captured["docs"] == ["doc-a", "doc-b"]


def test_index_discovers_docs_dir_and_builds_index(tmp_path, monkeypatch):
    settings = make_settings(docs_dir=tmp_path)
    monkeypatch.setattr(
        "research_assistant.cli.discover_sources",
        lambda docs_dir: [tmp_path / "a.pdf", tmp_path / "b.pdf"],
    )
    monkeypatch.setattr("research_assistant.cli.dispatch", lambda source: [f"doc-for-{source}"])
    captured = {}
    monkeypatch.setattr(
        "research_assistant.cli.build_index", lambda docs, settings: captured.setdefault("docs", docs)
    )

    parser = build_parser()
    args = parser.parse_args(["index"])
    args.func(args, settings)

    assert captured["docs"] == ["doc-for-a.pdf", "doc-for-b.pdf"]


def test_ask_prints_answer_and_citations(capsys, monkeypatch):
    monkeypatch.setattr(
        "research_assistant.cli.answer",
        lambda query, settings, k: AnswerResult(text="42", citations=["a.pdf"], context="ctx"),
    )

    parser = build_parser()
    args = parser.parse_args(["ask", "What is the answer?"])
    args.func(args, make_settings())

    out = capsys.readouterr().out
    assert "42" in out
    assert "a.pdf" in out


def test_evaluate_overrides_embeddings_backend_and_prints_results(capsys, monkeypatch):
    captured_settings = {}

    def fake_run_evaluation(settings, questions_path):
        captured_settings["backend"] = settings.embeddings_backend
        return [
            EvalResult(
                question="Q1",
                score=JudgeScore(
                    accuracy=1.0,
                    answer_relevance=1.0,
                    context_relevance=1.0,
                    completeness=1.0,
                    justification="ok",
                ),
                error=None,
            )
        ]

    monkeypatch.setattr("research_assistant.cli.run_evaluation", fake_run_evaluation)

    parser = build_parser()
    args = parser.parse_args(["evaluate", "--embeddings", "huggingface"])
    args.func(args, make_settings())

    assert captured_settings["backend"] == "huggingface"
    assert "Q1" in capsys.readouterr().out


def test_serve_starts_uvicorn(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        "research_assistant.cli.uvicorn.run",
        lambda app, host, port, log_level: captured.update(
            app=app, host=host, port=port, log_level=log_level
        ),
    )

    parser = build_parser()
    args = parser.parse_args(["serve"])
    args.func(args, make_settings())

    from research_assistant.api import app as api_app

    assert captured["app"] is api_app
    assert captured["host"] == "0.0.0.0"
    assert captured["port"] == 8000


def test_evaluate_reports_errors(capsys, monkeypatch):
    monkeypatch.setattr(
        "research_assistant.cli.run_evaluation",
        lambda settings, questions_path: [EvalResult(question="Q1", score=None, error="boom")],
    )

    parser = build_parser()
    args = parser.parse_args(["evaluate"])
    args.func(args, make_settings())

    out = capsys.readouterr().out
    assert "Q1" in out
    assert "boom" in out
