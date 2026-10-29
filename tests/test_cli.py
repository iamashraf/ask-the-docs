import pytest
from pydantic import ValidationError

from ask_the_docs.answer import AnswerResult
from ask_the_docs.cli import build_parser
from ask_the_docs.config import ProviderConfig, Settings
from ask_the_docs.evaluate import EvalResult, JudgeScore


def make_settings(**overrides) -> Settings:
    fields = dict(
        generator=ProviderConfig(provider="ollama", model="llama3.1:8b"),
        judge=ProviderConfig(provider="anthropic", model="claude-sonnet-5-5", api_key="x"),
    )
    fields.update(overrides)
    return Settings(**fields)


def test_ingest_dispatches_each_source_and_builds_index(monkeypatch):
    docs_by_source = {"a.pdf": ["doc-a"], "b.pdf": ["doc-b"]}
    monkeypatch.setattr("ask_the_docs.cli.dispatch", lambda source: docs_by_source[source])
    captured = {}
    monkeypatch.setattr(
        "ask_the_docs.cli.build_index", lambda docs, settings: captured.setdefault("docs", docs)
    )

    parser = build_parser()
    args = parser.parse_args(["ingest", "a.pdf", "b.pdf"])
    args.func(args, make_settings())

    assert captured["docs"] == ["doc-a", "doc-b"]


def test_index_discovers_docs_dir_and_builds_index(tmp_path, monkeypatch):
    settings = make_settings(docs_dir=tmp_path)
    monkeypatch.setattr(
        "ask_the_docs.cli.dispatch_all_under_docs_dir",
        lambda s: (["doc-for-a.pdf", "doc-for-b.pdf"], 2),
    )
    captured = {}
    monkeypatch.setattr(
        "ask_the_docs.cli.build_index", lambda docs, settings: captured.setdefault("docs", docs)
    )

    parser = build_parser()
    args = parser.parse_args(["index"])
    args.func(args, settings)

    assert captured["docs"] == ["doc-for-a.pdf", "doc-for-b.pdf"]


def test_ask_prints_answer_and_citations(capsys, monkeypatch):
    monkeypatch.setattr(
        "ask_the_docs.cli.answer",
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

    monkeypatch.setattr("ask_the_docs.cli.run_evaluation", fake_run_evaluation)

    parser = build_parser()
    args = parser.parse_args(["evaluate", "--embeddings", "huggingface"])
    args.func(args, make_settings())

    assert captured_settings["backend"] == "huggingface"
    assert "Q1" in capsys.readouterr().out


def test_evaluate_overrides_hybrid_alpha(monkeypatch):
    captured_settings = {}

    def fake_run_evaluation(settings, questions_path):
        captured_settings["alpha"] = settings.hybrid_alpha
        return []

    monkeypatch.setattr("ask_the_docs.cli.run_evaluation", fake_run_evaluation)

    parser = build_parser()
    args = parser.parse_args(["evaluate", "--hybrid-alpha", "0.8"])
    args.func(args, make_settings())

    assert captured_settings["alpha"] == 0.8


def test_evaluate_overrides_rerank_enabled(monkeypatch):
    captured_settings = {}

    def fake_run_evaluation(settings, questions_path):
        captured_settings["rerank"] = settings.rerank_enabled
        return []

    monkeypatch.setattr("ask_the_docs.cli.run_evaluation", fake_run_evaluation)

    parser = build_parser()
    args = parser.parse_args(["evaluate", "--rerank", "false"])
    args.func(args, make_settings())

    assert captured_settings["rerank"] is False


def test_evaluate_overrides_query_rewrite_enabled(monkeypatch):
    captured_settings = {}

    def fake_run_evaluation(settings, questions_path):
        captured_settings["rewrite"] = settings.query_rewrite_enabled
        return []

    monkeypatch.setattr("ask_the_docs.cli.run_evaluation", fake_run_evaluation)

    parser = build_parser()
    args = parser.parse_args(["evaluate", "--query-rewrite", "false"])
    args.func(args, make_settings())

    assert captured_settings["rewrite"] is False


def test_evaluate_overrides_graph_enabled(monkeypatch):
    captured_settings = {}

    def fake_run_evaluation(settings, questions_path):
        captured_settings["graph"] = settings.graph_enabled
        return []

    monkeypatch.setattr("ask_the_docs.cli.run_evaluation", fake_run_evaluation)

    parser = build_parser()
    args = parser.parse_args(["evaluate", "--graph", "true"])
    args.func(args, make_settings())

    assert captured_settings["graph"] is True


def test_evaluate_rejects_hybrid_alpha_out_of_range(monkeypatch):
    monkeypatch.setattr("ask_the_docs.cli.run_evaluation", lambda settings, path: [])
    parser = build_parser()
    args = parser.parse_args(["evaluate", "--hybrid-alpha", "5.0"])

    with pytest.raises(ValidationError):
        args.func(args, make_settings())


def test_evaluate_combines_multiple_overrides_in_one_call(monkeypatch):
    captured_settings = {}

    def fake_run_evaluation(settings, questions_path):
        captured_settings["backend"] = settings.embeddings_backend
        captured_settings["alpha"] = settings.hybrid_alpha
        captured_settings["rerank"] = settings.rerank_enabled
        captured_settings["rewrite"] = settings.query_rewrite_enabled
        return []

    monkeypatch.setattr("ask_the_docs.cli.run_evaluation", fake_run_evaluation)

    parser = build_parser()
    args = parser.parse_args(
        [
            "evaluate",
            "--embeddings",
            "huggingface",
            "--hybrid-alpha",
            "0.8",
            "--rerank",
            "false",
            "--query-rewrite",
            "false",
        ]
    )
    args.func(args, make_settings())

    assert captured_settings == {"backend": "huggingface", "alpha": 0.8, "rerank": False, "rewrite": False}


def test_serve_starts_uvicorn(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        "ask_the_docs.cli.uvicorn.run",
        lambda app, host, port, log_level: captured.update(
            app=app, host=host, port=port, log_level=log_level
        ),
    )

    parser = build_parser()
    args = parser.parse_args(["serve"])
    args.func(args, make_settings())

    from ask_the_docs.api import app as api_app

    assert captured["app"] is api_app
    assert captured["host"] == "0.0.0.0"
    assert captured["port"] == 8000


def test_worker_runs_the_worker_loop(monkeypatch):
    captured = {}

    async def fake_run_worker(settings):
        captured["settings"] = settings

    monkeypatch.setattr("ask_the_docs.worker.run_worker", fake_run_worker)

    parser = build_parser()
    args = parser.parse_args(["worker"])
    settings = make_settings()
    args.func(args, settings)

    assert captured["settings"] is settings


def test_evaluate_reports_errors(capsys, monkeypatch):
    monkeypatch.setattr(
        "ask_the_docs.cli.run_evaluation",
        lambda settings, questions_path: [EvalResult(question="Q1", score=None, error="boom")],
    )

    parser = build_parser()
    args = parser.parse_args(["evaluate"])
    args.func(args, make_settings())

    out = capsys.readouterr().out
    assert "Q1" in out
    assert "boom" in out
