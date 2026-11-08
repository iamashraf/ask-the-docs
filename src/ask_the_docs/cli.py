"""Command-line entry point: ingest, index, ask, evaluate."""

import argparse
import sys
from pathlib import Path

import uvicorn

from ask_the_docs.answer import answer
from ask_the_docs.api import app as api_app
from ask_the_docs.config import Settings, get_settings
from ask_the_docs.evaluate import run_evaluation, summarize
from ask_the_docs.index import build_index, prune_missing_docs_sources
from ask_the_docs.ingest import dispatch, dispatch_all_under_docs_dir
from ask_the_docs.retrieval_eval import retrieval_metrics, retrieval_report, run_retrieval_eval
from ask_the_docs.routing_eval import routing_report, run_routing_eval

GOLDEN_QUESTIONS_PATH = Path("evals/golden_questions.yaml")
ROUTING_CASES_PATH = Path("evals/routing_cases.yaml")
RETRIEVAL_CASES_PATH = Path("evals/retrieval_cases.yaml")


def _ingest(args: argparse.Namespace, settings) -> None:
    docs = []
    for source in args.sources:
        docs.extend(dispatch(source))
    if docs:
        build_index(docs, settings)
    print(f"Ingested {len(docs)} document(s) from {len(args.sources)} source(s).")


def _index(args: argparse.Namespace, settings) -> None:
    """Bulk re-index everything already under docs_dir; idempotent."""
    docs, file_count = dispatch_all_under_docs_dir(settings)
    if docs:
        build_index(docs, settings)
    prune_missing_docs_sources(settings)
    print(f"Indexed {len(docs)} document(s) from {file_count} file(s) under {settings.docs_dir}.")


def _ask(args: argparse.Namespace, settings) -> None:
    result = answer(args.query, settings, k=args.k)
    print(result.text)
    if result.citations:
        print("\nSources:")
        for citation in result.citations:
            print(f"- {citation}")


def _evaluate(args: argparse.Namespace, settings) -> None:
    overrides = {}
    if args.embeddings:
        overrides["embeddings_backend"] = args.embeddings
    if args.hybrid_alpha is not None:
        overrides["hybrid_alpha"] = args.hybrid_alpha
    if args.rerank is not None:
        overrides["rerank_enabled"] = args.rerank == "true"
    if args.query_rewrite is not None:
        overrides["query_rewrite_enabled"] = args.query_rewrite == "true"
    if args.graph is not None:
        overrides["graph_enabled"] = args.graph == "true"
    if overrides:
        # model_copy(update=...) skips validation entirely -- model_validate
        # re-runs it, so an out-of-range --hybrid-alpha is rejected here the
        # same way the API already rejects it, instead of silently passing
        # through to run_evaluation.
        settings = Settings.model_validate({**settings.model_dump(), **overrides})

    questions_path = Path(args.questions) if args.questions else GOLDEN_QUESTIONS_PATH
    results = run_evaluation(settings, questions_path)
    for result in results:
        if result.error:
            print(f"{result.question}: ERROR ({result.error})")
        else:
            s = result.score
            print(
                f"{result.question}: accuracy={s.accuracy} answer_relevance={s.answer_relevance} "
                f"context_relevance={s.context_relevance} completeness={s.completeness}"
            )

    def _fmt(value: float | None) -> str:
        return "n/a" if value is None else f"{value:.2f}"

    summary = summarize(results)
    print(
        f"\nintent_accuracy={_fmt(summary['intent_accuracy'])} "
        f"p50_latency_seconds={_fmt(summary['p50_latency_seconds'])} "
        f"p95_latency_seconds={_fmt(summary['p95_latency_seconds'])}"
    )


def _eval_routing(args: argparse.Namespace, settings) -> None:
    results = run_routing_eval(settings, args.cases)
    print(routing_report(results))
    if any(result.critical for result in results):
        sys.exit(1)


def _eval_retrieval(args: argparse.Namespace, settings) -> None:
    results = run_retrieval_eval(settings, args.cases, args.k)
    print(retrieval_report(results, args.k))
    if retrieval_metrics(results)["hit_at_k"] < args.min_hit_at_k:
        sys.exit(1)


def _serve(args: argparse.Namespace, settings) -> None:
    uvicorn.run(api_app, host=settings.api_host, port=8000, log_level="info")


def _worker(args: argparse.Namespace, settings) -> None:
    import asyncio

    from ask_the_docs.worker import run_worker

    asyncio.run(run_worker(settings))


def _watch(args: argparse.Namespace, settings) -> None:
    import asyncio

    from ask_the_docs.watcher import watch_docs_dir

    asyncio.run(watch_docs_dir(settings))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ask-the-docs")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser("ingest", help="Load and index one or more sources (paths or URLs)")
    ingest_parser.add_argument("sources", nargs="+")
    ingest_parser.set_defaults(func=_ingest)

    index_parser = subparsers.add_parser(
        "index", help="Bulk re-index everything under docs_dir (idempotent)"
    )
    index_parser.set_defaults(func=_index)

    ask_parser = subparsers.add_parser("ask", help="Ask a question against the indexed documents")
    ask_parser.add_argument("query")
    ask_parser.add_argument("--k", type=int, default=5)
    ask_parser.set_defaults(func=_ask)

    evaluate_parser = subparsers.add_parser("evaluate", help="Run the golden question set through the LLM judge")
    evaluate_parser.add_argument("--embeddings", choices=["ollama", "huggingface"], default=None)
    evaluate_parser.add_argument("--hybrid-alpha", type=float, default=None)
    evaluate_parser.add_argument("--rerank", choices=["true", "false"], default=None)
    evaluate_parser.add_argument("--query-rewrite", choices=["true", "false"], default=None)
    evaluate_parser.add_argument("--graph", choices=["true", "false"], default=None)
    evaluate_parser.add_argument("--questions", default=None, help="Path to a golden-question-set YAML; defaults to evals/golden_questions.yaml")
    evaluate_parser.set_defaults(func=_evaluate)

    eval_routing_parser = subparsers.add_parser(
        "eval-routing", help="Run the labeled routing case set through route() and report accuracy"
    )
    eval_routing_parser.add_argument("--cases", type=Path, default=ROUTING_CASES_PATH)
    eval_routing_parser.set_defaults(func=_eval_routing)

    eval_retrieval_parser = subparsers.add_parser(
        "eval-retrieval", help="Run the labeled retrieval case set through retrieve() and report hit@k/MRR"
    )
    eval_retrieval_parser.add_argument("--cases", type=Path, default=RETRIEVAL_CASES_PATH)
    eval_retrieval_parser.add_argument("--k", type=int, default=5)
    # Calibrated against the live stack on 2026-10-31: measured hit@5=0.87,
    # hit@1=0.73, MRR=0.79 over the 15 seeded papers (llama3.1:8b generator,
    # nomic-embed-text embeddings). Set just under the measured baseline.
    eval_retrieval_parser.add_argument("--min-hit-at-k", type=float, default=0.85)
    eval_retrieval_parser.set_defaults(func=_eval_retrieval)

    serve_parser = subparsers.add_parser("serve", help="Run the long-running HTTP API server")
    serve_parser.set_defaults(func=_serve)

    worker_parser = subparsers.add_parser("worker", help="Run the async ingestion worker, consuming jobs from NATS")
    worker_parser.set_defaults(func=_worker)

    watch_parser = subparsers.add_parser(
        "watch", help="Watch docs_dir and enqueue an ingest job for each new/changed file"
    )
    watch_parser.set_defaults(func=_watch)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args, get_settings())
