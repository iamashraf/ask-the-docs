"""Command-line entry point: ingest, index, ask, evaluate."""

import argparse
from pathlib import Path

from research_assistant.answer import answer
from research_assistant.config import get_settings
from research_assistant.evaluate import run_evaluation
from research_assistant.index import build_index
from research_assistant.ingest import dispatch, discover_sources

GOLDEN_QUESTIONS_PATH = Path("evals/golden_questions.yaml")


def _ingest(args: argparse.Namespace, settings) -> None:
    docs = []
    for source in args.sources:
        docs.extend(dispatch(source))
    if docs:
        build_index(docs, settings)
    print(f"Ingested {len(docs)} document(s) from {len(args.sources)} source(s).")


def _index(args: argparse.Namespace, settings) -> None:
    """Bulk re-index everything already under docs_dir. Running this twice on
    unchanged files will duplicate their chunks (document-update/dedup is not
    built yet)."""
    paths = discover_sources(settings.docs_dir)
    docs = []
    for path in paths:
        docs.extend(dispatch(str(path.relative_to(settings.docs_dir))))
    if docs:
        build_index(docs, settings)
    print(f"Indexed {len(docs)} document(s) from {len(paths)} file(s) under {settings.docs_dir}.")


def _ask(args: argparse.Namespace, settings) -> None:
    result = answer(args.query, settings, k=args.k)
    print(result.text)
    if result.citations:
        print("\nSources:")
        for citation in result.citations:
            print(f"- {citation}")


def _evaluate(args: argparse.Namespace, settings) -> None:
    if args.embeddings:
        settings = settings.model_copy(update={"embeddings_backend": args.embeddings})

    for result in run_evaluation(settings, GOLDEN_QUESTIONS_PATH):
        if result.error:
            print(f"{result.question}: ERROR ({result.error})")
        else:
            s = result.score
            print(
                f"{result.question}: accuracy={s.accuracy} answer_relevance={s.answer_relevance} "
                f"context_relevance={s.context_relevance} completeness={s.completeness}"
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="research-assistant")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser("ingest", help="Load and index one or more sources (paths or URLs)")
    ingest_parser.add_argument("sources", nargs="+")
    ingest_parser.set_defaults(func=_ingest)

    index_parser = subparsers.add_parser(
        "index", help="Bulk re-index everything under docs_dir (may duplicate already-indexed files)"
    )
    index_parser.set_defaults(func=_index)

    ask_parser = subparsers.add_parser("ask", help="Ask a question against the indexed documents")
    ask_parser.add_argument("query")
    ask_parser.add_argument("--k", type=int, default=5)
    ask_parser.set_defaults(func=_ask)

    evaluate_parser = subparsers.add_parser("evaluate", help="Run the golden question set through the LLM judge")
    evaluate_parser.add_argument("--embeddings", choices=["ollama", "huggingface"], default=None)
    evaluate_parser.set_defaults(func=_evaluate)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args, get_settings())
