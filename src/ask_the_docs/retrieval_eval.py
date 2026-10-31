"""Standalone eval: does retrieve() actually surface the right source paper for a labeled question set?"""

from dataclasses import dataclass
from pathlib import Path

import yaml

from ask_the_docs.config import Settings
from ask_the_docs.retrieve import retrieve


@dataclass
class RetrievalCase:
    question: str
    expected_source: str


@dataclass
class RetrievalResult:
    case: RetrievalCase
    rank: int | None  # 1-based rank of the first chunk from expected_source; None if absent
    returned_sources: list[str]


def load_retrieval_cases(path: Path) -> list[RetrievalCase]:
    data = yaml.safe_load(path.read_text())
    return [RetrievalCase(question=item["question"], expected_source=item["expected_source"]) for item in data]


def run_retrieval_eval(settings: Settings, cases_path: Path, k: int = 5) -> list[RetrievalResult]:
    results = []
    for case in load_retrieval_cases(cases_path):
        returned_sources = [doc.metadata["source"] for doc in retrieve(case.question, settings, k)]
        rank = next(
            (i for i, source in enumerate(returned_sources, start=1) if source == case.expected_source), None
        )
        results.append(RetrievalResult(case=case, rank=rank, returned_sources=returned_sources))
    return results


def retrieval_metrics(results: list[RetrievalResult]) -> dict:
    """hit@1, hit@k, and MRR over results; k here is the longest returned_sources seen, purely for reporting."""
    if not results:
        return {"hit_at_1": 0.0, "hit_at_k": 0.0, "mrr": 0.0, "k": 0.0}
    n = len(results)
    k = max(len(r.returned_sources) for r in results)
    hit_at_1 = sum(1 for r in results if r.rank == 1) / n
    hit_at_k = sum(1 for r in results if r.rank is not None) / n
    mrr = sum(1 / r.rank for r in results if r.rank is not None) / n
    return {"hit_at_1": hit_at_1, "hit_at_k": hit_at_k, "mrr": mrr, "k": k}


def retrieval_report(results: list[RetrievalResult], k: int) -> str:
    """Metrics summary plus every miss with the sources that came back instead."""
    metrics = retrieval_metrics(results)
    lines = [
        f"hit@1: {metrics['hit_at_1']:.2f}",
        f"hit@{k}: {metrics['hit_at_k']:.2f}",
        f"mrr: {metrics['mrr']:.2f}",
    ]

    misses = [r for r in results if r.rank is None]
    if misses:
        lines.append("\nMisses:")
        for r in misses:
            lines.append(f"- {r.case.question!r}: expected {r.case.expected_source}, got {r.returned_sources}")

    return "\n".join(lines)
