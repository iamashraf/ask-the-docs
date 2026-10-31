"""LLM-as-judge evaluation: scores a generated answer against its retrieved
context and the question, using a (usually stronger) separate judge model."""

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import yaml

from ask_the_docs.answer import answer
from ask_the_docs.config import Settings
from ask_the_docs.llm import LLMClient, get_llm_client
from ask_the_docs.router import Turn

JUDGE_PROMPT_TEMPLATE = """You are evaluating a RAG system's answer.

Question: {question}

Retrieved context:
{context}

Generated answer:
{answer_text}

Score the answer from 0.0 to 1.0 on each dimension, with a short justification.
Respond with ONLY a JSON object of this exact shape:
{{"accuracy": <float>, "answer_relevance": <float>, "context_relevance": <float>, "completeness": <float>, "justification": "<string>"}}
"""


@dataclass
class GoldenQuestion:
    question: str
    notes: str | None = None
    expected_refusal: bool = False
    expected_intent: Literal["doc", "chitchat", "meta"] | None = None
    history: list[Turn] = field(default_factory=list)
    max_latency_seconds: float | None = None


@dataclass
class JudgeScore:
    accuracy: float
    answer_relevance: float
    context_relevance: float
    completeness: float
    justification: str


@dataclass
class EvalResult:
    question: str
    score: JudgeScore | None
    error: str | None
    intent: str | None = None  # "doc" for kind answer/refusal, else the kind
    # Deviation from the brief: carries the golden question's expected_intent
    # (not just the observed one) so summarize() can compute accuracy from
    # EvalResults alone, without re-reading golden_questions.yaml.
    expected_intent: str | None = None
    latency_seconds: float | None = None


def load_golden_questions(path: Path) -> list[GoldenQuestion]:
    data = yaml.safe_load(path.read_text())
    return [
        GoldenQuestion(
            question=item["question"],
            notes=item.get("notes"),
            expected_refusal=item.get("expected_refusal", False),
            expected_intent=item.get("expected_intent"),
            history=[Turn(**turn) for turn in item.get("history", [])],
            max_latency_seconds=item.get("max_latency_seconds"),
        )
        for item in data
    ]


def judge(question: str, context: str, answer_text: str, judge_client: LLMClient) -> JudgeScore | None:
    """Returns None if the judge's response isn't parseable JSON matching the rubric shape."""
    prompt = JUDGE_PROMPT_TEMPLATE.format(question=question, context=context, answer_text=answer_text)
    raw = judge_client.generate(prompt)
    try:
        data = json.loads(raw)
        return JudgeScore(
            accuracy=float(data["accuracy"]),
            answer_relevance=float(data["answer_relevance"]),
            context_relevance=float(data["context_relevance"]),
            completeness=float(data["completeness"]),
            justification=str(data["justification"]),
        )
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None


def run_evaluation(settings: Settings, questions_path: Path) -> list[EvalResult]:
    """Runs every golden question through answer+judge; a failure on one
    question (bad judge output, LLM error) is recorded, not fatal to the run.

    A question marked expected_refusal skips the LLM judge entirely: the
    judge has no context to score when the correct answer is a refusal (it
    returns empty context), so it would otherwise score a correct refusal
    poorly on accuracy/completeness -- rewarding hallucination over
    correct refusal in exactly the hybrid/rerank comparisons this is meant
    to validate. A deterministic check replaces it instead.
    """
    questions = load_golden_questions(questions_path)
    judge_client: LLMClient | None = None

    results = []
    for golden in questions:
        intent: str | None = None
        latency: float | None = None
        try:
            start = time.monotonic()
            result = answer(golden.question, settings, history=golden.history, use_cache=False)
            latency = time.monotonic() - start
            intent = "doc" if result.kind in ("answer", "refusal") else result.kind

            if golden.expected_intent is not None and intent != golden.expected_intent:
                score = None
                error = f"expected intent {golden.expected_intent!r} but got {intent!r}"
            elif golden.max_latency_seconds is not None and latency > golden.max_latency_seconds:
                score = None
                error = f"latency {latency:.3f}s exceeded max {golden.max_latency_seconds}s"
            elif golden.expected_refusal:
                if result.kind == "refusal":
                    score = JudgeScore(1.0, 1.0, 1.0, 1.0, "correctly refused, as expected")
                    error = None
                else:
                    score = None
                    error = f"expected a refusal but got an answer: {result.text[:100]!r}"
            elif golden.expected_intent in ("chitchat", "meta"):
                # No retrieved context to judge for a canned chitchat/meta
                # reply -- same rationale as the expected_refusal case above.
                score = JudgeScore(1.0, 1.0, 1.0, 1.0, f"deterministic {golden.expected_intent} reply, not scored by judge")
                error = None
            else:
                if judge_client is None:
                    judge_client = get_llm_client(settings.judge)
                score = judge(golden.question, result.context, result.text, judge_client)
                error = None if score is not None else "judge response was not parseable JSON"
        except Exception as exc:
            score = None
            error = str(exc)
        results.append(
            EvalResult(
                question=golden.question,
                score=score,
                error=error,
                intent=intent,
                expected_intent=golden.expected_intent,
                latency_seconds=latency,
            )
        )

    return results


def _percentile(sorted_values: list[float], fraction: float) -> float | None:
    """Nearest-rank percentile: rank = ceil(fraction * n), 1-indexed, clamped to the last value."""
    if not sorted_values:
        return None
    n = len(sorted_values)
    rank = max(math.ceil(fraction * n), 1)
    return sorted_values[min(rank, n) - 1]


def summarize(results: list[EvalResult]) -> dict:
    """Aggregates intent-match rate (over questions with an expected_intent) and latency percentiles."""
    checked = [r for r in results if r.expected_intent is not None]
    intent_accuracy = (
        sum(1 for r in checked if r.intent == r.expected_intent) / len(checked) if checked else None
    )

    latencies = sorted(r.latency_seconds for r in results if r.latency_seconds is not None)
    return {
        "intent_accuracy": intent_accuracy,
        "p50_latency_seconds": _percentile(latencies, 0.50),
        "p95_latency_seconds": _percentile(latencies, 0.95),
    }
