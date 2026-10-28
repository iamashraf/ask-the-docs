"""LLM-as-judge evaluation: scores a generated answer against its retrieved
context and the question, using a (usually stronger) separate judge model."""

import json
from dataclasses import dataclass
from pathlib import Path

import yaml

from ask_the_docs.answer import answer
from ask_the_docs.config import REFUSAL_MESSAGE, Settings
from ask_the_docs.llm import LLMClient, get_llm_client

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


def load_golden_questions(path: Path) -> list[GoldenQuestion]:
    data = yaml.safe_load(path.read_text())
    return [
        GoldenQuestion(
            question=item["question"],
            notes=item.get("notes"),
            expected_refusal=item.get("expected_refusal", False),
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
        try:
            result = answer(golden.question, settings)
            if golden.expected_refusal:
                if result.text == REFUSAL_MESSAGE:
                    score = JudgeScore(1.0, 1.0, 1.0, 1.0, "correctly refused, as expected")
                    error = None
                else:
                    score = None
                    error = f"expected a refusal but got an answer: {result.text[:100]!r}"
            else:
                if judge_client is None:
                    judge_client = get_llm_client(settings.judge)
                score = judge(golden.question, result.context, result.text, judge_client)
                error = None if score is not None else "judge response was not parseable JSON"
        except Exception as exc:
            score = None
            error = str(exc)
        results.append(EvalResult(question=golden.question, score=score, error=error))

    return results
