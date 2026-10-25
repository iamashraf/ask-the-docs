import json
from unittest.mock import Mock

from research_assistant.answer import AnswerResult
from research_assistant.config import REFUSAL_MESSAGE, ProviderConfig, Settings
from research_assistant.evaluate import GoldenQuestion, JudgeScore, judge, load_golden_questions, run_evaluation


def make_settings() -> Settings:
    return Settings(
        generator=ProviderConfig(provider="ollama", model="llama3.1:8b"),
        judge=ProviderConfig(provider="anthropic", model="claude-sonnet-5-5", api_key="x"),
    )


def test_load_golden_questions_parses_yaml(tmp_path):
    path = tmp_path / "questions.yaml"
    path.write_text('- question: "What is X?"\n  notes: "note one"\n- question: "What is Y?"\n')

    questions = load_golden_questions(path)

    assert questions == [
        GoldenQuestion(question="What is X?", notes="note one"),
        GoldenQuestion(question="What is Y?", notes=None),
    ]


def test_load_golden_questions_parses_expected_refusal_flag(tmp_path):
    path = tmp_path / "questions.yaml"
    path.write_text('- question: "What is X?"\n  expected_refusal: true\n- question: "What is Y?"\n')

    questions = load_golden_questions(path)

    assert questions == [
        GoldenQuestion(question="What is X?", expected_refusal=True),
        GoldenQuestion(question="What is Y?", expected_refusal=False),
    ]


def test_judge_parses_well_formed_json_response():
    client = Mock()
    client.generate.return_value = json.dumps(
        {
            "accuracy": 0.9,
            "answer_relevance": 0.8,
            "context_relevance": 0.7,
            "completeness": 0.85,
            "justification": "Good answer.",
        }
    )

    score = judge("q", "context", "answer", client)

    assert score == JudgeScore(
        accuracy=0.9,
        answer_relevance=0.8,
        context_relevance=0.7,
        completeness=0.85,
        justification="Good answer.",
    )


def test_judge_returns_none_for_non_json_response():
    client = Mock()
    client.generate.return_value = "I cannot provide a numeric score."

    assert judge("q", "context", "answer", client) is None


def test_judge_returns_none_for_json_missing_required_field():
    client = Mock()
    client.generate.return_value = json.dumps({"accuracy": 0.9})

    assert judge("q", "context", "answer", client) is None


def test_run_evaluation_skips_judge_for_expected_refusal_question_that_correctly_refuses(tmp_path, monkeypatch):
    questions_path = tmp_path / "questions.yaml"
    questions_path.write_text('- question: "Q1"\n  expected_refusal: true\n')

    monkeypatch.setattr(
        "research_assistant.evaluate.answer",
        lambda query, settings: AnswerResult(text=REFUSAL_MESSAGE, citations=[], context=""),
    )
    mock_get_llm_client = Mock()
    monkeypatch.setattr("research_assistant.evaluate.get_llm_client", mock_get_llm_client)

    results = run_evaluation(make_settings(), questions_path)

    assert results[0].error is None
    assert results[0].score is not None
    mock_get_llm_client.assert_not_called()


def test_run_evaluation_flags_error_when_expected_refusal_question_gets_an_answer(tmp_path, monkeypatch):
    questions_path = tmp_path / "questions.yaml"
    questions_path.write_text('- question: "Q1"\n  expected_refusal: true\n')

    monkeypatch.setattr(
        "research_assistant.evaluate.answer",
        lambda query, settings: AnswerResult(
            text="a confident but wrong answer", citations=["a.pdf"], context="ctx"
        ),
    )

    results = run_evaluation(make_settings(), questions_path)

    assert results[0].error is not None
    assert results[0].score is None


def test_run_evaluation_marks_error_and_continues(tmp_path, monkeypatch):
    questions_path = tmp_path / "questions.yaml"
    questions_path.write_text('- question: "Q1"\n- question: "Q2"\n')

    monkeypatch.setattr(
        "research_assistant.evaluate.answer",
        lambda query, settings: AnswerResult(text="some answer", citations=["a.pdf"], context="ctx"),
    )

    responses = iter(
        [
            "not json",
            json.dumps(
                {
                    "accuracy": 1.0,
                    "answer_relevance": 1.0,
                    "context_relevance": 1.0,
                    "completeness": 1.0,
                    "justification": "ok",
                }
            ),
        ]
    )
    fake_client = Mock()
    fake_client.generate.side_effect = lambda prompt: next(responses)
    monkeypatch.setattr("research_assistant.evaluate.get_llm_client", lambda config: fake_client)

    results = run_evaluation(make_settings(), questions_path)

    assert len(results) == 2
    assert results[0].score is None
    assert results[0].error is not None
    assert results[1].error is None
    assert results[1].score.accuracy == 1.0
