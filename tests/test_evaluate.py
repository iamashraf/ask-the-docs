import json
from unittest.mock import Mock

from ask_the_docs.answer import AnswerResult
from ask_the_docs.config import REFUSAL_MESSAGE, ProviderConfig, Settings
from ask_the_docs.evaluate import (
    JUDGE_SYSTEM_PROMPT,
    EvalResult,
    GoldenQuestion,
    JudgeScore,
    judge,
    load_golden_questions,
    run_evaluation,
    summarize,
)
from ask_the_docs.router import Turn


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


def test_judge_sends_instructions_as_a_separate_system_prompt(monkeypatch):
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

    judge("what is the capital of France?", "retrieved passage about Paris", "Paris is the capital", client)

    client.generate.assert_called_once()
    call_args, call_kwargs = client.generate.call_args
    prompt = call_args[0]
    assert call_kwargs["system"] == JUDGE_SYSTEM_PROMPT
    # Distinctive values, not generic words like "context"/"answer" that the
    # template's own static labels ("Retrieved context:", "Generated answer:")
    # would already contain regardless of what's actually interpolated.
    assert "what is the capital of France?" in prompt
    assert "retrieved passage about Paris" in prompt
    assert "Paris is the capital" in prompt
    assert "You are evaluating a RAG system's answer." not in prompt
    assert "Respond with ONLY a JSON object" not in prompt


def test_run_evaluation_skips_judge_for_expected_refusal_question_that_correctly_refuses(tmp_path, monkeypatch):
    questions_path = tmp_path / "questions.yaml"
    questions_path.write_text('- question: "Q1"\n  expected_refusal: true\n')

    monkeypatch.setattr(
        "ask_the_docs.evaluate.answer",
        lambda query, settings, history=None, use_cache=True: AnswerResult(
            text=REFUSAL_MESSAGE, citations=[], context="", kind="refusal"
        ),
    )
    mock_get_llm_client = Mock()
    monkeypatch.setattr("ask_the_docs.evaluate.get_llm_client", mock_get_llm_client)

    results = run_evaluation(make_settings(), questions_path)

    assert results[0].error is None
    assert results[0].score is not None
    mock_get_llm_client.assert_not_called()


def test_run_evaluation_flags_error_when_expected_refusal_question_gets_an_answer(tmp_path, monkeypatch):
    questions_path = tmp_path / "questions.yaml"
    questions_path.write_text('- question: "Q1"\n  expected_refusal: true\n')

    monkeypatch.setattr(
        "ask_the_docs.evaluate.answer",
        lambda query, settings, history=None, use_cache=True: AnswerResult(
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
        "ask_the_docs.evaluate.answer",
        lambda query, settings, history=None, use_cache=True: AnswerResult(
            text="some answer", citations=["a.pdf"], context="ctx"
        ),
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
    fake_client.generate.side_effect = lambda prompt, system=None: next(responses)
    monkeypatch.setattr("ask_the_docs.evaluate.get_llm_client", lambda config: fake_client)

    results = run_evaluation(make_settings(), questions_path)

    assert len(results) == 2
    assert results[0].score is None
    assert results[0].error is not None
    assert results[1].error is None
    assert results[1].score.accuracy == 1.0


def test_load_golden_questions_parses_history_into_turns(tmp_path):
    path = tmp_path / "questions.yaml"
    path.write_text(
        '- question: "explain that more simply"\n'
        "  history:\n"
        '    - role: "user"\n'
        '      content: "What is the attention mechanism?"\n'
        '    - role: "assistant"\n'
        '      content: "Attention weighs tokens against each other [1]."\n'
        '  expected_intent: "doc"\n'
    )

    questions = load_golden_questions(path)

    assert questions[0].history == [
        Turn(role="user", content="What is the attention mechanism?"),
        Turn(role="assistant", content="Attention weighs tokens against each other [1]."),
    ]
    assert questions[0].expected_intent == "doc"


def test_load_golden_questions_defaults_history_and_expected_intent(tmp_path):
    path = tmp_path / "questions.yaml"
    path.write_text('- question: "Q1"\n')

    questions = load_golden_questions(path)

    assert questions[0].history == []
    assert questions[0].expected_intent is None
    assert questions[0].max_latency_seconds is None


def test_run_evaluation_calls_answer_with_history_and_without_cache(tmp_path, monkeypatch):
    questions_path = tmp_path / "questions.yaml"
    questions_path.write_text(
        '- question: "explain that more simply"\n'
        "  history:\n"
        '    - role: "user"\n'
        '      content: "What is the attention mechanism?"\n'
        '    - role: "assistant"\n'
        '      content: "Attention weighs tokens against each other [1]."\n'
    )

    mock_answer = Mock(return_value=AnswerResult(text="some answer", citations=["a.pdf"], context="ctx"))
    monkeypatch.setattr("ask_the_docs.evaluate.answer", mock_answer)
    fake_client = Mock()
    fake_client.generate.return_value = json.dumps(
        {"accuracy": 1.0, "answer_relevance": 1.0, "context_relevance": 1.0, "completeness": 1.0, "justification": "ok"}
    )
    monkeypatch.setattr("ask_the_docs.evaluate.get_llm_client", lambda config: fake_client)

    settings = make_settings()
    run_evaluation(settings, questions_path)

    mock_answer.assert_called_once_with(
        "explain that more simply",
        settings,
        history=[
            Turn(role="user", content="What is the attention mechanism?"),
            Turn(role="assistant", content="Attention weighs tokens against each other [1]."),
        ],
        use_cache=False,
    )


def test_run_evaluation_records_latency(tmp_path, monkeypatch):
    questions_path = tmp_path / "questions.yaml"
    questions_path.write_text('- question: "Q1"\n')

    monkeypatch.setattr(
        "ask_the_docs.evaluate.answer",
        lambda query, settings, history=None, use_cache=True: AnswerResult(
            text="some answer", citations=["a.pdf"], context="ctx"
        ),
    )
    times = iter([10.0, 10.25])
    monkeypatch.setattr("ask_the_docs.evaluate.time.monotonic", lambda: next(times))
    fake_client = Mock()
    fake_client.generate.return_value = json.dumps(
        {"accuracy": 1.0, "answer_relevance": 1.0, "context_relevance": 1.0, "completeness": 1.0, "justification": "ok"}
    )
    monkeypatch.setattr("ask_the_docs.evaluate.get_llm_client", lambda config: fake_client)

    results = run_evaluation(make_settings(), questions_path)

    assert results[0].latency_seconds == 0.25


def test_run_evaluation_flags_error_on_intent_mismatch_and_skips_judge(tmp_path, monkeypatch):
    questions_path = tmp_path / "questions.yaml"
    questions_path.write_text('- question: "hi"\n  expected_intent: "chitchat"\n')

    monkeypatch.setattr(
        "ask_the_docs.evaluate.answer",
        lambda query, settings, history=None, use_cache=True: AnswerResult(
            text="doc answer", citations=["a.pdf"], context="ctx", kind="answer"
        ),
    )
    mock_get_llm_client = Mock()
    monkeypatch.setattr("ask_the_docs.evaluate.get_llm_client", mock_get_llm_client)

    results = run_evaluation(make_settings(), questions_path)

    assert results[0].score is None
    assert results[0].error is not None
    assert "chitchat" in results[0].error
    assert "doc" in results[0].error
    assert results[0].intent == "doc"
    assert results[0].expected_intent == "chitchat"
    mock_get_llm_client.assert_not_called()


def test_run_evaluation_flags_error_when_latency_exceeds_max(tmp_path, monkeypatch):
    questions_path = tmp_path / "questions.yaml"
    questions_path.write_text('- question: "hi"\n  expected_intent: "chitchat"\n  max_latency_seconds: 1\n')

    monkeypatch.setattr(
        "ask_the_docs.evaluate.answer",
        lambda query, settings, history=None, use_cache=True: AnswerResult(
            text="Hi!", citations=[], context="", kind="chitchat"
        ),
    )
    times = iter([0.0, 2.0])
    monkeypatch.setattr("ask_the_docs.evaluate.time.monotonic", lambda: next(times))
    mock_get_llm_client = Mock()
    monkeypatch.setattr("ask_the_docs.evaluate.get_llm_client", mock_get_llm_client)

    results = run_evaluation(make_settings(), questions_path)

    assert results[0].score is None
    assert results[0].error is not None
    assert "latency" in results[0].error
    assert results[0].latency_seconds == 2.0
    mock_get_llm_client.assert_not_called()


def test_run_evaluation_flags_error_when_expected_answer_question_gets_a_refusal(tmp_path, monkeypatch):
    questions_path = tmp_path / "questions.yaml"
    questions_path.write_text('- question: "Q1"\n  expected_answer: true\n')

    monkeypatch.setattr(
        "ask_the_docs.evaluate.answer",
        lambda query, settings, history=None, use_cache=True: AnswerResult(
            text=REFUSAL_MESSAGE, citations=[], context="", kind="refusal"
        ),
    )
    mock_get_llm_client = Mock()
    monkeypatch.setattr("ask_the_docs.evaluate.get_llm_client", mock_get_llm_client)

    results = run_evaluation(make_settings(), questions_path)

    assert results[0].error == "expected an answer but got a refusal"
    assert results[0].score is None
    mock_get_llm_client.assert_not_called()


def test_run_evaluation_judges_expected_answer_question_normally_when_it_gets_an_answer(tmp_path, monkeypatch):
    questions_path = tmp_path / "questions.yaml"
    questions_path.write_text('- question: "Q1"\n  expected_answer: true\n')

    monkeypatch.setattr(
        "ask_the_docs.evaluate.answer",
        lambda query, settings, history=None, use_cache=True: AnswerResult(
            text="a real answer", citations=["a.pdf"], context="ctx", kind="answer"
        ),
    )
    fake_client = Mock()
    fake_client.generate.return_value = json.dumps(
        {"accuracy": 1.0, "answer_relevance": 1.0, "context_relevance": 1.0, "completeness": 1.0, "justification": "ok"}
    )
    monkeypatch.setattr("ask_the_docs.evaluate.get_llm_client", lambda config: fake_client)

    results = run_evaluation(make_settings(), questions_path)

    assert results[0].error is None
    assert results[0].score == JudgeScore(1.0, 1.0, 1.0, 1.0, "ok")
    fake_client.generate.assert_called_once()


def test_run_evaluation_scores_chitchat_deterministically_when_intent_matches(tmp_path, monkeypatch):
    questions_path = tmp_path / "questions.yaml"
    questions_path.write_text('- question: "hi"\n  expected_intent: "chitchat"\n')

    monkeypatch.setattr(
        "ask_the_docs.evaluate.answer",
        lambda query, settings, history=None, use_cache=True: AnswerResult(
            text="Hi! Ask me anything about the indexed documents.", citations=[], context="", kind="chitchat"
        ),
    )
    mock_get_llm_client = Mock()
    monkeypatch.setattr("ask_the_docs.evaluate.get_llm_client", mock_get_llm_client)

    results = run_evaluation(make_settings(), questions_path)

    assert results[0].error is None
    assert results[0].score == JudgeScore(1.0, 1.0, 1.0, 1.0, results[0].score.justification)
    assert results[0].intent == "chitchat"
    mock_get_llm_client.assert_not_called()


def test_summarize_computes_intent_accuracy_and_latency_percentiles():
    results = [
        EvalResult(question="q1", score=JudgeScore(1, 1, 1, 1, "x"), error=None, intent="doc", expected_intent="doc", latency_seconds=0.1),
        EvalResult(
            question="q2",
            score=None,
            error="expected intent 'doc' but got 'chitchat'",
            intent="chitchat",
            expected_intent="doc",
            latency_seconds=0.5,
        ),
        EvalResult(question="q3", score=JudgeScore(1, 1, 1, 1, "x"), error=None, intent="doc", latency_seconds=0.3),
    ]

    summary = summarize(results)

    assert summary["intent_accuracy"] == 0.5
    assert summary["p50_latency_seconds"] == 0.3
    assert summary["p95_latency_seconds"] == 0.5


def test_summarize_returns_none_when_no_data():
    results = [EvalResult(question="q1", score=None, error="boom")]

    summary = summarize(results)

    assert summary == {"intent_accuracy": None, "p50_latency_seconds": None, "p95_latency_seconds": None}
