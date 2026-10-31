import pytest
from pydantic import ValidationError

from ask_the_docs.router import (
    GOODBYE_REPLY,
    GREETING_REPLY,
    META_CAPABILITY_SENTENCE,
    NO_DOCUMENTS_SENTENCE,
    THANKS_REPLY,
    Turn,
    build_meta_reply,
    match_smalltalk,
    trim_history,
)


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("hi", GREETING_REPLY),
        ("Hi!!", GREETING_REPLY),
        ("hello", GREETING_REPLY),
        ("Hello.", GREETING_REPLY),
        ("hey", GREETING_REPLY),
        ("hey there", GREETING_REPLY),
        ("hi there", GREETING_REPLY),
        ("hello there", GREETING_REPLY),
        ("good morning", GREETING_REPLY),
        ("Good evening!", GREETING_REPLY),
        ("yo", GREETING_REPLY),
        ("hiya", GREETING_REPLY),
        ("\U0001f44b", GREETING_REPLY),
        ("  hi \n", GREETING_REPLY),
        ("thanks", THANKS_REPLY),
        ("Thank you!", THANKS_REPLY),
        ("thank you so much", THANKS_REPLY),
        ("thx", THANKS_REPLY),
        ("ty", THANKS_REPLY),
        ("thanks a lot", THANKS_REPLY),
        ("thanks \U0001f64f", THANKS_REPLY),
        ("\U0001f64f", THANKS_REPLY),
        ("bye", GOODBYE_REPLY),
        ("Goodbye!", GOODBYE_REPLY),
        ("see you", GOODBYE_REPLY),
        ("cya", GOODBYE_REPLY),
    ],
)
def test_match_smalltalk_matches(query, expected):
    assert match_smalltalk(query) == expected


@pytest.mark.parametrize(
    "query",
    [
        "hi, what is LoRA?",
        "hello, explain attention",
        "thanks, and what about RLHF?",
        "hey can you summarize the GPT-3 paper",
        "?",
        "",
        "   ",
        "hi " + "please tell me about transformers " * 3,
        "history of attention",
        "high-rank adaptation",
        "hello world program in python",
        "good morning, what is chain of thought?",
    ],
)
def test_match_smalltalk_rejects(query):
    assert match_smalltalk(query) is None


def _make_turns(n: int) -> list[Turn]:
    return [Turn(role="user" if i % 2 == 0 else "assistant", content=f"turn {i}") for i in range(n)]


def test_trim_history_none_returns_empty():
    assert trim_history(None, 6) == []


def test_trim_history_empty_returns_empty():
    assert trim_history([], 6) == []


def test_trim_history_under_limit_returns_all():
    turns = _make_turns(6)
    assert trim_history(turns, 6) == turns


def test_trim_history_over_limit_returns_last_n_in_order():
    turns = _make_turns(7)
    assert trim_history(turns, 6) == turns[-6:]


def test_trim_history_zero_max_turns_returns_empty():
    turns = _make_turns(3)
    assert trim_history(turns, 0) == []


def test_turn_rejects_empty_and_oversized_content():
    with pytest.raises(ValidationError):
        Turn(role="user", content="")
    with pytest.raises(ValidationError):
        Turn(role="user", content="x" * 4001)


def test_turn_rejects_unknown_role():
    with pytest.raises(ValidationError):
        Turn(role="system", content="hi")


def test_build_meta_reply_empty():
    assert build_meta_reply([]) == f"{META_CAPABILITY_SENTENCE} {NO_DOCUMENTS_SENTENCE}"


def test_build_meta_reply_lists_sources():
    reply = build_meta_reply(["b.pdf", "a.pdf"])
    assert reply.startswith(META_CAPABILITY_SENTENCE)
    assert "2 documents" in reply
    assert reply.splitlines()[1:] == ["- b.pdf", "- a.pdf"]


def test_build_meta_reply_singular_for_one_source():
    reply = build_meta_reply(["a.pdf"])
    assert "1 document" in reply
    assert "1 documents" not in reply
