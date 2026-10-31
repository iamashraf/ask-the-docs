"""Routing primitives: conversation history, smalltalk fast path, and meta replies."""

import dataclasses
import re
from typing import Literal

from pydantic import BaseModel, Field

Intent = Literal["doc", "chitchat", "meta"]
RouteSource = Literal["rules", "llm", "fallback", "disabled"]

GREETING_REPLY = "Hi! Ask me anything about the indexed documents."
THANKS_REPLY = "You're welcome! Ask me anything else about the indexed documents."
GOODBYE_REPLY = "Goodbye! Come back anytime you have questions about the documents."
META_CAPABILITY_SENTENCE = "I answer questions using only the indexed documents, and cite the sources I used."
NO_DOCUMENTS_SENTENCE = "No documents are indexed yet."

_GREETINGS = {"hi", "hello", "hey", "hey there", "hi there", "hello there", "good morning", "good afternoon", "good evening", "yo", "hiya"}
_THANKS = {"thanks", "thank you", "thank you so much", "thx", "ty", "thanks a lot"}
_BYE = {"bye", "goodbye", "see you", "see ya", "cya"}

_NON_WORD_PATTERN = re.compile(r"[^\w\s']", re.UNICODE)
_WHITESPACE_PATTERN = re.compile(r"\s+")


class Turn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=4000)


@dataclasses.dataclass
class Route:
    intent: Intent
    standalone_query: str
    reply: str
    source: RouteSource


def _only_character(stripped: str, char: str) -> bool:
    return bool(stripped) and all(c == char for c in stripped)


def match_smalltalk(query: str) -> str | None:
    """Cheap rule-based fast path for bare greetings/thanks/goodbyes, exact match only."""
    stripped = query.strip()
    if _only_character(stripped, "\U0001f44b"):
        return GREETING_REPLY
    if _only_character(stripped, "\U0001f64f"):
        return THANKS_REPLY

    normalized = _NON_WORD_PATTERN.sub("", query.lower())
    normalized = _WHITESPACE_PATTERN.sub(" ", normalized).strip()

    if normalized in _GREETINGS:
        return GREETING_REPLY
    if normalized in _THANKS:
        return THANKS_REPLY
    if normalized in _BYE:
        return GOODBYE_REPLY
    return None


def trim_history(history: list[Turn] | None, max_turns: int) -> list[Turn]:
    """Bounds how much conversation context gets sent to the analyze call."""
    if not history or max_turns <= 0:
        return []
    return history[-max_turns:]


def build_meta_reply(sources: list[str]) -> str:
    """Answers "what documents do you have?" deterministically from a list of source names."""
    if not sources:
        return f"{META_CAPABILITY_SENTENCE} {NO_DOCUMENTS_SENTENCE}"
    n = len(sources)
    header = f"{META_CAPABILITY_SENTENCE} I have {n} document{'s' if n != 1 else ''} indexed:\n"
    return header + "\n".join(f"- {s}" for s in sources)
