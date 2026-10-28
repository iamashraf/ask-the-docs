"""LLM-based triplet extraction: pulls (subject, relation, object) facts out of a chunk of text."""

import json
import logging
import re
from dataclasses import dataclass

from ask_the_docs.config import Settings
from ask_the_docs.llm import get_llm_client

logger = logging.getLogger(__name__)

TRIPLET_EXTRACTION_PROMPT_TEMPLATE = """Extract factual (subject, relation, object) triplets from the text below.

Text:
{chunk_text}

Respond with ONLY a JSON array of objects of this exact shape:
[{{"subject": "<string>", "relation": "<string>", "object": "<string>"}}]
If there are no triplets, respond with an empty array: []
"""


FENCED_BLOCK_PATTERN = re.compile(r"^```(?:json)?\s*\n?(.*?)\n?```$", re.DOTALL)


def _strip_code_fence(raw: str) -> str:
    """Small/local models often wrap JSON responses in markdown code fences; non-str input is left to json.loads to reject."""
    if not isinstance(raw, str):
        return raw
    match = FENCED_BLOCK_PATTERN.match(raw.strip())
    return match.group(1) if match else raw


@dataclass
class Triplet:
    subject: str
    relation: str
    obj: str


def extract_triplets(chunk_text: str, settings: Settings) -> list[Triplet]:
    """Returns [] if the LLM's response isn't parseable JSON matching the triplet shape."""
    client = get_llm_client(settings.generator)
    prompt = TRIPLET_EXTRACTION_PROMPT_TEMPLATE.format(chunk_text=chunk_text)
    raw = client.generate(prompt)
    try:
        data = json.loads(_strip_code_fence(raw))
        return [
            Triplet(subject=str(item["subject"]), relation=str(item["relation"]), obj=str(item["object"]))
            for item in data
        ]
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        # Unreachable here; kept to match evaluate.judge's except tuple.
        logger.warning("Triplet extraction response was invalid JSON or had an unexpected shape; skipping chunk.", exc_info=True)
        return []
