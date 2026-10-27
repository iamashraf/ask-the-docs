"""LLM-based triplet extraction: pulls (subject, relation, object) facts out of a chunk of text."""

import json
import logging
from dataclasses import dataclass

from research_assistant.config import Settings
from research_assistant.llm import get_llm_client

logger = logging.getLogger(__name__)

TRIPLET_EXTRACTION_PROMPT_TEMPLATE = """Extract factual (subject, relation, object) triplets from the text below.

Text:
{chunk_text}

Respond with ONLY a JSON array of objects of this exact shape:
[{{"subject": "<string>", "relation": "<string>", "object": "<string>"}}]
If there are no triplets, respond with an empty array: []
"""


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
        data = json.loads(raw)
        return [
            Triplet(subject=str(item["subject"]), relation=str(item["relation"]), obj=str(item["object"]))
            for item in data
        ]
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        # ValueError is unreachable through this function's own logic (no float()/int()
        # calls like evaluate.judge's), but kept for shape-consistency with judge's except
        # tuple -- json.JSONDecodeError already subclasses ValueError, so it's harmless.
        logger.warning("Triplet extraction response was invalid JSON or had an unexpected shape; skipping chunk.", exc_info=True)
        return []
