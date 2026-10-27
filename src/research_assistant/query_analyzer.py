"""LLM-based query rewrite and local/global classification, run before
retrieval to decide how a user's question should be routed."""

import json
import logging
from dataclasses import dataclass
from typing import Literal

from research_assistant.config import Settings
from research_assistant.llm import get_llm_client

logger = logging.getLogger(__name__)

QUERY_REWRITE_PROMPT_TEMPLATE = """Rewrite the question below into an effective search query for a document retrieval system.

Rules:
- Expand abbreviations and acronyms if you know what they stand for.
- Add likely synonyms or technical terms a relevant document would use.
- Keep it a single query, not a list of alternatives.
- Do not answer the question. Do not add explanation. Output only the rewritten query.

Question: {query}

Rewritten query:"""

QUERY_CLASSIFICATION_PROMPT_TEMPLATE = """Classify the question below as "local" or "global":
- "local": asks about specific named entities or the relationships between them.
- "global": asks for themes, summaries, or synthesis across the whole corpus.

Question: {query}

If "local", list the entities named in the question; if "global", use an empty list.

Respond with ONLY a JSON object of this exact shape:
{{"label": "local" or "global", "entities": ["<string>", ...]}}
"""


def rewrite_query(query: str, settings: Settings) -> str:
    client = get_llm_client(settings.generator)
    return client.generate(QUERY_REWRITE_PROMPT_TEMPLATE.format(query=query)).strip()


@dataclass
class QueryClassification:
    label: Literal["local", "global"]
    entities: list[str]


def classify_query(query: str, settings: Settings) -> QueryClassification:
    """Falls back to label="local", entities=[] if the LLM's response isn't parseable JSON matching the expected shape."""
    client = get_llm_client(settings.generator)
    prompt = QUERY_CLASSIFICATION_PROMPT_TEMPLATE.format(query=query)
    raw = client.generate(prompt)
    try:
        data = json.loads(raw)
        label = data["label"]
        if label not in ("local", "global"):
            raise ValueError(f"unexpected label: {label!r}")
        if not isinstance(data["entities"], list):
            raise TypeError(f"entities is not a list: {data['entities']!r}")
        entities = [str(entity) for entity in data["entities"]]
        return QueryClassification(label=label, entities=entities)
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        logger.warning("Query classification response was invalid JSON or had an unexpected shape; defaulting to local.", exc_info=True)
        return QueryClassification(label="local", entities=[])
