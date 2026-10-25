"""LLM-based query rewrite: normalizes a user's question into a more
effective search query before it hits vector/BM25 retrieval."""

from research_assistant.config import Settings
from research_assistant.llm import get_llm_client

QUERY_REWRITE_PROMPT_TEMPLATE = """Rewrite the question below into an effective search query for a document retrieval system.

Rules:
- Expand abbreviations and acronyms if you know what they stand for.
- Add likely synonyms or technical terms a relevant document would use.
- Keep it a single query, not a list of alternatives.
- Do not answer the question. Do not add explanation. Output only the rewritten query.

Question: {query}

Rewritten query:"""


def rewrite_query(query: str, settings: Settings) -> str:
    client = get_llm_client(settings.generator)
    return client.generate(QUERY_REWRITE_PROMPT_TEMPLATE.format(query=query)).strip()
