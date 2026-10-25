"""Top-k vector similarity retrieval."""

from langchain_core.documents import Document

from research_assistant.config import Settings
from research_assistant.index import load_index


def retrieve(query: str, settings: Settings, k: int = 5) -> list[Document]:
    """Up to k chunks most similar to query; [] if the index has no chunks yet."""
    index = load_index(settings)
    return index.similarity_search(query, k=k)
