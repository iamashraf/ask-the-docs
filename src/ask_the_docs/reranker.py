"""Cross-encoder reranking: narrows the merged hybrid candidate pool down
to the final top-k, by scoring (query, candidate) pairs jointly rather
than comparing independent embeddings."""

from functools import lru_cache

from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class Reranker:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = CrossEncoder(model_name)

    def rerank(self, query: str, candidates: list[Document], k: int) -> list[Document]:
        pairs = [(query, doc.page_content) for doc in candidates]
        scores = self._model.predict(pairs)
        ranked = sorted(zip(candidates, scores), key=lambda pair: pair[1], reverse=True)
        return [doc for doc, _ in ranked[:k]]


@lru_cache
def get_reranker(model_name: str) -> Reranker:
    """Loaded once per model name, not once per query -- the model load
    is real, measurable cost on a long-running server."""
    return Reranker(model_name)
