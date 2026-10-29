"""Embeddings abstraction: swappable backends (ollama/huggingface), both
implementing LangChain's own Embeddings interface so they plug directly
into any LangChain-compatible vector store, not just Chroma."""

from functools import lru_cache

import requests
from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer

from ask_the_docs.config import Settings

OLLAMA_EMBED_TIMEOUT_SECONDS = 30
# embed_documents sends every text in the batch as one request, so a flat
# per-call timeout sized for one text isn't enough -- found against a real
# Ollama server, where a 12-text batch took 28.41s. Base covers connection
# + per-request overhead; per-text covers the actual embedding work.
OLLAMA_EMBED_BATCH_BASE_TIMEOUT_SECONDS = 15
OLLAMA_EMBED_BATCH_PER_TEXT_TIMEOUT_SECONDS = 5


class OllamaEmbeddings(Embeddings):
    """Talks to a local Ollama server's /api/embeddings (single) and
    /api/embed (batch) endpoints."""

    def __init__(self, model: str, base_url: str):
        self.model = model
        self.base_url = base_url

    def embed_query(self, text: str) -> list[float]:
        response = requests.post(
            f"{self.base_url}/api/embeddings",
            json={"model": self.model, "prompt": text},
            timeout=OLLAMA_EMBED_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        return response.json()["embedding"]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        timeout = OLLAMA_EMBED_BATCH_BASE_TIMEOUT_SECONDS + OLLAMA_EMBED_BATCH_PER_TEXT_TIMEOUT_SECONDS * len(texts)
        response = requests.post(
            f"{self.base_url}/api/embed",
            json={"model": self.model, "input": texts},
            timeout=timeout,
        )
        response.raise_for_status()
        return response.json()["embeddings"]


class HuggingFaceEmbeddings(Embeddings):
    """Loads a sentence-transformers model directly into this process."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = SentenceTransformer(model_name)

    def embed_query(self, text: str) -> list[float]:
        return [float(x) for x in self._model.encode(text)]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[float(x) for x in vector] for vector in self._model.encode(texts)]


@lru_cache
def _get_embeddings_cached(
    backend: str, ollama_model: str, ollama_base_url: str, huggingface_model: str
) -> Embeddings:
    """Settings itself isn't hashable, so this is keyed on the specific
    primitives that determine the backend -- loaded once per distinct
    config, not once per call (HuggingFaceEmbeddings.__init__ does a real
    model load, same reasoning as the Reranker's own lru_cache)."""
    if backend == "ollama":
        return OllamaEmbeddings(model=ollama_model, base_url=ollama_base_url)
    if backend == "huggingface":
        return HuggingFaceEmbeddings(model_name=huggingface_model)
    raise ValueError(f"Unknown embeddings backend: {backend!r}")


def get_embeddings(settings: Settings) -> Embeddings:
    """Factory: builds the backend matching settings.embeddings_backend."""
    return _get_embeddings_cached(
        settings.embeddings_backend,
        settings.ollama_embedding_model,
        settings.ollama_base_url,
        settings.huggingface_embedding_model,
    )
