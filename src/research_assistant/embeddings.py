"""Embeddings abstraction: swappable backends (ollama/huggingface), both
implementing LangChain's own Embeddings interface so they plug directly
into any LangChain-compatible vector store, not just Chroma."""

import requests
from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer

from research_assistant.config import Settings

OLLAMA_EMBED_TIMEOUT_SECONDS = 30


class OllamaEmbeddings(Embeddings):
    """Talks to a local Ollama server's /api/embeddings endpoint."""

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
        return [self.embed_query(text) for text in texts]


class HuggingFaceEmbeddings(Embeddings):
    """Loads a sentence-transformers model directly into this process."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = SentenceTransformer(model_name)

    def embed_query(self, text: str) -> list[float]:
        return [float(x) for x in self._model.encode(text)]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[float(x) for x in vector] for vector in self._model.encode(texts)]


def get_embeddings(settings: Settings) -> Embeddings:
    """Factory: builds the backend matching settings.embeddings_backend."""
    if settings.embeddings_backend == "ollama":
        return OllamaEmbeddings(model=settings.ollama_embedding_model, base_url=settings.ollama_base_url)
    if settings.embeddings_backend == "huggingface":
        return HuggingFaceEmbeddings(model_name=settings.huggingface_embedding_model)
    raise ValueError(f"Unknown embeddings backend: {settings.embeddings_backend!r}")
