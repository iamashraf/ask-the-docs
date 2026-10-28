from types import SimpleNamespace

import requests

from ask_the_docs.config import ProviderConfig, Settings
from ask_the_docs.embeddings import (
    HuggingFaceEmbeddings,
    OllamaEmbeddings,
    _get_embeddings_cached,
    get_embeddings,
)


def make_settings(embeddings_backend: str) -> Settings:
    return Settings(
        generator=ProviderConfig(provider="ollama", model="llama3.1:8b"),
        judge=ProviderConfig(provider="anthropic", model="claude-sonnet-5-5", api_key="x"),
        embeddings_backend=embeddings_backend,
    )


def test_get_embeddings_returns_ollama_backend():
    assert isinstance(get_embeddings(make_settings("ollama")), OllamaEmbeddings)


def test_get_embeddings_returns_huggingface_backend(monkeypatch):
    monkeypatch.setattr("ask_the_docs.embeddings.SentenceTransformer", lambda model_name: object())
    assert isinstance(get_embeddings(make_settings("huggingface")), HuggingFaceEmbeddings)


def test_get_embeddings_loads_huggingface_model_only_once(monkeypatch):
    load_count = {"count": 0}

    class CountingModel:
        def __init__(self, model_name):
            load_count["count"] += 1

    monkeypatch.setattr("ask_the_docs.embeddings.SentenceTransformer", CountingModel)
    _get_embeddings_cached.cache_clear()
    settings = make_settings("huggingface")

    a = get_embeddings(settings)
    b = get_embeddings(settings)

    assert a is b
    assert load_count["count"] == 1


def test_ollama_embeddings_embed_query_calls_api(monkeypatch):
    captured = {}

    def fake_post(url, json, timeout):
        captured["url"] = url
        captured["json"] = json
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: {"embedding": [0.1, 0.2]})

    monkeypatch.setattr(requests, "post", fake_post)

    emb = OllamaEmbeddings(model="nomic-embed-text", base_url="http://localhost:11434")
    assert emb.embed_query("hello") == [0.1, 0.2]
    assert captured["url"] == "http://localhost:11434/api/embeddings"
    assert captured["json"] == {"model": "nomic-embed-text", "prompt": "hello"}


def test_ollama_embeddings_embed_documents_calls_api_per_text(monkeypatch):
    calls = []

    def fake_post(url, json, timeout):
        calls.append(json["prompt"])
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: {"embedding": [1.0]})

    monkeypatch.setattr(requests, "post", fake_post)

    emb = OllamaEmbeddings(model="nomic-embed-text", base_url="http://localhost:11434")
    assert emb.embed_documents(["a", "b"]) == [[1.0], [1.0]]
    assert calls == ["a", "b"]


def test_huggingface_embeddings_embed_query_uses_model_encode(monkeypatch):
    class FakeModel:
        def encode(self, text):
            return [0.5, 0.25]

    monkeypatch.setattr(
        "ask_the_docs.embeddings.SentenceTransformer", lambda model_name: FakeModel()
    )

    emb = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    assert emb.embed_query("hello") == [0.5, 0.25]


def test_huggingface_embeddings_embed_documents_uses_model_encode(monkeypatch):
    class FakeModel:
        def encode(self, texts):
            return [[0.1, 0.2], [0.3, 0.4]]

    monkeypatch.setattr(
        "ask_the_docs.embeddings.SentenceTransformer", lambda model_name: FakeModel()
    )

    emb = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    assert emb.embed_documents(["a", "b"]) == [[0.1, 0.2], [0.3, 0.4]]
