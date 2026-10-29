from types import SimpleNamespace

import requests

from ask_the_docs.config import ProviderConfig, Settings
from ask_the_docs.embeddings import (
    OLLAMA_EMBED_BATCH_BASE_TIMEOUT_SECONDS,
    OLLAMA_EMBED_BATCH_PER_TEXT_TIMEOUT_SECONDS,
    OLLAMA_EMBED_TIMEOUT_SECONDS,
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


def test_ollama_embeddings_embed_documents_calls_api_once_with_all_texts(monkeypatch):
    captured = {}

    def fake_post(url, json, timeout):
        captured["url"] = url
        captured["json"] = json
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: {"embeddings": [[1.0], [2.0]]})

    monkeypatch.setattr(requests, "post", fake_post)

    emb = OllamaEmbeddings(model="nomic-embed-text", base_url="http://localhost:11434")
    assert emb.embed_documents(["a", "b"]) == [[1.0], [2.0]]
    assert captured["url"] == "http://localhost:11434/api/embed"
    assert captured["json"] == {"model": "nomic-embed-text", "input": ["a", "b"]}


def test_ollama_embeddings_embed_documents_of_empty_list_skips_the_request(monkeypatch):
    def fail_if_called(*args, **kwargs):
        raise AssertionError("must not call the API for an empty text list")

    monkeypatch.setattr(requests, "post", fail_if_called)

    emb = OllamaEmbeddings(model="nomic-embed-text", base_url="http://localhost:11434")
    assert emb.embed_documents([]) == []


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


def test_ollama_embeddings_embed_documents_scales_timeout_with_batch_size(monkeypatch):
    """Reproduces a real failure found against a live Ollama server: a
    12-text batch took 28.41s against the old flat 30s timeout (sized for
    a single text, from before batching existed) -- right at the edge of
    timing out. The timeout must grow with the batch size actually sent."""
    captured = {}

    def fake_post(url, json, timeout):
        captured["timeout"] = timeout
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: {"embeddings": [[1.0]] * len(json["input"])})

    monkeypatch.setattr(requests, "post", fake_post)

    emb = OllamaEmbeddings(model="nomic-embed-text", base_url="http://localhost:11434")
    emb.embed_documents(["a"] * 12)

    assert captured["timeout"] == OLLAMA_EMBED_BATCH_BASE_TIMEOUT_SECONDS + OLLAMA_EMBED_BATCH_PER_TEXT_TIMEOUT_SECONDS * 12
    assert captured["timeout"] > 28.41


def test_ollama_embeddings_embed_query_timeout_is_unaffected_by_batch_timeout_change(monkeypatch):
    captured = {}

    def fake_post(url, json, timeout):
        captured["timeout"] = timeout
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: {"embedding": [0.1]})

    monkeypatch.setattr(requests, "post", fake_post)

    emb = OllamaEmbeddings(model="nomic-embed-text", base_url="http://localhost:11434")
    emb.embed_query("hello")

    assert captured["timeout"] == OLLAMA_EMBED_TIMEOUT_SECONDS
