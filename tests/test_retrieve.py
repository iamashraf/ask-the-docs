from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from research_assistant.config import ProviderConfig, Settings
from research_assistant.index import build_index
from research_assistant.retrieve import retrieve


def make_settings(tmp_path) -> Settings:
    return Settings(
        generator=ProviderConfig(provider="ollama", model="llama3.1:8b"),
        judge=ProviderConfig(provider="anthropic", model="claude-sonnet-5-5", api_key="x"),
        index_dir=tmp_path / "index",
    )


VECTORS = {
    "apples are red fruit": [1.0, 0.0],
    "bananas are yellow fruit": [0.9, 0.1],
    "rockets launch into space": [0.0, 1.0],
}


class FakeEmbeddings(Embeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [VECTORS[t] for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return VECTORS[text]


def test_retrieve_returns_empty_list_when_index_not_built(tmp_path, monkeypatch):
    monkeypatch.setattr("research_assistant.index.get_embeddings", lambda settings: FakeEmbeddings())
    settings = make_settings(tmp_path)

    assert retrieve("apples are red fruit", settings) == []


def test_retrieve_returns_top_k_ordered_by_relevance(tmp_path, monkeypatch):
    monkeypatch.setattr("research_assistant.index.get_embeddings", lambda settings: FakeEmbeddings())
    settings = make_settings(tmp_path)

    docs = [
        Document(page_content="apples are red fruit", metadata={"source": "a.txt", "source_doc_id": "a"}),
        Document(
            page_content="bananas are yellow fruit", metadata={"source": "b.txt", "source_doc_id": "b"}
        ),
        Document(
            page_content="rockets launch into space", metadata={"source": "c.txt", "source_doc_id": "c"}
        ),
    ]
    build_index(docs, settings)

    results = retrieve("apples are red fruit", settings, k=2)

    assert len(results) == 2
    assert results[0].page_content == "apples are red fruit"
    assert results[1].page_content == "bananas are yellow fruit"
