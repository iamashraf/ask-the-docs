from ask_the_docs.config import ProviderConfig, Settings
from ask_the_docs.entity_resolution import resolve_entities
from ask_the_docs.graph_extract import Triplet


def make_settings(threshold: float = 0.85) -> Settings:
    return Settings(
        generator=ProviderConfig(provider="ollama", model="llama3.1:8b"),
        judge=ProviderConfig(provider="anthropic", model="claude-sonnet-5-5", api_key="x"),
        entity_resolution_similarity_threshold=threshold,
    )


class FakeEmbeddings:
    VECTORS = {
        "Transformer": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "the transformer architecture": [0.99, 0.01, 0.0, 0.0, 0.0, 0.0],
        "attention": [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        "paper X": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        "BM25": [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        "keyword ranking function": [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    }

    def embed_documents(self, texts):
        return [self.VECTORS[t] for t in texts]


def test_resolve_entities_merges_near_duplicate_names(monkeypatch):
    monkeypatch.setattr("ask_the_docs.entity_resolution.get_embeddings", lambda s: FakeEmbeddings())
    settings = make_settings()
    triplets = [
        Triplet(subject="Transformer", relation="uses", obj="attention"),
        Triplet(subject="the transformer architecture", relation="introduced-in", obj="paper X"),
        Triplet(subject="BM25", relation="is-a", obj="keyword ranking function"),
    ]

    result = resolve_entities(triplets, settings)

    assert result[0].subject == result[1].subject  # merged into one canonical name
    assert result[0].subject == "Transformer"  # canonical name is the first-seen member
    assert result[2].subject == "BM25"  # distinct, untouched
    assert len(result) == len(triplets)


class _VectorEmbeddings:
    def __init__(self, vectors):
        self._vectors = vectors

    def embed_documents(self, texts):
        return [self._vectors[t] for t in texts]


def test_resolve_entities_does_not_chain_merges_through_a_bridge_entity(monkeypatch):
    # A~B and B~C clear the threshold (cos ~= 0.707) but A!~C (cos = 0): C must not
    # join A's cluster through B; each entity is compared to a cluster's first-seen member only.
    vectors = {"A": [1.0, 0.0], "B": [1.0, 1.0], "C": [0.0, 1.0], "other": [-1.0, -1.0]}
    monkeypatch.setattr("ask_the_docs.entity_resolution.get_embeddings", lambda s: _VectorEmbeddings(vectors))
    triplets = [
        Triplet(subject="A", relation="r1", obj="other"),
        Triplet(subject="B", relation="r2", obj="other"),
        Triplet(subject="C", relation="r3", obj="other"),
    ]

    result = resolve_entities(triplets, make_settings(threshold=0.7))

    assert [t.subject for t in result] == ["A", "A", "C"]


def test_resolve_entities_treats_a_zero_vector_as_dissimilar_instead_of_raising(monkeypatch):
    vectors = {"A": [1.0, 0.0], "zero": [0.0, 0.0], "B": [0.0, 1.0]}
    monkeypatch.setattr("ask_the_docs.entity_resolution.get_embeddings", lambda s: _VectorEmbeddings(vectors))
    triplets = [Triplet(subject="A", relation="r", obj="zero"), Triplet(subject="B", relation="r", obj="zero")]

    result = resolve_entities(triplets, make_settings())

    assert [(t.subject, t.obj) for t in result] == [("A", "zero"), ("B", "zero")]
