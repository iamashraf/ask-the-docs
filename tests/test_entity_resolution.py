from research_assistant.config import ProviderConfig, Settings
from research_assistant.entity_resolution import resolve_entities
from research_assistant.graph_extract import Triplet


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
    monkeypatch.setattr("research_assistant.entity_resolution.get_embeddings", lambda s: FakeEmbeddings())
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


def test_resolve_entities_merges_transitively_via_bridge_entity(monkeypatch):
    # X~Z and Y~Z clear the threshold (cos = 1/sqrt(2) ~= 0.707) but X~Y (cos = 0)
    # doesn't. A correct union-find still puts all three in one cluster, and the
    # canonical name must be the first-seen member ("X"), not whichever root a
    # naive (non-index-aware) union happens to leave on top.
    class BridgeFakeEmbeddings:
        VECTORS = {
            "X": [1.0, 0.0, 0.0],
            "Y": [0.0, 1.0, 0.0],
            "Z": [1.0, 1.0, 0.0],
            "other": [0.0, 0.0, 1.0],
        }

        def embed_documents(self, texts):
            return [self.VECTORS[t] for t in texts]

    monkeypatch.setattr("research_assistant.entity_resolution.get_embeddings", lambda s: BridgeFakeEmbeddings())
    settings = make_settings(threshold=0.7)
    triplets = [
        Triplet(subject="X", relation="r1", obj="other"),
        Triplet(subject="Y", relation="r2", obj="other"),
        Triplet(subject="Z", relation="r3", obj="other"),
    ]

    result = resolve_entities(triplets, settings)

    assert result[0].subject == result[1].subject == result[2].subject == "X"
