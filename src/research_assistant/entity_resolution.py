"""Entity resolution: merges near-duplicate entity names across triplets into one canonical name via embedding similarity."""

import math
from dataclasses import replace

from research_assistant.config import Settings
from research_assistant.embeddings import get_embeddings
from research_assistant.graph_extract import Triplet


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    return dot / (norm_a * norm_b)


def resolve_entities(triplets: list[Triplet], settings: Settings) -> list[Triplet]:
    names: list[str] = []
    seen: set[str] = set()
    for triplet in triplets:
        for name in (triplet.subject, triplet.obj):
            if name not in seen:
                seen.add(name)
                names.append(name)

    vectors = get_embeddings(settings).embed_documents(names)

    parent = list(range(len(names)))

    def find(i: int) -> int:
        while parent[i] != i:
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        root_i, root_j = find(i), find(j)
        if root_i == root_j:
            return
        # Merge by index, not rank/size -- canonical-name lookup relies on root == first-seen index.
        if root_i < root_j:
            parent[root_j] = root_i
        else:
            parent[root_i] = root_j

    threshold = settings.entity_resolution_similarity_threshold
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            if _cosine_similarity(vectors[i], vectors[j]) >= threshold:
                union(i, j)

    name_to_canonical = {name: names[find(i)] for i, name in enumerate(names)}

    return [
        replace(triplet, subject=name_to_canonical[triplet.subject], obj=name_to_canonical[triplet.obj])
        for triplet in triplets
    ]
