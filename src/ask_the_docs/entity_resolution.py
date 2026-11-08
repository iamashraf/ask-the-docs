"""Entity resolution: merges near-duplicate entity names across triplets into one canonical name via embedding similarity."""

import math
from dataclasses import replace

from ask_the_docs.config import Settings
from ask_the_docs.embeddings import get_embeddings
from ask_the_docs.graph_extract import Triplet


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
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

    threshold = settings.entity_resolution_similarity_threshold
    canonical_indices: list[int] = []
    canonical_of: list[int] = []
    for i in range(len(names)):
        # Compared to each cluster's first-seen member only, so A~B and B~C never chain A with C.
        # shortcut: pairwise comparison is still O(n * clusters); switch to an ANN index for large corpora.
        match = next((c for c in canonical_indices if _cosine_similarity(vectors[c], vectors[i]) >= threshold), i)
        if match == i:
            canonical_indices.append(i)
        canonical_of.append(match)

    name_to_canonical = {name: names[canonical_of[i]] for i, name in enumerate(names)}

    return [
        replace(triplet, subject=name_to_canonical[triplet.subject], obj=name_to_canonical[triplet.obj])
        for triplet in triplets
    ]
