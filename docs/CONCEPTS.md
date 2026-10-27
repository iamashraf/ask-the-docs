# Concepts

What each piece of the pipeline actually is, and where to read the
primary/authoritative source. Pairs with the "Concepts used" summary in
[ARCHITECTURE.md](ARCHITECTURE.md) — this doc goes one level deeper.

## Vector (dense) retrieval

Embed the query and every chunk into the same vector space with a model
trained so that semantically similar text ends up close together; rank
chunks by similarity (`index.py`, Chroma). Catches paraphrase and
synonymy a keyword match would miss, but can miss exact identifiers.

- [Karpukhin et al., 2020 — Dense Passage Retrieval for Open-Domain Question Answering (arXiv:2004.04906)](https://arxiv.org/abs/2004.04906) — the paper that established the dense dual-encoder approach this style of retrieval is based on.

## BM25 (sparse/keyword retrieval)

A bag-of-words ranking function: score = sum over query terms of an
IDF-weighted term-frequency score, saturating as a term repeats and
normalized by document length (`bm25_index.py`). Catches exact
terms/acronyms/identifiers vector search can miss; blind to paraphrase.

- [Okapi BM25 — Wikipedia](https://en.wikipedia.org/wiki/Okapi_BM25) — the formula and parameters (`k1`, `b`) in plain terms.
- [Robertson & Zaragoza, 2009 — The Probabilistic Relevance Framework: BM25 and Beyond](https://dx.doi.org/10.1561/1500000019) — the full theoretical treatment, for going deeper than the formula.

## Hybrid fusion

Combining a lexical (BM25) and a dense (vector) ranking into one. The
standard technique is Reciprocal Rank Fusion (RRF) — fuse by rank
position, not raw score, because BM25's score is unbounded/corpus-dependent
while cosine similarity is bounded, so they can't be averaged directly.
This project's `hybrid_merge()` (`retrieve.py`) instead min-max
normalizes each side's *scores* to `[0, 1]` first, then blends
`alpha * vector + (1 - alpha) * bm25` — a scored variant rather than
rank-only RRF, chosen so `alpha` gives a direct, tunable knob.

- [Weaviate Academy — Hybrid Search](https://docs.weaviate.io/academy/py/zero_to_mvp/queries_2/hybrid) — explains why raw scores don't mix and walks through the rank-based fix this project's approach deliberately departs from.

## Query rewriting

Before retrieval, ask an LLM to rewrite the user's raw question into a
better search query — expanding acronyms, adding likely synonyms
(`query_analyzer.py`). Distinct from generation-time prompting: this
rewrite only ever reaches the retriever, never the final answer.

- [Ma et al., 2023 — Query Rewriting for Retrieval-Augmented Large Language Models (arXiv:2305.14283)](https://arxiv.org/abs/2305.14283) — the "Rewrite-Retrieve-Read" framing this stage implements a simplified (prompted, not trained) version of.

## Cross-encoder reranking

A bi-encoder (vector search) embeds the query and each document
*independently*; a cross-encoder instead feeds `(query, document)` into
one model *together*, so it can weigh their interaction directly. More
accurate, much more expensive per pair — so it only runs on the already
-narrowed hybrid candidate pool, not the whole corpus (`reranker.py`,
`cross-encoder/ms-marco-MiniLM-L-6-v2`).

- [SBERT — Cross Encoder usage](https://sbert.net/docs/cross_encoder/usage/usage.html) — the library and model class this project calls directly.

## Citation-grounded generation

The generator is instructed to cite `[n]` markers tied to specific
retrieved chunks; a deterministic guardrail (`answer.py:
citation_guardrail_passes`) — not another LLM call — rejects any answer
with no citations or a citation number that doesn't exist, refusing
rather than retrying. This is the project's answer to the open problem
the original RAG paper names explicitly: giving a generator's output
traceable provenance.

- [Lewis et al., 2020 — Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks (arXiv:2005.11401)](https://arxiv.org/abs/2005.11401) — the paper that introduced pairing a generator with a retrieved, non-parametric document index.

## Graph RAG (local/global search)

Build a knowledge graph alongside the vector/BM25 index, then route each
query down whichever path fits the question. At ingest, an LLM extracts
`(subject, relation, object)` triplets per chunk (`graph_extract.py`);
entity mentions are resolved to canonical nodes by embedding-similarity
clustering, not a second LLM call (`entity_resolution.py`) — cheaper,
faster, and deterministic, at the cost of weaker handling of acronym/
synonym variants than an LLM resolver would give. Resolved triplets are
written to Neo4j (`graph_store.py`), then Neo4j GDS's Leiden algorithm
(an undirected graph projection, required by the algorithm) partitions
entities into communities (`community.py`), each summarized once by an
LLM.

At query time, an LLM classifies the question **local** (asks about
specific named entities/relationships) or **global** (asks for themes or
synthesis across the corpus) (`query_analyzer.classify_query`). A local
query traverses the graph a fixed number of hops from the named entities
and merges the chunks those entities link back to into the existing
hybrid candidate pool, so they compete for a spot in the final reranked
answer (`graph_retrieve.local_search`). A global query instead runs a
map step — scoring every community summary's relevance to the question
with an LLM call per community — and returns the top-scored summaries
directly as citable context, skipping hybrid/rerank entirely
(`graph_retrieve.global_search`): this is the published GraphRAG
technique's map-reduce idea, simplified to a map step feeding this
project's existing citation-grounded generator rather than a separate
reduce LLM call.

Every Neo4j read/write and every LLM call on this path (classification,
extraction, summarization, relevance scoring) degrades independently —
a failure logs a warning and falls back to the plain hybrid result,
never raising into a live request.

- [Edge et al., 2024 — From Local to Global: A Graph RAG Approach to Query-Focused Summarization (arXiv:2404.16130)](https://arxiv.org/abs/2404.16130) — the paper this local/global, community-summary design is based on.
- [Traag, Waltman & van Eck, 2019 — From Louvain to Leiden: guaranteeing well-connected communities (arXiv:1810.08473)](https://arxiv.org/abs/1810.08473) — why Leiden is used instead of the more commonly-cited Louvain algorithm it improves on.
- [Neo4j GDS — Leiden algorithm](https://neo4j.com/docs/graph-data-science/current/algorithms/leiden/) — the in-database implementation this project calls directly; notably requires an undirected graph projection.

## LLM-as-judge evaluation

Score generated answers on accuracy/relevance/completeness using a
separately-configured model as the grader (`evaluate.py`), instead of
hand-written metrics — so retrieval/config changes (hybrid alpha,
rerank on/off, rewrite on/off) can be compared against a fixed golden
question set.

- [Zheng et al., 2023 — Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena (arXiv:2306.05685)](https://arxiv.org/abs/2306.05685) — measures how well an LLM judge's grades track human preference, and the biases (position, verbosity, self-enhancement) to watch for.
