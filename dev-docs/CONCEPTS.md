# Concepts

What each piece of the pipeline is, and where to read the primary source.
This doc goes deeper than the "Concepts used" summary in
[ARCHITECTURE.md](ARCHITECTURE.md).

## Vector (dense) retrieval

Embed the query and every chunk into the same vector space with a model
trained to place semantically similar text close together. Then rank chunks by
similarity (`index.py`, Chroma). This catches paraphrase and synonymy that a
keyword match misses, but it can miss exact identifiers.

- [Karpukhin et al., 2020: Dense Passage Retrieval for Open-Domain Question Answering (arXiv:2004.04906)](https://arxiv.org/abs/2004.04906): the paper that established the dense dual-encoder approach.

## BM25 (sparse/keyword retrieval)

A bag-of-words ranking function. The score is the sum over query terms of an
IDF-weighted term-frequency score. The score saturates as a term repeats and
is normalized by document length (`bm25_index.py`, `bm25s` library with
Lucene-style IDF). Search returns only the top `candidate_pool_size` hits.
BM25 catches exact terms, acronyms, and identifiers that vector search can
miss. It is blind to paraphrase.

- [Okapi BM25: Wikipedia](https://en.wikipedia.org/wiki/Okapi_BM25): the formula and parameters (`k1`, `b`) in plain terms.
- [Robertson & Zaragoza, 2009: The Probabilistic Relevance Framework: BM25 and Beyond](https://dx.doi.org/10.1561/1500000019): the full theoretical treatment.

## Hybrid fusion

Combining a lexical (BM25) ranking and a dense (vector) ranking into one.
The standard technique is Reciprocal Rank Fusion (RRF). It fuses by rank
position, not raw score. BM25's score is unbounded and depends on the corpus,
while cosine similarity is bounded, so the two cannot be averaged directly.

This project's `hybrid_merge()` (`retrieve.py`) instead min-max normalizes
each side's *scores* to `[0, 1]`, then blends
`alpha * vector + (1 - alpha) * bm25`. This scored variant departs from
rank-only RRF so that `alpha` is a direct, tunable parameter.

- [Weaviate Academy: Hybrid Search](https://docs.weaviate.io/academy/py/zero_to_mvp/queries_2/hybrid): explains why raw scores do not mix and walks through the rank-based fix this project departs from.

## Intent routing

Classify each query before retrieval runs. A bare greeting, thanks, or
goodbye matches a free rules-based check (`match_smalltalk`, `router.py`).
Anything else goes through one LLM call (`analyze`, `router.py`). That
call returns four fields. The first is `intent`, one of `doc`, `chitchat`, or
`meta`. The others are `standalone_query`, `search_query`, and `reply`.
`search_query` is `standalone_query` expanded with synonyms, and only hybrid
retrieval uses it.

Routing decides what *kind* of input the query is, not whether the documents
can answer it. An off-topic question such as "how does CRISPR work?" is still
classified `doc`. It reaches retrieval and gets a refusal there if nothing
relevant comes back, instead of being redirected at the routing stage. `meta`
is for questions about the document collection or the assistant's own
capabilities. `build_meta_reply` builds its reply deterministically from the
list of indexed sources, and the LLM does not generate it. If listing the
sources fails, the question falls back to the `doc` path with the resolved
queries.

For `doc`, the same call rewrites the query into `standalone_query`. It
resolves follow-up references from conversation history and expands
abbreviations and synonyms. A follow-up like "what about the second one?"
becomes a self-contained search query. The generator receives the user's
original wording next to the resolved question, so style requests such as "in
bullets" are kept.

Routing runs before retrieval instead of thresholding a retrieval score
afterward. Chitchat and meta queries then never touch the vector, BM25, or
rerank pipeline.

If the `analyze` call fails, the query falls back to `doc` with the raw query
text. Disabling routing (`query_rewrite_enabled=False`) produces the same
fallback.

- [Ma et al., 2023: Query Rewriting for Retrieval-Augmented Large Language Models (arXiv:2305.14283)](https://arxiv.org/abs/2305.14283): the "Rewrite-Retrieve-Read" framing. The `doc`-path rewrite inside `analyze` implements a simplified version (prompted, not trained).

## Cross-encoder reranking

A bi-encoder (vector search) embeds the query and each document
*independently*. A cross-encoder feeds `(query, document)` into one model
*together*, so it weighs their interaction directly. It is more accurate but
much more expensive per pair. It runs only on the already narrowed hybrid
candidate pool, not the whole corpus (`reranker.py`,
`cross-encoder/ms-marco-MiniLM-L-6-v2`).

- [SBERT: Cross Encoder usage](https://sbert.net/docs/cross_encoder/usage/usage.html): the library and model class this project calls directly.

## Citation-grounded generation

The prompt tells the generator to cite `[n]` markers tied to specific
retrieved chunks. A deterministic guardrail (`answer.py:
citation_guardrail_passes`), not another LLM call, rejects any answer with no
citations or a citation number that does not exist. The pipeline retries a
rejected answer once with a corrective note, then refuses. The check confirms
the markers exist and are in range. It does not confirm the cited chunk
supports the claim. This addresses the open problem the original RAG paper
names: giving a generator's output traceable provenance.

Retrieved text is also untrusted input. A document can contain a sentence such
as "ignore your instructions", and a model may follow it. This is called
indirect prompt injection, because the attacker's text arrives through the
documents, not through the user's question. The prompt separates the two.
Chunks sit inside `<documents>` tags, the question sits inside `<question>`
tags, and the system prompt says the document text is data. The project
escapes any such tag inside a chunk, so a document cannot close the block
early. This reduces the risk and does not remove it.

- [Lewis et al., 2020: Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks (arXiv:2005.11401)](https://arxiv.org/abs/2005.11401): the paper that introduced pairing a generator with a retrieved, non-parametric document index.

## Graph RAG (local/global search)

Build a knowledge graph alongside the vector and BM25 index, then route each
query down the path that fits the question. At ingest, an LLM extracts
`(subject, relation, object)` triplets per chunk (`graph_extract.py`).
Embedding-similarity clustering, not a second LLM call, resolves entity
mentions to canonical nodes (`entity_resolution.py`). This is cheaper, faster,
and deterministic. The cost is weaker handling of acronym and synonym variants
than an LLM resolver would give. The pipeline writes resolved triplets to Neo4j
(`graph_store.py`). Neo4j GDS's Leiden algorithm (on an undirected graph
projection, which the algorithm requires) then partitions entities into
communities (`community.py`), and an LLM summarizes each once.

At query time, an LLM classifies the question as **local** (asks about
specific named entities or relationships) or **global** (asks for themes or
synthesis across the corpus) (`query_analyzer.classify_query`). A local query
traverses the graph a fixed number of hops from the named entities. It merges
the chunks those entities link back to into the existing hybrid candidate
pool, so they compete for a spot in the final reranked answer
(`graph_retrieve.local_search`).

A global query runs a map step instead. One LLM call per community scores that
community summary's relevance to the question. The top-scored summaries go
straight to the generator as citable context, skipping hybrid retrieval and
rerank (`graph_retrieve.global_search`). This simplifies the map-reduce idea
of the published GraphRAG technique. The map step feeds this project's
existing citation-grounded generator, with no separate reduce LLM call.

Every Neo4j read or write and every LLM call on this path (classification,
extraction, summarization, relevance scoring) degrades independently. A
failure logs a warning and falls back to the plain hybrid result. It never
raises into a live request. A global search where no community scores above 0,
or every scoring call fails, returns nothing, so retrieval falls back to the
hybrid path.

Local search includes the start entities' own chunks and orders entities by
hop distance. Entity resolution compares each entity only against the
first-seen entity of each cluster, so A~B and B~C never chain A with C. The
pairwise comparison is O(n^2) in the worst case. The build status (`ok`,
`skipped` or `failed: <ExceptionClass>`) appears as `graph` in the job result,
and a graph failure never fails the vector index job. The extraction, summary,
and scoring prompts fence chunk text, entity names, queries, and summaries as
tagged data, and escape closing tags inside them.

Graph indexing is incremental. When a source is new, replaced, or re-ingested
with no chunks, `build_index` tells `build_graph` which sources changed. The
graph drops their facts (`chunk_refs` on entities and `RELATED_TO` edges, and
any entity or edge left with none) and extracts only the new chunks. It
resolves new entities against the existing entity names, and existing names
win as canonical. Community detection and summaries then re-run over the whole
graph. An unchanged re-ingest does no extraction.

A full wipe and rebuild from all chunks happens in three cases:

- The graph is empty.
- Old edges have no `chunk_refs`. The first ingest after upgrading re-tags
  them.
- The incremental update fails before it writes.

`ask-the-docs index --rebuild-graph` (or `build_index(...,
full_graph_rebuild=True)`) forces it. The API does not. Limits:

- Chunks whose extraction failed earlier are not retried on an unchanged re-ingest.
- Incremental and full rebuild can resolve entities slightly differently (first-seen cluster rule).
- Every changed ingest still embeds all existing entity names and re-runs Leiden over the whole graph.

Two Valkey caches cut repeat LLM calls. Extracted triplets are cached per
chunk, keyed on chunk text, extraction prompt, and model, with a 30 day TTL.
Unparseable or failed extractions are not cached. Community summaries are
cached keyed on the sorted member names, prompt, and model. Leiden community
ids are unstable, so a summary is reused only when membership is identical. If
Valkey is down, extraction and summarization run as before.

- [Edge et al., 2024: From Local to Global: A Graph RAG Approach to Query-Focused Summarization (arXiv:2404.16130)](https://arxiv.org/abs/2404.16130): the paper this local/global, community-summary design is based on.
- [Traag, Waltman & van Eck, 2019: From Louvain to Leiden: guaranteeing well-connected communities (arXiv:1810.08473)](https://arxiv.org/abs/1810.08473): why this project uses Leiden instead of the Louvain algorithm it improves on.
- [Neo4j GDS: Leiden algorithm](https://neo4j.com/docs/graph-data-science/current/algorithms/leiden/): the in-database implementation this project calls directly. It requires an undirected graph projection.

## LLM-as-judge evaluation

Score generated answers on accuracy, relevance, and completeness, using a
separately configured model as the grader (`evaluate.py`) instead of
hand-written metrics. This lets you compare retrieval and config changes
(hybrid alpha, rerank on or off, rewrite on or off) against a fixed golden
question set.

- [Zheng et al., 2023: Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena (arXiv:2306.05685)](https://arxiv.org/abs/2306.05685): measures how well an LLM judge's grades track human preference, and the biases (position, verbosity, self-enhancement) to watch for.

The project also uses it to compare the generator itself (Ollama vs. hosted) on
latency, accuracy, and grounding: [MODEL_COMPARISON.md](MODEL_COMPARISON.md).

## Routing evaluation

A labeled case set (`evals/routing_cases.yaml`) runs every case through
`route()` and compares the returned intent to the expected one, grouped
by category (chitchat, meta, doc, off-topic-doc, mixed, follow-up,
adversarial).

Misroutes are not scored symmetrically. A case expected to be `doc` that
routes to anything else is "critical", because it silently drops a real
question instead of answering it. A `chitchat` or `meta` case that routes to
`doc` only costs an extra retrieval pass, not a lost answer. `eval-routing`
exits non-zero only on a critical misroute.

## Retrieval evaluation (hit@1 / hit@k / MRR)

A labeled case set (`evals/retrieval_cases.yaml`, 30 cases, two per each
of the 15 seeded arXiv papers) runs every question through `retrieve()`
and checks whether a chunk from the expected source paper appears in the
results.

- **hit@1.** The expected source is the top-ranked result.
- **hit@k.** The expected source appears anywhere in the top k.
- **MRR** (mean reciprocal rank). The average of `1/rank` over cases where
  the expected source appears at all. A correct result that ranks higher
  scores more than one that barely makes the cutoff.

`eval-retrieval` calls `retrieve()` directly and bypasses routing and its
query rewrite. It measures retrieval alone, not the full ask path. It exits
non-zero if hit@k falls below `--min-hit-at-k` (default 0.9).
