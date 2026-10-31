# Usage Guide

Full command reference: what each operation does internally, how to run
it from the CLI or the HTTP API, and how to verify it's working. See
[README.md](README.md) for the quickstart and [CLAUDE.md](CLAUDE.md) for
architecture.

## Prerequisites

- Podman (or Docker) with `compose` support.
- An Anthropic API key, only if you plan to run `evaluate` (the judge
  defaults to Claude; the generator defaults to local Ollama, which needs
  no key).
- The external `llm-observability` Docker/Podman network — `app`, `worker`,
  and `web` join it, and Compose refuses to start them without it. Create
  it once with `podman network create llm-observability`. To also collect
  traces/metrics/logs, start the sibling `llm-observability` repo's stack
  first (see its README's "Start" section); without it, the app runs
  normally and the OTLP exporters just retry/warn.

## Setup (one-time)

```bash
cp .env.example .env.local
```

`.env.example` is a committed reference file; `.env.local` is what's
actually loaded (git-ignored — never commit it). Edit `.env.local` to set
`JUDGE__API_KEY` if you'll run `evaluate`; the other defaults already
point at the right service names for the containerized setup.

```bash
podman compose up -d ollama-init   # starts Ollama, waits until healthy,
                                    # pulls llama3.1:8b + nomic-embed-text
podman compose build app && podman image prune -f
```

Model pulls only need to happen once — re-running `ollama-init` later is a
fast no-op (`ollama pull` skips already-present models).
`podman image prune -f` reclaims the disk space the multi-stage build's
intermediate layer leaves behind, at the cost of a slightly slower next
build.

After any later code change, you don't need to re-run `build` as a
separate step — add `--build` to whichever command you run next instead
(`podman compose run --build app ...` or `podman compose up -d --build
app`). It rebuilds only what changed (Docker/Podman's layer cache skips
the rest), so it stays fast when nothing relevant changed.

**First server start seeds a reading list automatically.** The first time
`podman compose up -d app` runs (CLI's `ingest`/`index` don't trigger
this), it ingests every source in `seed_sources.yaml` — a curated set of
foundational and recent LLM/RAG papers — so there's already real content
to `ask` against without any manual ingestion. This takes noticeably
longer than later starts (15 documents to fetch, chunk, and embed); a
marker file in `data/index/` means it only happens once per fresh index,
not on every restart. Edit `seed_sources.yaml` to change what gets seeded.

What's seeded (see `seed_sources.yaml` for the source URLs):

| Topic | Paper |
|---|---|
| Core architecture | Attention Is All You Need |
| Core architecture | Language Models are Few-Shot Learners (GPT-3) |
| RAG | Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks |
| Alignment | Training LMs to Follow Instructions with Human Feedback (InstructGPT/RLHF) |
| Alignment | LoRA: Low-Rank Adaptation of Large Language Models |
| Reasoning/prompting | Chain-of-Thought Prompting Elicits Reasoning in LLMs |
| Reasoning/prompting | Self-Consistency Improves Chain of Thought Reasoning |
| Reasoning/prompting | ReAct: Synergizing Reasoning and Acting in Language Models |
| Reasoning/prompting | Tree of Thoughts |
| Reasoning/prompting | Chain-of-Table |
| Reasoning/prompting | Everything of Thoughts |
| Reasoning/prompting | Quantifying LLM Sensitivity to Spurious Features in Prompt Design |
| LLM behavior | What is In-Context Learning (blog) |
| LLM behavior | Dissociating Language and Thought in LLMs |
| Applications | LLMs as Data Preprocessors |

Once seeding finishes, query any of those papers right away — no manual
`ingest` needed:

```bash
curl -X POST localhost:8000/v1/ask -H "Content-Type: application/json" \
  -d '{"query": "What is Tree of Thoughts and how does it differ from chain-of-thought prompting?"}'
```

```json
{
  "answer": "Tree of Thoughts (ToT) generalizes chain-of-thought prompting by letting the model explore multiple reasoning paths...",
  "citations": ["https://arxiv.org/pdf/2305.10601"],
  "kind": "answer"
}
```

## Two interfaces, one pipeline

Every operation below has two equivalent forms, both calling the exact
same underlying code and the exact same persisted `data/index/`. Pick
whichever fits; nothing about the pipeline itself differs.

- **CLI** — `podman compose run app <subcommand>`. A fresh container per
  command, nothing else needs to be running first. Simple for one-off use.
- **HTTP API** — one long-running container answering many `curl`
  requests (what a chatbot/UI would use). **Start it once, before any of
  the `curl` examples below:**

  ```bash
  podman compose up -d --build app
  ```

  Every `curl .../v1/...` example in this guide assumes that's already
  running.

## Ingest — load a document into the index

```bash
# CLI
podman compose run app ingest https://arxiv.org/pdf/2305.10601
podman compose run app ingest data/documents/some-paper.pdf

# API
curl -X POST localhost:8000/v1/ingest -H "Content-Type: application/json" \
  -d '{"source": "https://arxiv.org/pdf/2305.10601"}'
```

What happens:

1. The source string is routed by scheme: a URL goes through the SSRF
   guard (hostname resolved and rejected if it points at a
   private/loopback/internal address — re-checked on every redirect hop);
   a local path is checked against path-traversal, extension, and size
   limits.
2. The matching loader turns the content into one or more chunks of text,
   each tagged with its source. A PDF produces one unit per page (pages
   with no extractable text, e.g. scanned images, are dropped); a PDF
   served directly over HTTP is detected by `Content-Type`, not just
   URL suffix (arXiv's URLs have none).
3. Each unit is split into ~512-character chunks with overlap, embedded
   with the configured backend, and written into a vector collection
   named for that backend — so switching embedding backends never mixes
   incompatible vectors.

CLI output: `Ingested N document(s) from 1 source(s).` — `N` is the number
of page/section units *before* chunking (a 10-page PDF with text on every
page → `N=10`), not the chunk count. The API returns the same count as
`{"documents_ingested": N}`.

## Ask — query the index

```bash
# CLI
podman compose run app ask "What is Tree of Thoughts and how does it differ from chain-of-thought prompting?"

# API
curl -X POST localhost:8000/v1/ask -H "Content-Type: application/json" \
  -d '{"query": "What is Tree of Thoughts?", "k": 5, "history": []}'
```

`history` is optional (defaults to `[]`) and carries prior turns as
`{"role": "user"|"assistant", "content": "..."}`, so a follow-up question
like "how does it compare to chain-of-thought?" can be resolved into a
standalone search query. It's an API-only field; the CLI's `ask` always
sends an empty history.

What happens:

1. The query is routed first (`query_rewrite_enabled`, on by default). A
   bare greeting, thanks, or goodbye matches a free rules check;
   anything else goes through one LLM call that classifies the query
   `doc`, `chitchat`, or `meta`. For `doc`, that same call rewrites the
   query into a standalone search query. It resolves follow-up
   references using `history`, trimmed to the trailing
   `HISTORY_MAX_TURNS` turns (default 6). It also expands abbreviations
   and synonyms. If routing is disabled, or the routing call itself
   fails, the query is treated as `doc` with the raw query text.
2. `chitchat` returns a short reply and `meta` returns a deterministic
   list of indexed sources. Both skip retrieval and generation entirely
   (steps 3-6 below), with no citations. The response `kind` is
   `"chitchat"` or `"meta"`.
3. For `doc`, retrieval is hybrid on the standalone query: vector
   similarity and BM25 keyword search each run, get independently
   normalized, and are blended (`hybrid_alpha`); the merged pool is
   optionally reranked by a cross-encoder (`rerank_enabled`). If nothing
   has been indexed yet, or nothing matches, this returns no results,
   not an error.
4. If nothing relevant was found, the answer is the refusal message
   below (`kind: "refusal"`) and the generator is never called. An
   empty-context prompt risks a confident-sounding hallucination instead
   of an honest "I don't know."
5. Otherwise, the retrieved chunks are numbered and given to the
   generator, with grounding/citation instructions in a system prompt
   (kept separate from the question itself). A deterministic guardrail
   then checks every `[n]` citation the model used actually exists among
   the retrieved chunks — an answer with no citations, or one citing a
   chunk number that doesn't exist, gets the same refusal message instead
   of an ungrounded answer.
6. Citations in a successful response (`kind: "answer"`) come from the
   same chunks' source metadata (deduplicated to unique sources,
   preserving order), so what the model was told to cite and what gets
   reported can't drift apart.
7. The refusal message, used whenever `kind` is `"refusal"`, is: `"I
   don't have enough information in the available documents to answer
   that."`
8. Repeated identical requests (same query, `k`, `history`, and
   retrieval/generation config, against the same corpus state) are
   served from a Valkey-backed cache instead of re-running routing,
   retrieval, and generation.

## Evaluate — score answer quality (LLM-as-judge)

```bash
# CLI
podman compose run app evaluate
podman compose run app evaluate --embeddings huggingface   # compare backends
podman compose run app evaluate --hybrid-alpha 0.8 --rerank false --query-rewrite false

# API
curl -X POST localhost:8000/v1/evaluate -H "Content-Type: application/json" -d '{}'
curl -X POST localhost:8000/v1/evaluate -H "Content-Type: application/json" \
  -d '{"hybrid_alpha": 0.8, "rerank_enabled": false, "query_rewrite_enabled": false}'
```

Runs every question in `evals/golden_questions.yaml` through the ask
pipeline, then has a separately-configured judge model score each answer
on four dimensions (accuracy, answer relevance, context relevance,
completeness). A judge response that isn't parseable is recorded as an
error for that question; the run continues rather than aborting.

Every override (`embeddings`, `hybrid-alpha`, `rerank`, `query-rewrite`)
is temporary (doesn't touch `.env.local`), so you can compare retrieval
strategies against the same golden questions. The `embeddings` override
needs its own prior `ingest`/`index` run against that backend first,
since collections are backend-specific; the others don't.

## eval-routing: score intent classification against labeled cases

CLI only.

```bash
podman compose run app eval-routing
```

Runs the labeled case set (default `evals/routing_cases.yaml`) through
`route()`. Calls the configured generator live, not mocked, and prints
per-category accuracy plus every misroute. A case expected to be `doc`
that routes to anything else is "critical"; the command exits non-zero
only if any misroute is critical.

## eval-retrieval: score retrieval against labeled cases

CLI only.

```bash
podman compose run app eval-retrieval
```

Runs the labeled case set (default `evals/retrieval_cases.yaml`, 30
cases over the 15 seeded papers) through `retrieve()` against whatever is
currently indexed, and prints hit@1, hit@k, and MRR. Exits non-zero if
hit@k falls below `--min-hit-at-k` (default 0.9); `--k` controls how many
chunks `retrieve()` returns per case (default 5).

## Bulk re-index — everything under `data/documents/`

```bash
# CLI
podman compose run app index

# API
curl -X POST localhost:8000/v1/index -H "Content-Type: application/json" -d '{}'
```

Walks `data/documents/` for files with a supported extension and runs the
same ingest pipeline on each, without naming files individually.
**Caveat:** there's no dedup yet — running this repeatedly on unchanged
files duplicates their chunks (document-update/delete is a planned future
addition, not built yet).

## API error responses

The CLI prints errors as text; the API returns real HTTP status codes:

| Situation | Status |
|---|---|
| Bad/unsafe `source` (path traversal, SSRF, unsupported extension) | 400 |
| File not found | 404 |
| Empty `source`/`query`, `k <= 0`, or a malformed `history` turn | 422 |
| Ollama unreachable | 503 |

`GET /health` always returns `200 {"status": "ok"}` independent of
Settings or Ollama — it only reflects whether the process is running
(what the container healthcheck relies on).

## Worked example

```bash
podman compose run app ingest https://arxiv.org/pdf/2305.10601
podman compose run app ask "What is Tree of Thoughts?"
podman compose run app ask "How does Tree of Thoughts differ from chain-of-thought prompting?"
podman compose run app ask "What is the capital of France?"   # off-topic on purpose
```

(`2305.10601` is "Tree of Thoughts: Deliberate Problem Solving with Large
Language Models".)

What to check:

- First two: answers should reference Tree-of-Thoughts/search concepts,
  with sources naming the arXiv URL; the second, more specific question
  should surface noticeably different content than the first.
- Third: intentionally off-topic. Should return the refusal message
  ("I don't have enough information in the available documents to answer
  that."), not a confident wrong answer — if it answers anyway, that's a
  grounding failure worth reporting.

## Running tests locally

```bash
uv run pytest -v
```

The suite mocks every network/SDK/model-loading boundary (no real Ollama,
Anthropic, or HuggingFace calls) — it verifies the code's logic, not that
a live model gives good answers. The worked example above is what proves
the end-to-end pipeline against a real model; the suite below automates it.

### Python e2e tests (`e2e/`)

```bash
uv run pytest e2e/   # real stack must already be running
```

Drives the real HTTP API (`podman compose up -d --build app`, plus
whatever else is already running) against real Ollama/Chroma/Redis —
idempotent re-ingest, large-batch ingest, cache invalidation on content
change, grounded/cited answers, and refusals. Not part of `uv run
pytest`'s default run (see `pyproject.toml`'s `testpaths`), same split as
the Chat UI tests below.

**Leaves fictional chunks in your real index.** Fixture files are deleted
after each run, but there's no delete/prune endpoint, so the chunks
`build_index` already wrote for them stay in Chroma/BM25 permanently —
each full run adds on the order of 150-200 orphaned chunks. Harmless to
correctness (nothing else references them), but they do dilute real
retrieval and `evaluate` scores over repeated runs. Rebuild a clean index
by removing `data/index/` and re-running `podman compose up -d app` (the
startup reseed will repopulate it from `seed_sources.yaml`) if this
suite has been run enough times to matter.

### Chat UI tests (`web/`)

```bash
cd web && npm test        # fast, mocked (component tests, no real backend)
```

```bash
cd web && npm run test:e2e   # real stack must already be running
```

`npm run test:e2e` drives a real browser (Playwright) against the real
running stack — `podman compose up app web` (plus whatever else is
already running) — and asks real questions, asserting on real grounded
answers/citations/refusals. Not part of `npm test`'s default run, same
split as the Python suite above.

## Troubleshooting

- **Connection error reaching Ollama**: check `podman compose ps` shows
  `ollama` healthy; confirm `.env.local`'s `*_BASE_URL` values say
  `ollama`, not `localhost` (inside a container, `localhost` means that
  container itself).
- **Permission denied writing to `data/`**: the app runs as a non-root
  user inside the container; `chmod -R a+rwX data/` on the host, or run
  Podman with `--userns=keep-id`.
- **`evaluate` always errors**: check `JUDGE__API_KEY` is set in
  `.env.local` and valid for whichever `JUDGE__PROVIDER` is configured.
- **`UrlSecurityError` on a real public URL**: check the host actually
  resolves publicly (`dig <host>`) — this should only trigger for
  private/loopback/internal addresses.
