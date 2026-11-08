# Usage guide

Full command reference. For each operation: what it does internally, how to
run it from the CLI or the HTTP API, and how to check it works. See
[README.md](README.md) for the quickstart and [dev-docs/ARCHITECTURE.md](dev-docs/ARCHITECTURE.md) for
architecture.

## Prerequisites

- Podman (or Docker) with `compose` support.
- An Anthropic API key. The generator and the judge both default to Claude.
  Export `ANTHROPIC_API_KEY` before `podman compose up`. Processes that run
  on the host (the CLI, `uv run ask-the-docs serve`) can instead use an
  `ant auth login` session. To run fully local with no key, set the
  Ollama variables described in `.env.example`.
- The external `llm-observability` Docker/Podman network. `app`, `worker`,
  and `web` join it, and Compose refuses to start them without it. Create
  it once with `podman network create llm-observability`. To also collect
  traces/metrics/logs, start the sibling `llm-observability` repo's stack
  first (see its README's "Start" section). Without it, the app runs
  normally and the OTLP exporters retry and log warnings.

## Setup (one-time)

```bash
cp .env.example .env.local
```

`.env.example` is a committed reference file. The app loads `.env.local`,
which is git-ignored. Never commit it. The defaults
already use Claude and the right service names for the containerized setup.
Anthropic looks for a credential in this order: `GENERATOR__API_KEY` or
`JUDGE__API_KEY`, then `ANTHROPIC_API_KEY`, then the `ant auth login` session
(host processes only). Containers need `ANTHROPIC_API_KEY`.

```bash
podman compose up -d ollama-init   # starts Ollama, waits until healthy,
                                    # pulls nomic-embed-text
podman compose build app
```

Embeddings always run on Ollama, because Anthropic has no embeddings API.
To also pull local chat models, run
`OLLAMA_PULL_MODELS="nomic-embed-text llama3.1:8b" podman compose up -d ollama-init`.
Model pulls happen once. Re-running `ollama-init` later is a
fast no-op (`ollama pull` skips already-present models).
Building the app image needs about 20 GB of free disk at its peak. The
finished image is about 5 GB.

After a code change, skip the separate `build` step. Add `--build` to the
next command instead (`podman compose run --build app ...` or
`podman compose up -d --build app`). The layer cache skips unchanged layers,
so a rebuild with no relevant changes is fast.

**The first server start seeds a reading list.** The first time
`podman compose up -d app` runs, it ingests every source in
`seed_sources.yaml`, a curated set of foundational and recent LLM/RAG papers.
You can `ask` against real content without ingesting anything. The CLI's
`ingest` and `index` do not trigger seeding. The first start is slower than
later ones because it fetches, chunks, and embeds 15 documents. A marker file
in `data/index/` limits seeding to once per fresh index. Edit
`seed_sources.yaml` to change what gets seeded.

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

Once seeding finishes, query any of those papers without a manual `ingest`:

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

## Network access

Compose publishes every port on `127.0.0.1` only: the API (8000), the chat UI
(3000), Ollama (11434), and Neo4j (7687). Only your machine can reach them.
Inside the `app` container the API listens on `0.0.0.0`, set by `API_HOST`,
because the port mapping needs that. Outside a container, `API_HOST` defaults
to `127.0.0.1`.

The API has no authentication. If you publish a port on another interface,
put your own access control in front of it.

## Enable Graph RAG

Graph RAG is off by default, and the `neo4j` service does not start with the
rest of the stack. To turn it on:

1. In `.env.local`, set `GRAPH_ENABLED=true`, `NEO4J_PASSWORD=<password>`, and
   `NEO4J_AUTH=neo4j/<password>`. Use a password of your own.
2. Start Neo4j with `podman compose --profile graph up -d neo4j`.
3. Recreate the services that read the setting with
   `podman compose up -d --force-recreate app worker`.

While `GRAPH_ENABLED=true`, settings validation rejects a blank password,
`neo4j`, and `ask-the-docs-graph`. Neo4j stores the password it first started
with in the `neo4j_data` volume, so changing `NEO4J_AUTH` later does not
change the login of an existing database.

## Two interfaces, one pipeline

Every operation below has two forms. Both call the same code and use the
same persisted `data/index/`. The pipeline is identical.

- **CLI.** `podman compose run app <subcommand>` starts a fresh container per
  command. Nothing else needs to be running first. Use it for one-off work.
- **HTTP API.** One long-running container answers many `curl` requests. A
  chatbot or UI uses this form. Start it once, before any of the `curl`
  examples below:

  ```bash
  podman compose up -d --build app worker
  ```

  `app` answers `curl` requests. `worker` processes queued `/v1/ingest` and
  `/v1/index` jobs. Without it, those jobs stay `queued` forever. Every
  `curl .../v1/...` example in this guide assumes both are running.

## Ingest: load a document into the index

```bash
# CLI
podman compose run app ingest https://arxiv.org/pdf/2305.10601
podman compose run app ingest data/documents/some-paper.pdf

# API
curl -X POST localhost:8000/v1/ingest -H "Content-Type: application/json" \
  -d '{"source": "https://arxiv.org/pdf/2305.10601"}'
```

What happens:

1. The pipeline routes the source string by scheme and validates it. A URL
   goes through the SSRF guard: the hostname is resolved and rejected if it
   points at a private, loopback, or internal address, and the check repeats
   on every redirect hop. A local path is checked against path-traversal,
   extension, and size limits.
2. The matching loader turns the content into one or more chunks of text,
   each tagged with its source. A PDF becomes one markdown document. A PDF
   with no extractable text, such as scanned images, produces nothing. The
   loader detects a PDF served over HTTP by `Content-Type`, not by URL suffix
   (arXiv's URLs have none).
3. The pipeline splits each unit into ~512-character chunks with overlap,
   embeds them with the configured backend, and writes them into a vector
   collection named for that backend. Switching embedding backends never
   mixes incompatible vectors.

The CLI runs all three steps synchronously. It prints `Ingested N
document(s) from 1 source(s).`, where `N` is the number of page or section
units *before* chunking, not the chunk count. A 10-page PDF with text on every
page gives `N=10`.

The API runs only step 1 inline, so a bad source gets an immediate `400` or
`404`. It then enqueues a job on NATS JetStream and returns `202
{"job_id": "...", "status": "queued"}`. A separate `worker` process runs
steps 2-3. Poll for completion:

```bash
curl localhost:8000/v1/jobs/<job_id>
# {"job_id": "...", "status": "done", "result": {"documents_ingested": N}, ...}
```

`status` moves from `queued` to `processing` to `done` or `failed`.
`GET /v1/jobs` lists recent jobs, newest first, without needing an id. A job
id is a UUID, and any other id returns `404`. On `failed`, `error` holds the
message only for the project's own typed errors, such as an unreachable
Ollama or an unsafe source. Any other failure shows just the exception class
name.

## Ask: query the index

```bash
# CLI
podman compose run app ask "What is Tree of Thoughts and how does it differ from chain-of-thought prompting?"

# API
curl -X POST localhost:8000/v1/ask -H "Content-Type: application/json" \
  -d '{"query": "What is Tree of Thoughts?", "k": 5, "history": []}'
```

`history` is optional (default `[]`). It carries prior turns as
`{"role": "user"|"assistant", "content": "..."}`, which lets the pipeline
turn a follow-up like "how does it compare to chain-of-thought?" into a
standalone search query. It is API-only. The CLI's `ask` always sends an
empty history. The API accepts up to 50 turns, and the server trims them to
`HISTORY_MAX_TURNS`. The web chat sends only the most recent 20 turns, each
truncated.

The API accepts a `query` of up to 4000 characters and a `k` from 1 to 50
(default 5).

What happens:

1. The pipeline routes the query first (`query_rewrite_enabled`, on by
   default). A free rules check matches a bare greeting, thanks, or goodbye.
   Anything else goes through one LLM call that classifies the query as
   `doc`, `chitchat`, or `meta`. For `doc`, that call also rewrites the query
   into a standalone search query. It resolves follow-up references using
   `history`, trimmed to the last `HISTORY_MAX_TURNS` turns (default 6), and
   expands abbreviations and synonyms. If routing is disabled or the routing
   call fails, the pipeline treats the query as `doc` with the raw query text.
2. `chitchat` returns a short reply. `meta` returns a deterministic list of
   indexed sources. Both skip retrieval and generation (steps 3-6 below) and
   return no citations. The response `kind` is `"chitchat"` or `"meta"`.
3. For `doc`, retrieval is hybrid on the standalone query. Vector similarity
   and BM25 keyword search each run, the scores are normalized independently,
   and `hybrid_alpha` blends them. A cross-encoder can rerank the merged pool
   (`rerank_enabled`). If nothing is indexed yet, retrieval returns no
   results, not an error.
4. If nothing was retrieved, such as when nothing is indexed yet, the answer
   is the refusal message below (`kind: "refusal"`) and the generator is never
   called. There is no relevance score cutoff. Any non-empty index returns its
   nearest chunks, and the model refuses when they do not contain the answer.
5. Otherwise, the pipeline numbers the retrieved chunks and gives them to the
   generator inside `<documents>` tags, followed by the question inside
   `<question>` tags. The system prompt holds the grounding and citation
   instructions and tells the model to treat the documents as data, never as
   instructions. A deterministic guardrail then checks that every `[n]`
   citation the model used exists among the retrieved chunks. The pipeline
   retries once if the answer has no citations or cites a chunk number that
   does not exist. If the retry fails too, the user gets the refusal message
   instead of an ungrounded answer.
6. Citations in a successful response (`kind: "answer"`) come from the same
   chunks' source metadata, deduplicated to unique sources in order. What the
   model was told to cite and what the response reports cannot drift apart.
7. Whenever `kind` is `"refusal"`, the message is: `"I don't have enough
   information in the available documents to answer that."`
8. A Valkey-backed cache serves repeated identical requests instead of
   re-running routing, retrieval, and generation. Identical means the same
   query, `k`, `history`, and retrieval/generation config against the same
   corpus state.

## Evaluate: score answer quality (LLM-as-judge)

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
pipeline. A separately configured judge model then scores each answer on
four dimensions: accuracy, answer relevance, context relevance, and
completeness. If the judge response cannot be parsed, the run records an error
for that question and continues.

Every override (`embeddings`, `hybrid-alpha`, `rerank`, `query-rewrite`) is
temporary and leaves `.env.local` alone, so you can compare retrieval
strategies against the same golden questions. The `embeddings` override needs
a prior `ingest` or `index` run against that backend, because collections are
backend-specific. The others do not.

`--query-rewrite false` also disables intent routing, not only query
rewriting, for the whole run. Golden questions that expect a `chitchat` or
`meta` intent then show up as intent-mismatch errors.

## eval-routing: score intent classification against labeled cases

CLI only.

```bash
podman compose run app eval-routing
```

Runs the labeled case set (default `evals/routing_cases.yaml`) through
`route()`. It calls the configured generator live, not mocked, and prints
per-category accuracy plus every misroute. A case expected to be `doc` that
routes to anything else is "critical". The command exits non-zero only if a
misroute is critical.

## eval-retrieval: score retrieval against labeled cases

CLI only.

```bash
podman compose run app eval-retrieval
```

Runs the labeled case set (default `evals/retrieval_cases.yaml`, 30 cases
over the 15 seeded papers) through `retrieve()` against the current index and
prints hit@1, hit@k, and MRR. It exits non-zero if hit@k falls below
`--min-hit-at-k` (default 0.9). `--k` sets how many chunks `retrieve()`
returns per case (default 5).

## Bulk re-index: everything under `data/documents/`

```bash
# CLI
podman compose run app index

# API
curl -X POST localhost:8000/v1/index -H "Content-Type: application/json" -d '{}'
```

Walks `data/documents/` for files with a supported extension and runs the
same ingest pipeline on each, so you do not name files individually.
Re-running it on unchanged files is safe. It skips a source whose content hash
matches the indexed one and replaces a changed source.

Like `/v1/ingest`, the API returns `202 {"job_id": "...", "status":
"queued"}` and runs the walk and ingest in the `worker`. Poll `GET
/v1/jobs/<job_id>` as above. The `result` also includes `files_found` next to
`documents_ingested`.

## API error responses

The CLI prints errors as text. The API returns HTTP status codes:

| Situation | Status |
|---|---|
| `/v1/ingest`/`/v1/index` accepted, job queued | 202 |
| Bad/unsafe `source` (path traversal, SSRF, unsupported extension) | 400 |
| File not found, or `/v1/jobs/{job_id}` with an unknown or non-UUID id | 404 |
| Empty `source`/`query`, `query` over 4000 characters, `k` outside 1 to 50, or a malformed `history` turn | 422 |
| Ollama unreachable, or the job queue cannot accept the job | 503 |

`GET /health` always returns `200 {"status": "ok"}`, whatever the Settings or
Ollama state. It reports only whether the process is running, which is what
the container healthcheck uses.

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

- First two questions: the answers should reference Tree-of-Thoughts and
  search concepts, with sources naming the arXiv URL. The second, more
  specific question should surface different content than the first.
- Third question: it is off-topic on purpose. It should return the refusal
  message ("I don't have enough information in the available documents to
  answer that."), not a confident wrong answer. If it answers anyway, that is
  a grounding failure. Report it.

## Running tests locally

```bash
uv run pytest -v
```

The suite mocks every network, SDK, and model-loading boundary, so it makes no
real Ollama, Anthropic, or HuggingFace calls. It verifies the code's logic,
not that a live model gives good answers. The worked example above checks the
end-to-end pipeline against a real model, and the suite below automates it.

### Python e2e tests (`e2e/`)

```bash
uv run pytest e2e/   # real stack must already be running
```

The tests drive the real HTTP API (`podman compose up -d --build app worker`,
plus whatever else is already running) against real Ollama, Chroma, and Redis.
They cover idempotent re-ingest, large-batch ingest, cache invalidation on
content change, grounded and cited answers, and refusals. `uv run pytest`
skips them by default (see `pyproject.toml`'s `testpaths`), as it skips the
Chat UI tests below.

**Fictional chunks stay in your real index until the next full index.** The
suite deletes its fixture files after each run, but their chunks stay in
Chroma and BM25. A full `ask-the-docs index` (or the next full re-index job)
prunes sources whose files under `docs_dir` no longer exist, which removes the
chunks. The file watcher also sends a remove job when it sees a file deleted.
Until then each run leaves roughly 150-200 orphaned chunks, which dilute
retrieval and `evaluate` scores over repeated runs.

### Chat UI tests (`web/`)

```bash
cd web && npm test        # fast, mocked (component tests, no real backend)
```

```bash
cd web && npm run test:e2e   # real stack must already be running
```

`npm run test:e2e` drives a real browser (Playwright) against the running
stack (`podman compose up app worker web`, plus whatever else is already
running). The ingest page needs `worker` to process what it uploads. The tests
ask real questions and assert on grounded answers, citations, and refusals.
`npm test` skips them by default, as `uv run pytest` skips the Python e2e
tests above.

## Troubleshooting

- **Connection error reaching Ollama.** Check that `podman compose ps` shows
  `ollama` healthy. Confirm `.env.local`'s `*_BASE_URL` values say `ollama`,
  not `localhost`. Inside a container, `localhost` means that container.
- **Permission denied writing to `data/`.** The app runs as a non-root user
  inside the container. Run `chmod -R a+rwX data/` on the host, or run Podman
  with `--userns=keep-id`.
- **Answers or `evaluate` fail with "No Anthropic credential found" or
  "rejected the credentials".** Export `ANTHROPIC_API_KEY` in the shell that
  runs `podman compose`, or set `GENERATOR__API_KEY` and `JUDGE__API_KEY` in
  `.env.local`. Containers cannot use the `ant auth login` session.
- **Another machine or phone cannot reach the API or the chat UI.** Compose
  publishes ports on `127.0.0.1` only. See Network access above.
- **Error that names `NEO4J_PASSWORD`.** `GRAPH_ENABLED=true` needs a password
  of your own. See Enable Graph RAG above.
- **Ollama cuts off long prompts.** Raise the context window with
  `OLLAMA_CONTEXT_LENGTH` (default 8192) in your shell or a project-level
  `.env` file, then recreate the `ollama` container. Compose does not read it
  from `.env.local`.
- **`UrlSecurityError` on a real public URL.** Check that the host resolves
  publicly (`dig <host>`). The error should trigger only for private,
  loopback, or internal addresses.
