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

Model pulls and the image build each only need to happen once — re-running
them later is a fast no-op (`ollama pull` skips already-present models;
`podman image prune -f` reclaims the disk space the multi-stage build's
intermediate layer leaves behind, at the cost of a slightly slower next
build).

## Two interfaces, one pipeline

Every operation below has two equivalent forms, both calling the exact
same underlying code and the exact same persisted `data/index/`:

- **CLI** (`podman compose run app <subcommand>`) — a fresh container per
  command. Simple for one-off use.
- **HTTP API** (`podman compose up -d app`, then `curl`) — one long-running
  container answering many requests. What a chatbot/UI would use.

Pick whichever fits; nothing about the pipeline itself differs.

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
  -d '{"query": "What is Tree of Thoughts?", "k": 5}'
```

What happens:

1. The query is embedded with the same model used at index time, and a
   similarity search returns the top-k most relevant chunks. (If nothing
   has been indexed yet, this returns no results, not an error.)
2. If nothing relevant was found, the answer is `"no relevant context
   found"` and the language model is never called — an empty-context
   prompt risks a confident-sounding hallucination instead of an honest
   "I don't know."
3. Otherwise, the retrieved chunks are numbered and given to the
   generator with an instruction to answer only from that context and
   cite by number.
4. Citations in the response come from the same chunks' source metadata
   (deduplicated to unique sources, preserving order) — what the model
   was told to cite and what gets reported can't drift apart.

## Evaluate — score answer quality (LLM-as-judge)

```bash
# CLI
podman compose run app evaluate
podman compose run app evaluate --embeddings huggingface   # compare backends

# API
curl -X POST localhost:8000/v1/evaluate -H "Content-Type: application/json" -d '{}'
curl -X POST localhost:8000/v1/evaluate -H "Content-Type: application/json" \
  -d '{"embeddings": "huggingface"}'
```

Runs every question in `evals/golden_questions.yaml` through the ask
pipeline, then has a separately-configured judge model score each answer
on four dimensions (accuracy, answer relevance, context relevance,
completeness). A judge response that isn't parseable is recorded as an
error for that question; the run continues rather than aborting.

The `embeddings` override is temporary (doesn't touch `.env.local`), so
you can compare backends — each needs its own prior `ingest`/`index` run
against that backend first, since collections are backend-specific.

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
| Empty `source`/`query`, or `k <= 0` | 422 |
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
- Third: intentionally off-topic. Should return "no relevant context
  found," not a confident wrong answer — if it answers anyway, that's a
  grounding failure worth reporting.

## Running tests locally

```bash
uv run pytest -v
```

The suite mocks every network/SDK/model-loading boundary (no real Ollama,
Anthropic, or HuggingFace calls) — it verifies the code's logic, not that
a live model gives good answers. The worked example above is what proves
the end-to-end pipeline against a real model.

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
