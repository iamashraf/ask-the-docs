# AskTheDocs

A RAG (Retrieval-Augmented Generation) system with local storage and search.
Point it at PDFs, Word docs, markdown/text files, or URLs (any documents, not
only research papers), then ask questions grounded in that content with cited
sources. Documents, the index, search, reranking and embeddings run on your
machine. By default Claude writes the answers, so each question and its
retrieved text go to Anthropic as is. Set `GENERATOR__PROVIDER=ollama` to
keep generation local too. Data also leaves if you ingest a URL or set a
telemetry endpoint.

![Ingesting a document, asking grounded questions about it, then asking something outside its scope and getting a clean refusal](dev-docs/demo.gif)

## Features

- **Claude by default, Ollama optional.** Claude Haiku writes the answers and
  Claude Sonnet judges them. Authenticate with `ANTHROPIC_API_KEY` or, for
  processes outside containers, an `ant auth login` session. Ollama (local) and
  any OpenAI-compatible endpoint work through config alone.
- **Hybrid retrieval.** Vector similarity is blended with BM25 keyword
  search. LLM query rewriting and cross-encoder reranking are optional.
- **Grounded generation.** A deterministic citation guardrail refuses to
  answer unless the model cites retrieved chunks by number. It checks that the
  markers exist, not that the chunk supports the claim. Valkey caches answers,
  so repeated questions skip retrieval and generation.
- **Swappable backends.** Config selects the generator, the judge, and the
  embeddings backend independently. No code changes.
- **LLM-as-judge evaluation.** A separately configured model (for example
  Claude) scores answers on accuracy, relevance, and completeness. Toggles
  compare retrieval strategies.
- **Three interfaces.** A one-shot CLI, a long-running HTTP API, and a chat
  UI call the same code.
- **Graph RAG.** LLM-based triplet extraction, embedding-similarity entity
  resolution, and Neo4j GDS community detection add a local (entity-traversal)
  and a global (corpus-theme) search path next to hybrid retrieval. It is off
  by default.
- **Containerized.** `podman compose` (or `docker compose`) runs the whole
  stack: Ollama, Valkey, the app, and the chat UI. Nothing else to install.
- **Ingestion checks.** Local paths are checked for traversal, file type,
  and size. URLs are checked against private, loopback, and internal
  addresses on every redirect. There is no malware scanning or sandboxed
  parsing.

## Architecture

The [interactive runtime architecture diagram](https://iamashraf.github.io/ask-the-docs/runtime-architecture.html)
shows components, trust boundaries, and the primary request path.

Ingestion and query share Chroma, Valkey, and Neo4j. Graph RAG and Neo4j
are off by default. See `dev-docs/ARCHITECTURE.md` for the full ingest and
query flow and the reasons behind the tech stack. See `dev-docs/CONCEPTS.md`
for the concepts behind each stage (vector search, BM25, reranking, Graph
RAG).

## Quickstart

`app`, `worker`, and `web` join an external `llm-observability` Docker/Podman
network, and Compose refuses to start them without it. Create it once with
`podman network create llm-observability`. To also collect traces/metrics/
logs, start the sibling `llm-observability` repo's stack first (see its
README's "Start" section). Without it, the app runs normally and the OTLP
exporters retry and log warnings.

```bash
git clone <this-repo> && cd ask-the-docs
cp .env.example .env.local        # reference copy -> your local config
export ANTHROPIC_API_KEY=sk-ant-...  # compose passes it into app and worker
podman compose up -d ollama-init  # starts Ollama, pulls the embedding model (first run only)
podman compose up -d --build app  # builds if missing, rebuilds if stale, then starts it
```

The first start auto-ingests a curated set of LLM/RAG papers
(`seed_sources.yaml`), so there is content to query:

```bash
curl -X POST localhost:8000/v1/ask -H "Content-Type: application/json" \
  -d '{"query": "What is Tree of Thoughts?"}'
```

The services listen on `127.0.0.1` only, and the API has no authentication.

This is the shortest path to a working answer. [USAGE.md](USAGE.md) covers
every command, the HTTP API equivalents, evaluation, bulk re-indexing, and
troubleshooting.

## Development

```bash
uv sync          # installs locked dependencies for tests/tooling
uv run pytest    # runs the full test suite (no containers or API keys needed)
```

See **[dev-docs/ARCHITECTURE.md](dev-docs/ARCHITECTURE.md)** for
architecture and **[CLAUDE.md](CLAUDE.md)** for contribution conventions.

## Guardrails

The checks are layered, and the project treats the model as untrusted: it
checks the output before you see it. The API has no authentication, so run it
on localhost only.

- **Input.** Request size limits, plus path, file type, size, and URL checks
  (public addresses only) before anything is ingested.
- **Generation.** Retrieved text is fenced in `<documents>` tags, and the
  prompt tells the model it is data, not instructions.
- **Output.** Citation markers must match retrieved chunks. The system retries
  once, then refuses.
- **Exposure.** Ports are loopback-only by default. Graph RAG requires a Neo4j
  password.

Rate limits, PII handling, and output scans are not built yet. See
[dev-docs/GUARDRAILS.md](dev-docs/GUARDRAILS.md) for details, failure
behavior, and gaps, and [SECURITY.md](SECURITY.md) to report a problem.

## Layout

- `src/ask_the_docs/`: the package, with modules `config`, `llm`, `embeddings`,
  `ingest`, `index`, `retrieve`, `answer`, `evaluate`, `cli`, `api`
- `web/`: the Next.js chat UI
- `e2e/`: end-to-end tests against the running stack
- `dev-docs/`: architecture, concepts, model comparison, guardrails
- `data/documents/`: input documents (git-ignored)
- `data/index/`: persisted vector store (git-ignored)
- `evals/`: golden question set for evaluation
- `tests/`: unit tests

## Contributing and license

See [CONTRIBUTING.md](CONTRIBUTING.md). Released under the [MIT License](LICENSE).
