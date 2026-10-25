# research-assistant

A local-first RAG research assistant over configurable document sources
(PDF/DOCX/TXT/MD files, or URLs), using a local Ollama model to generate
answers and a configurable LLM (Claude by default) as judge.

## Dev setup (for tests/tooling only -- the app itself always runs via containers)

```bash
uv sync    # creates .venv and installs locked dependencies, for pytest/ruff/IDE support
```

```bash
uv run pytest             # whole suite
uv run pytest -v          # verbose, one line per test
uv run pytest tests/test_config.py -v   # a single module's tests
```

The test suite needs no `.env.local` or real API keys -- every test mocks
the network/SDK boundary.

## Running the app

Everything (the app and a local Ollama server) runs through `podman compose`
(the same `docker-compose.yml` works with `docker compose` too):

```bash
cp .env.example .env.local    # .env.example is reference only (committed);
                               # .env.local is what's actually loaded, never committed
podman compose up -d ollama-init   # starts ollama, waits for it to be healthy,
                                    # pulls llama3.1:8b + nomic-embed-text, then exits
podman compose run app ingest data/documents/some-paper.pdf
podman compose run app ask "What does this paper say?"
podman compose run app evaluate
```

The model pull only happens once (the models persist in the `ollama_data`
volume); re-running `ollama-init` later is a fast no-op.

Set `JUDGE__API_KEY` in `.env.local` before running `evaluate` (the judge defaults
to Anthropic, which needs a real key; the generator defaults to the local
Ollama model, which needs none).

Running `research-assistant index` with no arguments re-indexes everything
already under `data/documents/` -- note this has no dedup yet, so running it
twice on unchanged files duplicates their chunks (the document-update/delete
workflow is intentionally out of scope for this sub-project).

## Layout

- `src/research_assistant/` package: config, ingest, index, retrieve, answer, evaluate, cli
- `data/documents/` input documents: pdf/docx/txt/md (git-ignored)
- `data/index/` persisted vector store (git-ignored)
- `evals/` golden question set for evaluation
- `tests/` unit tests
