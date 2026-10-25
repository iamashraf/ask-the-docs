# research-assistant

A local-first RAG research assistant over configurable document sources
(PDF/DOCX/TXT/MD files, or URLs), using a local Ollama model to generate
answers and a configurable LLM (Claude by default) as judge.

## Setup

```bash
uv sync                 # creates .venv and installs locked dependencies
cp .env.example .env    # then add your API key
```

Run the CLI directly on your host (requires Ollama running locally, and
`.env`'s base URLs set to `http://localhost:11434` rather than `ollama`):

```bash
uv run research-assistant ingest data/documents/some-paper.pdf
uv run research-assistant ask "What does this paper say?"
```

## Testing

`uv run pytest` runs the suite inside the project's `uv`-managed virtualenv
(no manual `source .venv/bin/activate` needed).

```bash
uv run pytest             # whole suite
uv run pytest -v          # verbose, one line per test
uv run pytest tests/test_config.py -v   # a single module's tests
```

## Running via containers

Everything (the app and a local Ollama server) runs through `podman compose`
(the same `docker-compose.yml` works with `docker compose` too):

```bash
cp .env.example .env    # defaults point at the "ollama" service name
podman compose up -d ollama
podman exec -it $(podman compose ps -q ollama) ollama pull llama3.1:8b
podman exec -it $(podman compose ps -q ollama) ollama pull nomic-embed-text

podman compose run app ingest data/documents/some-paper.pdf
podman compose run app ask "What does this paper say?"
podman compose run app evaluate
```

Set `JUDGE__API_KEY` in `.env` before running `evaluate` (the judge defaults
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
