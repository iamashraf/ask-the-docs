# research-assistant

RAG-based assistant for answering questions about agentic AI research papers.

## Setup

```bash
uv sync                 # creates .venv and installs locked dependencies
cp .env.example .env    # then add your API key
```

Run the CLI:

```bash
uv run research-assistant
```

## Testing

`uv run pytest` runs the suite inside the project's `uv`-managed virtualenv
(no manual `source .venv/bin/activate` needed).

```bash
uv run pytest             # whole suite
uv run pytest -v          # verbose, one line per test
uv run pytest tests/test_config.py -v   # a single module's tests
```

## Layout

- `src/research_assistant/` package: config, ingest, index, retrieve, answer, evaluate, cli
- `data/papers/` input PDFs (git-ignored)
- `data/index/` persisted vector store (git-ignored)
- `evals/` golden question set for evaluation
- `tests/` unit tests
