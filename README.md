# research-assistant

RAG-based assistant for answering questions about agentic AI research papers.

## Setup

```bash
uv sync                 # creates .venv and installs locked dependencies
cp .env.example .env    # then add your API key
```

Run the tests and the CLI:

```bash
uv run pytest
uv run research-assistant
```

## Layout

- `src/research_assistant/` package: config, ingest, index, retrieve, answer, evaluate, cli
- `data/papers/` input PDFs (git-ignored)
- `data/index/` persisted vector store (git-ignored)
- `evals/` golden question set for evaluation
- `tests/` unit tests
