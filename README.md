# research-assistant

A local-first RAG (Retrieval-Augmented Generation) research assistant.
Point it at PDFs, Word docs, markdown/text files, or URLs, then ask
questions grounded in that content with cited sources — no data leaves
your machine except the LLM calls you explicitly configure.

**Key features**

- **Local generation by default** — Ollama runs the language model; no
  API key required for the core pipeline.
- **Pluggable everywhere** — swap the generator, the judge, and the
  embeddings backend independently via config, without touching code.
- **Configurable LLM-as-judge evaluation** — score answer quality on
  accuracy, relevance, and completeness using a separately-configured
  model (e.g. Claude).
- **Two interfaces, same logic** — a one-shot CLI and a long-running HTTP
  API, both calling identical underlying code.
- **Containerized** — `podman compose` (or `docker compose`) runs the
  whole stack; nothing to install beyond that.
- **Security-hardened ingestion** — path-traversal and SSRF guards on
  every local file and URL before anything is read.

> Currently implements vector RAG (chunk → embed → retrieve → generate).
> Hybrid search and Graph RAG are planned as separate future additions.

## Quickstart

```bash
git clone <this-repo> && cd research-assistant
cp .env.example .env.local        # reference copy -> your local config
podman compose up -d ollama-init  # starts Ollama, pulls the models (first run only)
podman compose build app && podman image prune -f

podman compose run app ingest https://arxiv.org/pdf/2305.10601
podman compose run app ask "What is Tree of Thoughts?"
```

That's the minimum path to a working answer. For every command, the HTTP
API equivalent, evaluation, bulk re-indexing, and troubleshooting, see
**[USAGE.md](USAGE.md)**.

## Development

```bash
uv sync          # installs locked dependencies for tests/tooling
uv run pytest    # runs the full test suite (no containers or API keys needed)
```

See **[CLAUDE.md](CLAUDE.md)** for architecture and contribution
conventions.

## Layout

- `src/research_assistant/` — package: `config`, `llm`, `embeddings`,
  `ingest`, `index`, `retrieve`, `answer`, `evaluate`, `cli`, `api`
- `data/documents/` — input documents (git-ignored)
- `data/index/` — persisted vector store (git-ignored)
- `evals/` — golden question set for evaluation
- `tests/` — unit tests
