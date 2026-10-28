# CLAUDE.md (ask-the-docs)

## Project

A local-first, containerized RAG system over configurable document
sources (white papers, markdown, text, docx, URLs) — general-purpose,
not limited to research papers. Educational build: implement
incrementally, in small reviewed slices, not generated wholesale.

Scope: vector + hybrid (BM25) RAG with cross-encoder reranking and
LLM-based query rewriting (sub-project 1-2); Graph RAG — LLM-based
triplet extraction, embedding-similarity entity resolution, Neo4j GDS
Leiden community detection/summarization, and local/global query
routing (sub-project 3) — built, off by default (`graph_enabled`), see
`docs/ARCHITECTURE.md`/`docs/CONCEPTS.md`. A chatbot/UI consuming the
existing HTTP API (sub-project 4) is the next planned sub-project, not
built yet. Don't add it speculatively.


## Conventions

- `uv run pytest` runs the suite (no manual venv activation). Run the
  whole suite, not just the new file, before committing.
- One task = one slice: write the failing test, confirm it fails,
  implement, confirm it passes, commit. Never implement ahead of tests,
  never bundle multiple slices into one commit.
- Unit tests mock the network/SDK/model-loading boundary (`requests.post`,
  SDK classes, `SentenceTransformer`, `uvicorn.run`) — the fast suite must
  stay fast and network-free. Real Ollama/Chroma integration is a
  separate, smaller smoke-test pass (see `USAGE.md`).
- Any package imported directly must be an explicit dependency
  (`uv add X`), even if already present transitively through another
  package.
- `torch` is pinned via `tool.uv.sources` in `pyproject.toml` to PyTorch's
  CPU-only index on Linux specifically — plain PyPI `torch` on Linux
  bundles multi-GB CUDA runtime libraries we never use. Don't remove that
  override; don't add `torchvision`/other torch-family packages without
  giving them the same override.
- Docstrings: one line, intent/why only, and only where it isn't obvious
  from the name/signature. No multi-line docstrings, no comment bloat —
  deeper explanation belongs in conversation, not in the file.
- Work happens on a feature branch per sub-project (check `git branch` for
  the active one); `main` only gets the sub-project's work once it's
  reviewed complete.
