# CLAUDE.md (research-assistant)

## Project

A local-first, containerized RAG research assistant over configurable
document sources (white papers, markdown, text, docx, URLs). Educational
build: implemented incrementally, in small reviewed slices, not generated
wholesale. Currently on sub-project 1 of 3 (plain vector RAG); hybrid
search/reranking and Graph RAG are later, separately-specced sub-projects.


## Architecture

- **`config.py`** — `Settings`/`ProviderConfig`: generator and judge are
  independently-configured LLM roles (provider/model/api_key/base_url).
- **`llm.py`** — `LLMClient` protocol + `OllamaLLMClient`/`AnthropicLLMClient`/
  `OpenAILLMClient`, built by `get_llm_client(config)`. Same interface
  serves both the generator and the judge.
- **`embeddings.py`** — `Embeddings` protocol + `OllamaEmbeddings`/
  `HuggingFaceEmbeddings`, built by `get_embeddings(settings)`. Each
  backend writes to its own named Chroma collection so both can be
  indexed from the same corpus and compared.
- **`ingest.py`** — security gates (`validate_local_path` for path
  traversal/extension/size, `validate_url` for SSRF) run before any
  content is read or fetched. `make_source_doc_id` tags every document
  for future bulk-delete-by-source.
- Both plugin points follow the same **Strategy + Factory** shape: a
  `Protocol` interface, concrete implementations, one factory function
  that's the only place branching on config happens. Adding a new
  provider/backend later means one class + one branch, no call-site
  changes.

## Conventions

- `uv run pytest` runs the suite (no manual venv activation). Run the
  whole suite, not just the new file, before committing.
- One task = one slice: write the failing test, confirm it fails,
  implement, confirm it passes, commit. Never implement ahead of tests,
  never bundle multiple slices into one commit.
- Unit tests mock the network/SDK/model-loading boundary (`requests.post`,
  SDK classes, `SentenceTransformer`) — the fast suite must stay fast and
  network-free. Real Ollama/Chroma integration is a separate, smaller
  smoke-test pass.
- Docstrings: one line, intent/why only, and only where it isn't obvious
  from the name/signature. No multi-line docstrings, no comment bloat —
  deeper explanation belongs in conversation, not in the file.
- Branch: `vector-rag-foundation` for all of sub-project 1. `main` stays
  at the bare scaffold until this sub-project is reviewed and merged.

## Out of scope for now

Hybrid search/BM25, cross-encoder reranking, query
rewriting/expansion, and Graph RAG (triplet extraction, entity
resolution, community summaries) are sub-projects 2 and 3 — each gets
its own brainstorm → spec → plan cycle once sub-project 1 is done.
