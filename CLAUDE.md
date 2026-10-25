# CLAUDE.md (research-assistant)

## Project

A local-first, containerized RAG research assistant over configurable
document sources (white papers, markdown, text, docx, URLs). Educational
build: implement incrementally, in small reviewed slices, not generated
wholesale.

Scope: vector RAG only. Hybrid search/BM25, cross-encoder reranking, query
rewriting/expansion, and Graph RAG (triplet extraction, entity resolution,
community summaries) are explicitly out of scope — each would be its own
future spec, not built yet. Don't add them speculatively.


## Architecture

- **`config.py`** — `Settings`/`ProviderConfig`: generator and judge are
  independently-configured LLM roles (provider/model/api_key/base_url).
  `Settings` reads only real process env vars, never a file directly —
  `.env.local` (git-ignored, actually loaded) vs `.env.example` (committed,
  reference only) is populated into the environment by whatever starts the
  process (`docker-compose.yml`'s `env_file:`, or a test's `monkeypatch`).
- **`llm.py`** — `LLMClient` protocol + `OllamaLLMClient`/`AnthropicLLMClient`/
  `OpenAILLMClient`, built by `get_llm_client(config)`. Same interface
  serves both the generator and the judge.
- **`embeddings.py`** — `Embeddings` protocol (subclasses
  `langchain_core.embeddings.Embeddings`) + `OllamaEmbeddings`/
  `HuggingFaceEmbeddings`, built by `get_embeddings(settings)`. Each
  backend writes to its own named Chroma collection (`chunks_<backend>`)
  so both can be indexed from the same corpus and compared.
- **`ingest.py`** — security gates (`validate_local_path` for path
  traversal/extension/size, `validate_url` for SSRF, re-checked on every
  redirect hop) run before any content is read or fetched. `load_url`
  detects a PDF response by `Content-Type`, not just URL suffix (arXiv
  URLs have none). `make_source_doc_id` tags every document for future
  bulk-delete-by-source. `discover_sources` walks `docs_dir` for bulk
  re-indexing.
- **`index.py`** — chunking (512 chars / 50 overlap) + per-backend Chroma
  store. **`retrieve.py`** — top-k similarity search. **`answer.py`** —
  builds a cited prompt, calls the generator; short-circuits to "no
  relevant context found" (no LLM call) when retrieval is empty.
  **`evaluate.py`** — LLM-as-judge scoring against `evals/golden_questions.yaml`.
- **Two interfaces, same business logic, both maintained**: `cli.py`
  (`ingest`/`index`/`ask`/`evaluate`/`serve` subcommands, one-shot per
  `podman compose run app <subcommand>`) and `api.py` (FastAPI, `/v1/...`
  routes, long-running via `podman compose up app` → the `serve`
  subcommand). Every route/handler calls the exact same underlying
  functions — never duplicate logic into `api.py` or `cli.py` directly.
- Both plugin points (`LLMClient`, `Embeddings`) follow the same
  **Strategy + Factory** shape: a `Protocol`/ABC interface, concrete
  implementations, one factory function that's the only place branching
  on config happens. Adding a new provider/backend later means one class +
  one branch, no call-site changes.
- `torch` is pinned via `tool.uv.sources` in `pyproject.toml` to PyTorch's
  CPU-only index on Linux specifically — plain PyPI `torch` on Linux
  bundles multi-GB CUDA runtime libraries we never use. Don't remove that
  override; don't add `torchvision`/other torch-family packages without
  giving them the same override.

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
- Docstrings: one line, intent/why only, and only where it isn't obvious
  from the name/signature. No multi-line docstrings, no comment bloat —
  deeper explanation belongs in conversation, not in the file.
- Work happens on a feature branch per sub-project (check `git branch` for
  the active one); `main` only gets the sub-project's work once it's
  reviewed complete.
