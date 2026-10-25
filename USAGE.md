# Usage Guide

How to run the pipeline end to end, what each command actually does under
the hood, and how to verify it's working. See `README.md` for install/setup
and `CLAUDE.md` for architecture.

## Prerequisites

- Podman (or Docker) with `compose` support.
- An Anthropic API key, only if you plan to run `evaluate` (the judge
  defaults to Claude).

## 0. One-time setup

```bash
cp .env.example .env.local
```

`.env.example` is reference only (committed); `.env.local` is what's
actually loaded, and is never committed (listed in `.gitignore`).

Edit `.env.local`: set `JUDGE__API_KEY` if you'll run `evaluate`. The other
defaults (`GENERATOR__*`, `OLLAMA_BASE_URL`, etc.) already point at the
`ollama` service name, which is correct for the containerized setup.

```bash
podman compose up -d ollama-init   # starts ollama, waits until healthy,
                                     # pulls llama3.1:8b + nomic-embed-text, exits
podman compose build app
```

`ollama-init` only downloads models once — the pulled weights persist in
the `ollama_data` volume, so re-running this later is a fast no-op.

## 1. `ingest <source>` — load a document into the index

```bash
podman compose run app ingest https://arxiv.org/pdf/2305.10601
# or a local file already under data/documents/:
podman compose run app ingest data/documents/some-paper.pdf
```

What happens, in order:

1. **`dispatch(source)`** (`ingest.py`) looks at the source string. A URL
   scheme (`http`/`https`) routes to `load_url`; anything else is treated
   as a local path and routes to `load_path` after `validate_local_path`
   checks it doesn't escape `docs_dir`, has an allowed extension, and is
   under the size limit.
2. For a URL, **`validate_url`** resolves the hostname and rejects it if it
   points at a private/loopback/internal address (SSRF guard) — this runs
   again on every redirect hop, not just the original URL.
3. The matching **loader** (`load_pdf`, `load_text`, `load_docx`, or
   `load_url`) turns the raw content into one or more LangChain `Document`
   objects, each tagged with `source` and `source_doc_id`. A PDF produces
   one `Document` per page (pages with no extractable text, e.g. scanned
   images, are dropped). `load_url` detects a PDF response by
   `Content-Type` even when the URL itself has no `.pdf` suffix (true for
   arXiv's URLs) and routes through the same page-extraction logic.
4. **`build_index(docs, settings)`** (`index.py`) chunks each document
   (`RecursiveCharacterTextSplitter`, 512 chars / 50 overlap) and tags
   every chunk with `chunk_index`, `total_chunks`, `embedding_model`, and
   `ingested_at`. Each chunk is embedded (via whichever backend
   `EMBEDDINGS_BACKEND` selects) and written into a Chroma collection named
   for that backend (`chunks_ollama` or `chunks_huggingface`) — this keeps
   the two backends' vectors from ever mixing.

Output: `Ingested N document(s) from 1 source(s).` — `N` is the number of
page/section `Document`s produced *before* chunking (e.g. a 10-page PDF
with real text on every page → `N=10`), not the chunk count.

## 2. `ask "<question>"` — query the index

```bash
podman compose run app ask "What is Voyager and how does it explore Minecraft?"
```

What happens:

1. **`retrieve(query, settings, k=5)`** embeds the query with the *same*
   embedding backend/model used at index time, and runs a similarity
   search against the active collection, returning the top-k chunks.
   (If nothing has been indexed yet, this returns `[]`, not an error.)
2. If no chunks came back, **`answer()`** short-circuits to `"no relevant
   context found"` and never calls the LLM — this is deliberate: an
   empty-context prompt sent to an LLM risks a confident-sounding
   hallucination instead of an honest "I don't know."
3. Otherwise, **`format_context`** numbers the chunks `[1]`, `[2]`, ... with
   their source, **`build_prompt`** wraps that with an instruction to
   answer only from the given context and cite with those markers, and the
   **generator** `LLMClient` (Ollama by default) produces the answer text.
4. Citations printed under `Sources:` come straight from the same chunks'
   `source` metadata — so what the model was told to cite and what gets
   reported can't drift apart.

## 3. `evaluate [--embeddings ollama|huggingface]` — score answer quality

```bash
podman compose run app evaluate
```

Runs every question in `evals/golden_questions.yaml` through `answer()`,
then has the **judge** `LLMClient` (Claude by default — a separate,
independently-configured role from the generator) score each answer on
four dimensions (accuracy, answer relevance, context relevance,
completeness), per your course's LLM-as-judge rubric. A judge response that
isn't parseable JSON is recorded as `ERROR` for that question; the run
continues rather than aborting.

`--embeddings huggingface` temporarily overrides the embeddings backend for
just this run (without touching your `.env.local`), so you can compare backends:

```bash
podman compose run app evaluate --embeddings ollama
podman compose run app evaluate --embeddings huggingface
```

Each needs its own prior `ingest`/`index` run against that backend first —
collections are backend-specific, so comparing requires having indexed the
same corpus under both.

## 4. `index` — bulk re-index everything under `data/documents/`

```bash
podman compose run app index
```

Walks `data/documents/` for files with an allowed extension and runs the
same `dispatch` + `build_index` pipeline as `ingest`, without needing to
name each file. **Caveat:** there's no dedup yet (the document-update
workflow is out of scope for this sub-project) — running `index` twice on
unchanged files duplicates their chunks.

## Worked example: testing with a real paper

```bash
podman compose run app ingest https://arxiv.org/pdf/2305.10601
podman compose run app ask "What is Voyager?"
podman compose run app ask "How does Voyager explore Minecraft without human demonstrations?"
podman compose run app ask "What is the capital of France?"   # should find no relevant context
```

What to check:

- First `ask`: answer should reference Voyager/Minecraft concepts, with
  `Sources:` naming the arxiv URL.
- Second `ask`: a more specific question should surface different/more
  specific chunks than the first — if the answers look identical, retrieval
  may not be distinguishing between chunks well.
- Third `ask`: this is intentionally off-topic. It should return
  `no relevant context found`, not a confidently wrong answer made up from
  the LLM's own training data — if it answers anyway, that's the
  hallucination-without-grounding failure RAG exists to prevent, and worth
  flagging.

## Running tests locally (fast, no containers needed)

```bash
uv run pytest -v
```

All 68 unit tests mock the network/SDK/model-loading boundary (no real
Ollama, Anthropic, or HuggingFace calls) — they verify the code's logic,
not that a live LLM gives good answers. The container walkthrough above is
what actually proves the end-to-end pipeline works against a real model.

## Troubleshooting

- **Connection error reaching Ollama**: check `podman compose ps` shows
  `ollama` healthy; confirm `.env.local`'s `*_BASE_URL` values say `ollama`, not
  `localhost` (inside a container, `localhost` means that container
  itself).
- **Permission denied writing to `data/`**: the app runs as a non-root
  user (uid 1000) inside the container; `chmod -R a+rwX data/` on the host,
  or run Podman with `--userns=keep-id`.
- **`evaluate` always errors**: check `JUDGE__API_KEY` is set in `.env.local` and
  is a valid key for whichever `JUDGE__PROVIDER` you've configured.
- **`UrlSecurityError` on a real public URL**: check the host actually
  resolves publicly (`dig <host>`) — this should only trigger for
  private/loopback/internal addresses; if a legitimate public site is
  rejected, that's a bug worth reporting.
