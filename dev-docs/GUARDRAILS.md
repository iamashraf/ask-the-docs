# Guardrails

This page lists the checks ask-the-docs runs today, where they live, and what
happens when one fails. It also lists what is missing. Every claim here ties
to code and a test.

## Scope

The project is built for local use by one person on one machine. It has no
user accounts, no tenants, and no tools or agent loop, so those controls do
not apply. The API has no authentication. Do not expose it to a network you
do not trust.

The checks aim to stop these problems:

- Instructions hidden in a document being followed as if the user wrote them.
- Answers that cite sources the model never saw.
- Ingestion reaching local or internal addresses, or reading files outside the
  documents folder.
- Oversized requests and files.

They do not stop a determined attacker with access to the API. See
[SECURITY.md](../SECURITY.md).

## Guardrails by layer

| Layer | Guardrail | Where | Config | On failure |
|---|---|---|---|---|
| Input | Query at most 4000 characters, `k` from 1 to 50, history at most 50 turns of 4000 characters | `api.py` (`MAX_QUERY_CHARS`, `MAX_K`), `router.py` (`Turn`) | constants | HTTP 422 |
| Input | Local files: path must stay under the documents folder, extension in `.pdf .docx .txt .md`, size cap | `ingest.py` (`validate_local_path`) | `MAX_FILE_SIZE_MB` (25) | HTTP 400 |
| Input | URLs: http or https only, every resolved address must be public, redirects re-checked (max 3), size cap, per-read timeout | `ingest.py` (`validate_url`) | `MAX_URL_FETCH_SIZE_MB` (25), `URL_FETCH_TIMEOUT_SECONDS` (10) | HTTP 400 |
| Retrieval | None beyond returning the nearest chunks | `retrieve.py` | | |
| Generation | Retrieved chunks sit inside `<documents>` tags and the question inside `<question>` tags. Tag text inside a chunk is escaped. The system prompt says document text is data, not instructions. | `answer.py` (`build_prompt`, `format_context`), `config.py` | `SYSTEM_PROMPT_TEMPLATE` (hosted providers only) | Reduces risk. A model can still be persuaded. |
| Generation | No retrieved chunks means a refusal without calling the model | `answer.py` | | Refusal message |
| Output | Citation check: the answer needs `[n]` markers and each must match a retrieved chunk | `answer.py` (`citation_guardrail_passes`, `structured_answer_passes`) | `MAX_GENERATION_ATTEMPTS` (2, constant) | One retry, then refusal |
| Exposure | API listens on `127.0.0.1` by default. Compose publishes every port on the host loopback only. | `config.py` (`api_host`), `docker-compose.yml` | `API_HOST` | |
| Exposure | Graph RAG refuses to start with a blank or well-known Neo4j password. The Neo4j service only starts with the `graph` compose profile. | `config.py`, `docker-compose.yml` | `NEO4J_PASSWORD` | Startup error |

On Ollama, `answer.py` uses its own structured prompt, and a custom
`SYSTEM_PROMPT_TEMPLATE` has no effect. On a hosted provider, a custom template
replaces the default one, including its data-not-instructions line, so keep
that line.

The citation check confirms the markers exist and are in range. It does not
confirm that the cited chunk supports the claim.

## Failure behavior

"Closed" means the request is refused. "Open" means the system carries on with
a simpler path.

| Check or dependency | Fails | What the user sees |
|---|---|---|
| Citation check, twice in a row | Closed | Refusal message |
| Model says the context lacks the answer | Closed | Refusal message, no retry |
| Nothing retrieved | Closed | Refusal message, no model call |
| Request over a limit | Closed | HTTP 422 |
| Ingest path or URL check, or DNS lookup | Closed | HTTP 400 |
| Ollama unreachable | Closed | HTTP 503 |
| Intent router errors | Open | Treated as a document question |
| Graph query classification errors | Open | Local search |
| Vector search errors | Open | BM25 results only |
| Cache errors | Open | Answered without the cache |

## Testing

Run everything with `uv run pytest`. CI (`.github/workflows/ci.yml`) also runs
`uv run ruff check .`. The relevant test files:

- `tests/test_ingest_security.py` covers path and URL checks.
- `tests/test_answer.py` covers the citation check, retry, refusal, and
  prompt isolation.
- `tests/test_api.py` covers request limits.
- `tests/test_config.py` covers the Neo4j password rule.

There is no attack-style test suite yet. Four adversarial routing cases run
by hand with `uv run ask-the-docs eval-routing`, which needs a live model.

To add a case, write a failing test next to the check it targets, make it
pass, and commit it with the fix.

## Not yet implemented

- Authentication, rate limits, and token or spend limits per caller.
- PII detection, and redaction before text goes to a hosted model.
- Output scans for secrets, personal data, and system prompt text.
- A check that each cited chunk supports the claim it is attached to.
- Stricter URL fetching: an HTTP status check, a total time limit, and pinning
  the resolved address.
- Timeouts, sandboxing, and malware scanning for file parsing.
- Logging of retrieved document IDs, and redaction of prompts in logs.
- Alerts when checks fire more often than usual.
- A red-team test suite that runs in CI.
