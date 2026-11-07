"""Long-running HTTP API: thin routes over the existing ingest/index/answer/evaluate functions.

Ingestion is async: /v1/ingest and /v1/index validate the source
synchronously (so a bad request still gets an immediate 400/404, exactly
as before) then enqueue a job and return 202 -- the real dispatch/
build_index work happens in the separate worker process (worker.py),
never during a request."""

import asyncio
import logging
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import requests
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from opentelemetry import trace
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from pydantic import BaseModel, Field

from ask_the_docs.answer import answer
from ask_the_docs.bm25_index import get_bm25_index
from ask_the_docs.config import get_settings
from ask_the_docs.evaluate import run_evaluation, summarize
from ask_the_docs.index import get_chunk_count, load_index
from ask_the_docs.ingest import (
    PathSecurityError,
    UnsupportedFileTypeError,
    UrlSecurityError,
    validate_source,
)
from ask_the_docs.jobs import JobStoreError, create_job, get_async_cache_client, get_job, list_jobs, update_job
from ask_the_docs.llm import OllamaUnavailableError
from ask_the_docs.queue import QueuePublishError, publish_ingest_job
from ask_the_docs.reranker import DEFAULT_RERANKER_MODEL, get_reranker
from ask_the_docs.router import Turn
from ask_the_docs.seed import seed_if_needed
from ask_the_docs.tracing import configure_logging, configure_metrics, configure_tracing

logger = logging.getLogger(__name__)

READY_OLLAMA_TIMEOUT_SECONDS = 5

# uvicorn only configures its own loggers (uvicorn, uvicorn.access), never
# arbitrary app loggers -- without this, seed.py's INFO-level progress logs
# are silently dropped by the default root logger (WARNING threshold).
_package_logger = logging.getLogger("ask_the_docs")
if not _package_logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(name)s: %(message)s"))
    _package_logger.addHandler(_handler)
_package_logger.setLevel(logging.INFO)


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_tracing("ask-the-docs-app", get_settings())
    configure_logging("ask-the-docs-app", get_settings())
    configure_metrics("ask-the-docs-app", get_settings())
    settings = get_settings()
    seed_if_needed(settings)
    if settings.rerank_enabled:
        # Loads the CrossEncoder model once at startup instead of on the
        # first real request -- get_reranker is itself lru_cache'd, so this
        # just moves the one-time load cost off the user's first message.
        get_reranker(DEFAULT_RERANKER_MODEL)
    # Same reasoning, applied to the Chroma/BM25 resident caches added in
    # this branch -- moves their one-time construction cost off the first
    # real /v1/ask request instead of paying it on the user's behalf.
    # Best-effort: a broken index directory or an unreadable cache entry
    # must not crash startup -- the first real request rebuilds anyway.
    try:
        load_index(settings)
        get_bm25_index(settings)
    except Exception:
        logger.warning("Warming the resident index/BM25 caches failed; continuing without warm-up.", exc_info=True)
    yield

GOLDEN_QUESTIONS_PATH = Path("evals/golden_questions.yaml")

app = FastAPI(title="ask-the-docs", lifespan=lifespan)

FastAPIInstrumentor.instrument_app(app)


@app.exception_handler(PathSecurityError)
@app.exception_handler(UrlSecurityError)
@app.exception_handler(UnsupportedFileTypeError)
def _bad_request_handler(request: Request, exc: Exception) -> JSONResponse:
    return JSONResponse(status_code=400, content={"detail": str(exc)})


@app.exception_handler(FileNotFoundError)
def _not_found_handler(request: Request, exc: Exception) -> JSONResponse:
    return JSONResponse(status_code=404, content={"detail": str(exc)})


@app.exception_handler(JobStoreError)
@app.exception_handler(OllamaUnavailableError)
def _service_unavailable_handler(request: Request, exc: Exception) -> JSONResponse:
    return JSONResponse(status_code=503, content={"detail": str(exc)})


@app.get("/health")
def health() -> dict[str, str]:
    """No dependency on Settings/Ollama/anything external -- the compose
    healthcheck gates on this, so it must reflect only "is the process up"."""
    return {"status": "ok"}


@app.get("/ready")
async def ready() -> JSONResponse:
    """Checks each dependency independently so a 503 body says which one is down."""
    settings = get_settings()
    dependencies = {}

    try:
        # Not load_index(settings) -- that's resident (Task 1), so after
        # one success it never touches Chroma again and would report "ok"
        # forever even if Chroma died later. get_chunk_count() runs a real
        # count() query against the resident client's backing store every
        # single call.
        get_chunk_count(settings)
        dependencies["chroma"] = "ok"
    except Exception:
        dependencies["chroma"] = "unreachable"

    try:
        await get_async_cache_client(settings).ping()
        dependencies["redis"] = "ok"
    except Exception:
        dependencies["redis"] = "unreachable"

    try:
        response = await asyncio.to_thread(
            requests.get,
            f"{settings.ollama_base_url}/api/tags",
            timeout=READY_OLLAMA_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        dependencies["ollama"] = "ok"
    except Exception:
        dependencies["ollama"] = "unreachable"

    status_code = 200 if all(status == "ok" for status in dependencies.values()) else 503
    return JSONResponse(status_code=status_code, content=dependencies)


class IngestRequest(BaseModel):
    source: str = Field(min_length=1)


class JobAcceptedResponse(BaseModel):
    job_id: str
    status: str


class JobStatusResponse(BaseModel):
    job_id: str
    kind: str
    source: str | None
    status: str
    result: dict | None
    error: str | None


async def _create_and_publish_job(settings, kind: str, source: str | None) -> JobAcceptedResponse:
    job_id = str(uuid.uuid4())
    client = get_async_cache_client(settings)
    await create_job(client, job_id, kind=kind, source=source)
    try:
        await publish_ingest_job(settings, job_id, kind=kind, source=source)
    except QueuePublishError as exc:
        try:
            await update_job(client, job_id, status="failed", error=str(exc))
        except Exception:
            logger.warning("Failed to mark job %s failed after a publish error", job_id, exc_info=True)
    return JobAcceptedResponse(job_id=job_id, status="queued")


@app.post("/v1/ingest", status_code=202)
async def ingest(request: IngestRequest) -> JobAcceptedResponse:
    settings = get_settings()
    validate_source(request.source)
    return await _create_and_publish_job(settings, kind="ingest", source=request.source)


@app.post("/v1/index", status_code=202)
async def index() -> JobAcceptedResponse:
    """Enqueues a bulk re-index of everything under docs_dir; the worker
    discovers and dispatches the files."""
    return await _create_and_publish_job(get_settings(), kind="index", source=None)


class JobListItem(JobStatusResponse):
    created_at: str
    updated_at: str


class JobListResponse(BaseModel):
    jobs: list[JobListItem]


@app.get("/v1/jobs")
async def get_jobs(limit: int = Query(default=50, ge=1, le=200)) -> JobListResponse:
    settings = get_settings()
    client = get_async_cache_client(settings)
    jobs = await list_jobs(client, limit)
    return JobListResponse(jobs=[JobListItem(**job) for job in jobs])


@app.get("/v1/jobs/{job_id}")
async def get_job_status(job_id: str) -> JobStatusResponse:
    settings = get_settings()
    client = get_async_cache_client(settings)
    job = await get_job(client, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return JobStatusResponse(**job)


MAX_QUERY_CHARS = 4000  # same bound as a history turn
MAX_K = 50  # the retrieval candidate pool's default size


class AskRequest(BaseModel):
    query: str = Field(min_length=1, max_length=MAX_QUERY_CHARS)
    k: int = Field(default=5, ge=1, le=MAX_K)
    history: list[Turn] = Field(default=[], max_length=50)
    session_id: str | None = None


class AskResponse(BaseModel):
    answer: str
    citations: list[str]
    kind: Literal["answer", "refusal", "chitchat", "meta"]


@app.post("/v1/ask")
def ask(request: AskRequest) -> AskResponse:
    if request.session_id is not None:
        # Lets Tempo group every trace in one chat conversation together
        # (TraceQL `span.session.id = ...`) -- any span in the trace carries
        # this, so setting it once here is enough, no propagation needed.
        trace.get_current_span().set_attribute("session.id", request.session_id)
    result = answer(request.query, get_settings(), k=request.k, history=request.history)
    return AskResponse(answer=result.text, citations=result.citations, kind=result.kind)


class EvaluateRequest(BaseModel):
    embeddings: Literal["ollama", "huggingface"] | None = None
    hybrid_alpha: float | None = Field(default=None, ge=0.0, le=1.0)
    rerank_enabled: bool | None = None
    query_rewrite_enabled: bool | None = None


class JudgeScoreOut(BaseModel):
    accuracy: float
    answer_relevance: float
    context_relevance: float
    completeness: float
    justification: str


class EvalResultOut(BaseModel):
    question: str
    score: JudgeScoreOut | None
    error: str | None
    intent: str | None = None
    expected_intent: str | None = None
    latency_seconds: float | None = None


class EvaluateSummaryOut(BaseModel):
    intent_accuracy: float | None
    p50_latency_seconds: float | None
    p95_latency_seconds: float | None


class EvaluateResponse(BaseModel):
    results: list[EvalResultOut]
    summary: EvaluateSummaryOut


@app.post("/v1/evaluate")
def evaluate(request: EvaluateRequest) -> EvaluateResponse:
    settings = get_settings()
    overrides = {}
    if request.embeddings:
        overrides["embeddings_backend"] = request.embeddings
    if request.hybrid_alpha is not None:
        overrides["hybrid_alpha"] = request.hybrid_alpha
    if request.rerank_enabled is not None:
        overrides["rerank_enabled"] = request.rerank_enabled
    if request.query_rewrite_enabled is not None:
        overrides["query_rewrite_enabled"] = request.query_rewrite_enabled
    if overrides:
        settings = settings.model_copy(update=overrides)

    results = run_evaluation(settings, GOLDEN_QUESTIONS_PATH)
    summary = EvaluateSummaryOut(**summarize(results))
    return EvaluateResponse(
        results=[
            EvalResultOut(
                question=r.question,
                score=JudgeScoreOut(**vars(r.score)) if r.score else None,
                error=r.error,
                intent=r.intent,
                expected_intent=r.expected_intent,
                latency_seconds=r.latency_seconds,
            )
            for r in results
        ],
        summary=summary,
    )
