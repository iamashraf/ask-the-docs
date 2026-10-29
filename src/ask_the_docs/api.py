"""Long-running HTTP API: thin routes over the existing ingest/index/answer/evaluate functions.

Ingestion is async: /v1/ingest and /v1/index validate the source
synchronously (so a bad request still gets an immediate 400/404, exactly
as before) then enqueue a job and return 202 -- the real dispatch/
build_index work happens in the separate worker process (worker.py),
never during a request."""

import logging
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from ask_the_docs.answer import answer
from ask_the_docs.config import get_settings
from ask_the_docs.evaluate import run_evaluation
from ask_the_docs.ingest import (
    PathSecurityError,
    UnsupportedFileTypeError,
    UrlSecurityError,
    validate_source,
)
from ask_the_docs.jobs import JobStoreError, create_job, get_async_cache_client, get_job, update_job
from ask_the_docs.llm import OllamaUnavailableError
from ask_the_docs.queue import QueuePublishError, publish_ingest_job
from ask_the_docs.seed import seed_if_needed

logger = logging.getLogger(__name__)

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
    seed_if_needed(get_settings())
    yield

GOLDEN_QUESTIONS_PATH = Path("evals/golden_questions.yaml")

app = FastAPI(title="ask-the-docs", lifespan=lifespan)


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


@app.get("/v1/jobs/{job_id}")
async def get_job_status(job_id: str) -> JobStatusResponse:
    settings = get_settings()
    client = get_async_cache_client(settings)
    job = await get_job(client, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return JobStatusResponse(**job)


class AskRequest(BaseModel):
    query: str = Field(min_length=1)
    k: int = Field(default=5, ge=1)


class AskResponse(BaseModel):
    answer: str
    citations: list[str]


@app.post("/v1/ask")
def ask(request: AskRequest) -> AskResponse:
    result = answer(request.query, get_settings(), k=request.k)
    return AskResponse(answer=result.text, citations=result.citations)


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


class EvaluateResponse(BaseModel):
    results: list[EvalResultOut]


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
    return EvaluateResponse(
        results=[
            EvalResultOut(
                question=r.question,
                score=JudgeScoreOut(**vars(r.score)) if r.score else None,
                error=r.error,
            )
            for r in results
        ]
    )
