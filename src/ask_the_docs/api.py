"""Long-running HTTP API: thin routes over the existing ingest/index/answer/evaluate functions."""

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from ask_the_docs.answer import answer
from ask_the_docs.config import get_settings
from ask_the_docs.evaluate import run_evaluation
from ask_the_docs.index import build_index
from ask_the_docs.ingest import (
    PathSecurityError,
    UnsupportedFileTypeError,
    UrlSecurityError,
    discover_sources,
    dispatch,
)
from ask_the_docs.llm import OllamaUnavailableError
from ask_the_docs.seed import seed_if_needed

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


class IngestResponse(BaseModel):
    documents_ingested: int


@app.post("/v1/ingest")
def ingest(request: IngestRequest) -> IngestResponse:
    docs = dispatch(request.source)
    if docs:
        build_index(docs, get_settings())
    return IngestResponse(documents_ingested=len(docs))


class IndexResponse(BaseModel):
    documents_ingested: int
    files_found: int


@app.post("/v1/index")
def index() -> IndexResponse:
    """Bulk re-index everything under docs_dir. Running this repeatedly on
    unchanged files duplicates their chunks (no dedup yet)."""
    settings = get_settings()
    paths = discover_sources(settings.docs_dir)
    docs = []
    for path in paths:
        docs.extend(dispatch(str(path.relative_to(settings.docs_dir))))
    if docs:
        build_index(docs, settings)
    return IndexResponse(documents_ingested=len(docs), files_found=len(paths))


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
