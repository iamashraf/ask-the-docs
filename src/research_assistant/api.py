"""Long-running HTTP API: thin routes over the existing ingest/index/answer/evaluate functions."""

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from research_assistant.config import get_settings
from research_assistant.index import build_index
from research_assistant.ingest import (
    PathSecurityError,
    UnsupportedFileTypeError,
    UrlSecurityError,
    discover_sources,
    dispatch,
)

app = FastAPI(title="research-assistant")


@app.exception_handler(PathSecurityError)
@app.exception_handler(UrlSecurityError)
@app.exception_handler(UnsupportedFileTypeError)
def _bad_request_handler(request: Request, exc: Exception) -> JSONResponse:
    return JSONResponse(status_code=400, content={"detail": str(exc)})


@app.exception_handler(FileNotFoundError)
def _not_found_handler(request: Request, exc: Exception) -> JSONResponse:
    return JSONResponse(status_code=404, content={"detail": str(exc)})


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
