"""Long-running HTTP API: thin routes over the existing ingest/index/answer/evaluate functions."""

from fastapi import FastAPI

app = FastAPI(title="research-assistant")


@app.get("/health")
def health() -> dict[str, str]:
    """No dependency on Settings/Ollama/anything external -- the compose
    healthcheck gates on this, so it must reflect only "is the process up"."""
    return {"status": "ok"}
