# syntax=docker/dockerfile:1

# Build stage: installs dependencies and the app into a virtualenv.
# Kept separate from the final stage so build tools never ship in the
# runtime image.
FROM python:3.14-slim-bookworm AS builder

# Pinned by digest (was :latest as of 2026-10-24) so the uv binary itself
# can never be silently swapped; actual dependency versions are pinned
# separately by uv.lock regardless. Re-pin periodically for uv updates.
COPY --from=ghcr.io/astral-sh/uv@sha256:32a7bb39b496babbd997b0a39a4d66c126bdaf5661b302bf6f06abda8deca6ff /uv /uvx /bin/

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

# Dependencies first, in their own layer, so an app-code-only change doesn't
# invalidate (and re-download) this layer.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev

# Everything else; .dockerignore excludes what shouldn't ship (tests,
# .venv, .git, data/, .env.local, caches, ...).
COPY . .
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# Final stage: just the built virtualenv and app code, no build tools, no root.
FROM python:3.14-slim-bookworm

RUN useradd --create-home --uid 1000 appuser

WORKDIR /app
COPY --from=builder --chown=appuser:appuser /app /app

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1

USER appuser

ENTRYPOINT ["ask-the-docs"]
