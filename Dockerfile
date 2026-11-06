# syntax=docker/dockerfile:1

# Single stage on purpose. A builder stage plus a final `COPY --from=builder
# /app` holds the virtualenv twice during the build, which needs well over
# 25 GB of free disk. The uv binary is mounted only for the install steps, so
# no build tool ends up in the image.
FROM python:3.14-slim-bookworm

RUN useradd --create-home --uid 1000 appuser \
    && mkdir /app \
    && chown appuser:appuser /app

WORKDIR /app
USER appuser

# UV_CACHE_DIR points at the cache mount below. The uv binary is pinned by
# digest (was :latest as of 2026-10-24) so it can never be silently swapped;
# dependency versions are pinned separately by uv.lock. Re-pin periodically.
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_CACHE_DIR=/tmp/uv-cache \
    PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1

# Dependencies first, in their own layer, so an app-code-only change doesn't
# invalidate (and re-download) this layer.
COPY --chown=appuser:appuser pyproject.toml uv.lock ./
RUN --mount=type=bind,from=ghcr.io/astral-sh/uv@sha256:32a7bb39b496babbd997b0a39a4d66c126bdaf5661b302bf6f06abda8deca6ff,source=/uv,target=/bin/uv \
    --mount=type=cache,target=/tmp/uv-cache,uid=1000,gid=1000 \
    uv sync --frozen --no-install-project --no-dev

# Everything else; .dockerignore excludes what shouldn't ship (tests,
# .venv, .git, data/, .env.local, caches, ...).
COPY --chown=appuser:appuser . .
RUN --mount=type=bind,from=ghcr.io/astral-sh/uv@sha256:32a7bb39b496babbd997b0a39a4d66c126bdaf5661b302bf6f06abda8deca6ff,source=/uv,target=/bin/uv \
    --mount=type=cache,target=/tmp/uv-cache,uid=1000,gid=1000 \
    uv sync --frozen --no-dev

ENTRYPOINT ["ask-the-docs"]
