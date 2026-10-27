# syntax=docker/dockerfile:1

# Build stage: installs deps and builds the standalone production bundle.
FROM node:26.12.1-slim AS builder

WORKDIR /app

# Dependencies first, in their own layer, so an app-code-only change
# doesn't invalidate (and re-download) this layer.
COPY package.json package-lock.json ./
RUN npm ci

COPY . .
RUN npm run build

# Final stage: just the standalone server output, no build tools, no root.
# The official node image already ships a non-root "node" user (uid 1000) --
# reused here instead of creating a conflicting one.
FROM node:26.12.1-slim

WORKDIR /app
COPY --from=builder --chown=node:node /app/.next/standalone ./
COPY --from=builder --chown=node:node /app/.next/static ./.next/static

ENV NODE_ENV=production \
    PORT=3000

USER node

ENTRYPOINT ["node", "server.js"]
