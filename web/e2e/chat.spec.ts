import { expect, test } from "@playwright/test";

// These tests require the real stack already running (podman/docker compose
// up): real Next.js, real FastAPI, real Ollama/Chroma, and real Neo4j if
// GRAPH_ENABLED=true. Not part of `npm test` -- see USAGE.md.
//
// Repeat runs within the answer cache's TTL (Valkey, 24h) hit the cache
// and return in milliseconds rather than exercising real retrieval/LLM
// latency -- expected, not a test bug (see USAGE.md).

const REFUSAL_MESSAGE =
  "I don't have enough information in the available documents to answer that.";
const ANSWER_TIMEOUT = 60_000; // uncached cold-model latency can exceed 30s

test("answers a specific-entity question with a real citation", async ({ page }) => {
  await page.goto("/");
  await page.getByRole("textbox").fill("What is the ReAct framework?");
  await page.getByRole("button", { name: /send/i }).click();
  // Citations render as <li> elements (ChatMessage); the corpus is seeded
  // from URLs, not local filenames, so assert on structure, not a guessed
  // string shape.
  await expect(page.locator(".message__citations li").first()).toBeVisible({
    timeout: ANSWER_TIMEOUT,
  });
});

test("answers a corpus-wide synthesis question", async ({ page }) => {
  await page.goto("/");
  await page
    .getByRole("textbox")
    .fill("What themes connect the reasoning and prompting papers in this corpus?");
  await page.getByRole("button", { name: /send/i }).click();
  await expect(page.locator(".message--assistant").last()).toBeVisible({
    timeout: ANSWER_TIMEOUT,
  });
  // .message--assistant alone doesn't rule out a refusal (a refusal also
  // renders as an assistant message) -- a real synthesis answer must not
  // be the refusal text, which a corpus-wide question like this should
  // never trigger.
  await expect(page.getByText(REFUSAL_MESSAGE)).not.toBeVisible();
});

test("refuses a deliberately out-of-corpus question", async ({ page }) => {
  await page.goto("/");
  await page.getByRole("textbox").fill("What's your return policy on flights to Mars?");
  await page.getByRole("button", { name: /send/i }).click();
  await expect(page.getByText(REFUSAL_MESSAGE)).toBeVisible({ timeout: ANSWER_TIMEOUT });
});
