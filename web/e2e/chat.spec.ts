import { expect, test } from "@playwright/test";

// These tests require the real stack already running (podman/docker compose
// up): real Next.js, real FastAPI, real Ollama/Chroma, and real Neo4j if
// GRAPH_ENABLED=true. Not part of `npm test` -- see USAGE.md.

test("answers a specific-entity question with a real citation", async ({ page }) => {
  await page.goto("http://localhost:3000");
  await page.getByRole("textbox").fill("What is the ReAct framework?");
  await page.getByRole("button", { name: /send/i }).click();
  // Citations render as <li> elements (ChatMessage); the corpus is seeded
  // from URLs, not local filenames, so assert on structure, not a guessed
  // string shape.
  await expect(page.locator(".message__citations li").first()).toBeVisible({ timeout: 30_000 });
});

test("answers a corpus-wide synthesis question", async ({ page }) => {
  await page.goto("http://localhost:3000");
  await page
    .getByRole("textbox")
    .fill("What themes connect the reasoning and prompting papers in this corpus?");
  await page.getByRole("button", { name: /send/i }).click();
  await expect(page.locator(".message--assistant").last()).toBeVisible({ timeout: 30_000 });
});

test("refuses a deliberately out-of-corpus question", async ({ page }) => {
  await page.goto("http://localhost:3000");
  await page.getByRole("textbox").fill("What's your return policy on flights to Mars?");
  await page.getByRole("button", { name: /send/i }).click();
  await expect(
    page.getByText("I don't have enough information in the available documents to answer that."),
  ).toBeVisible({ timeout: 30_000 });
});
