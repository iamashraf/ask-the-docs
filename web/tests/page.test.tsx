import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, test, vi } from "vitest";
import Page from "../app/page";

describe("Page", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  test("renders the chat page heading", () => {
    render(<Page />);
    expect(screen.getByRole("heading")).toBeInTheDocument();
  });

  test("sends a question and renders the answer", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        new Response(
          JSON.stringify({
            answer: "It's a ranking function.",
            citations: ["bm25.pdf"],
          }),
          { status: 200 },
        ),
      ),
    );

    render(<Page />);
    await userEvent.type(screen.getByRole("textbox"), "What is BM25?");
    await userEvent.click(screen.getByRole("button", { name: /send/i }));

    expect(await screen.findByText("It's a ranking function.")).toBeInTheDocument();
    expect(screen.getByText("bm25.pdf")).toBeInTheDocument();
  });

  test("shows an error message when the request fails", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        new Response(JSON.stringify({ error: "Backend unreachable" }), {
          status: 502,
        }),
      ),
    );

    render(<Page />);
    await userEvent.type(screen.getByRole("textbox"), "anything");
    await userEvent.click(screen.getByRole("button", { name: /send/i }));

    expect(await screen.findByText("Backend unreachable")).toBeInTheDocument();
  });

  test("the send button is disabled for empty input", () => {
    render(<Page />);
    expect(screen.getByRole("button", { name: /send/i })).toBeDisabled();
  });

  test("the send button stays disabled for whitespace-only input", async () => {
    render(<Page />);
    await userEvent.type(screen.getByRole("textbox"), "   ");
    expect(screen.getByRole("button", { name: /send/i })).toBeDisabled();
  });
});
