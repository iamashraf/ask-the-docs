import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, test, vi } from "vitest";
import Page from "../app/page";

describe("Page", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
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

    expect(await screen.findByText(/Backend unreachable/)).toBeInTheDocument();
  });

  test("shows an error message when fetch itself rejects (client can't reach /api/ask)", async () => {
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new Error("network down")));

    render(<Page />);
    await userEvent.type(screen.getByRole("textbox"), "anything");
    await userEvent.click(screen.getByRole("button", { name: /send/i }));

    expect(await screen.findByText(/couldn't reach the server/i)).toBeInTheDocument();
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

  test("shows an empty-state prompt before any message is sent", () => {
    render(<Page />);
    expect(screen.getByText(/ask a question/i)).toBeInTheDocument();
  });

  test("shows a thinking indicator while a request is in flight, then replaces it with the answer", async () => {
    let resolveFetch: (value: Response) => void;
    vi.stubGlobal(
      "fetch",
      vi.fn().mockReturnValue(
        new Promise<Response>((resolve) => {
          resolveFetch = resolve;
        }),
      ),
    );

    render(<Page />);
    await userEvent.type(screen.getByRole("textbox"), "What is BM25?");
    await userEvent.click(screen.getByRole("button", { name: /send/i }));

    expect(await screen.findByText(/thinking/i)).toBeInTheDocument();

    resolveFetch!(new Response(JSON.stringify({ answer: "A ranking function.", citations: [] }), { status: 200 }));

    expect(await screen.findByText("A ranking function.")).toBeInTheDocument();
    expect(screen.queryByText(/thinking/i)).not.toBeInTheDocument();
  });

  test("shows an elapsed-time counter while waiting, reset for the next question", async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    let resolveFetch: (value: Response) => void;
    vi.stubGlobal(
      "fetch",
      vi.fn().mockReturnValue(
        new Promise<Response>((resolve) => {
          resolveFetch = resolve;
        }),
      ),
    );

    render(<Page />);
    await userEvent.type(screen.getByRole("textbox"), "What is BM25?");
    await userEvent.click(screen.getByRole("button", { name: /send/i }));

    expect(screen.getByText("0s")).toBeInTheDocument();
    await vi.advanceTimersByTimeAsync(3000);
    expect(screen.getByText("3s")).toBeInTheDocument();

    resolveFetch!(new Response(JSON.stringify({ answer: "A ranking function.", citations: [] }), { status: 200 }));
    await screen.findByText("A ranking function.");

    expect(screen.queryByText(/^\d+s$/)).not.toBeInTheDocument();
    vi.useRealTimers();
  });

  test("auto-focuses the question input on load", () => {
    render(<Page />);
    expect(screen.getByRole("textbox")).toHaveFocus();
  });

  test("scrolls the message log to the latest message when one arrives", async () => {
    const scrollIntoView = vi.spyOn(Element.prototype, "scrollIntoView");

    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        new Response(JSON.stringify({ answer: "Answer.", citations: [] }), { status: 200 }),
      ),
    );

    render(<Page />);
    await userEvent.type(screen.getByRole("textbox"), "anything");
    await userEvent.click(screen.getByRole("button", { name: /send/i }));
    await screen.findByText("Answer.");

    expect(scrollIntoView).toHaveBeenCalled();
  });
});
