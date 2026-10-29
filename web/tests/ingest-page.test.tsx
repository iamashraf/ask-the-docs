import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, test, vi } from "vitest";
import IngestPage, { pollJob } from "../app/ingest/page";

function jsonResponse(body: unknown, status = 200) {
  return new Response(JSON.stringify(body), { status });
}

describe("IngestPage", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  });

  test("renders a heading and a link back to the chat page", () => {
    render(<IngestPage />);
    expect(screen.getByRole("heading", { level: 1 })).toBeInTheDocument();
    expect(screen.getByRole("link", { name: /chat/i })).toHaveAttribute("href", "/");
  });

  test("ingests a single source and shows the result once the job finishes", async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(jsonResponse({ job_id: "job-1", status: "queued" }, 202))
      .mockResolvedValueOnce(jsonResponse({ status: "done", result: { documents_ingested: 2 } }));
    vi.stubGlobal("fetch", fetchMock);

    render(<IngestPage />);
    await userEvent.type(
      screen.getByRole("textbox", { name: /source/i }),
      "https://example.com/paper.pdf",
    );
    await userEvent.click(screen.getByRole("button", { name: /^ingest$/i }));

    expect(await screen.findByText(/2 documents ingested/i)).toBeInTheDocument();
    expect(fetchMock).toHaveBeenNthCalledWith(2, "/api/jobs/job-1");
  });

  test("shows a visible error when the initial ingest request fails", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(jsonResponse({ error: "Unsupported file type" }, 400)));

    render(<IngestPage />);
    await userEvent.type(screen.getByRole("textbox", { name: /source/i }), "bad.exe");
    await userEvent.click(screen.getByRole("button", { name: /^ingest$/i }));

    expect(await screen.findByText(/Unsupported file type/i)).toBeInTheDocument();
  });

  test("shows a visible error when the job itself fails", async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(jsonResponse({ job_id: "job-1", status: "queued" }, 202))
      .mockResolvedValueOnce(jsonResponse({ status: "failed", error: "Unsupported file type" }));
    vi.stubGlobal("fetch", fetchMock);

    render(<IngestPage />);
    await userEvent.type(screen.getByRole("textbox", { name: /source/i }), "https://example.com/x.pdf");
    await userEvent.click(screen.getByRole("button", { name: /^ingest$/i }));

    expect(await screen.findByText(/Unsupported file type/i)).toBeInTheDocument();
  });

  test("the ingest button is disabled for empty input", () => {
    render(<IngestPage />);
    expect(screen.getByRole("button", { name: /^ingest$/i })).toBeDisabled();
  });

  test("re-indexes everything and shows the result once the job finishes", async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(jsonResponse({ job_id: "job-2", status: "queued" }, 202))
      .mockResolvedValueOnce(
        jsonResponse({ status: "done", result: { documents_ingested: 10, files_found: 12 } }),
      );
    vi.stubGlobal("fetch", fetchMock);

    render(<IngestPage />);
    await userEvent.click(screen.getByRole("button", { name: /re-index everything/i }));

    expect(await screen.findByText(/10 documents ingested.*12 files found/i)).toBeInTheDocument();
  });

  test("shows a visible indicator while a job is still queued, then the result once it's done", async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(jsonResponse({ job_id: "job-3", status: "queued" }, 202))
      .mockResolvedValueOnce(jsonResponse({ status: "queued" }))
      .mockResolvedValueOnce(jsonResponse({ status: "processing" }))
      .mockResolvedValueOnce(jsonResponse({ status: "done", result: { documents_ingested: 1 } }));
    vi.stubGlobal("fetch", fetchMock);
    const user = userEvent.setup({ delay: null });

    render(<IngestPage />);
    await user.type(screen.getByRole("textbox", { name: /source/i }), "https://example.com/x.pdf");
    await user.click(screen.getByRole("button", { name: /^ingest$/i }));

    expect(await screen.findByText(/ingesting/i)).toBeInTheDocument();

    await vi.advanceTimersByTimeAsync(1000);
    await vi.advanceTimersByTimeAsync(1000);

    expect(await screen.findByText(/1 documents ingested/i)).toBeInTheDocument();
    expect(screen.queryByText(/ingesting…/i)).not.toBeInTheDocument();
    vi.useRealTimers();
  });

  test("shows a visible error when re-indexing's initial request fails", async () => {
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new Error("network down")));

    render(<IngestPage />);
    await userEvent.click(screen.getByRole("button", { name: /re-index everything/i }));

    expect(await screen.findByText(/couldn't reach the server/i)).toBeInTheDocument();
  });

});

describe("pollJob", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
    vi.useRealTimers();
  });

  test("returns a timeout error if the job never reaches done or failed", async () => {
    vi.useFakeTimers();
    // mockImplementation, not mockResolvedValue: a Response body can only be
    // read once, so reusing one Response instance across every poll call
    // throws "Body is unusable" on the second .json() -- each call needs
    // its own fresh Response.
    vi.stubGlobal("fetch", vi.fn().mockImplementation(() => Promise.resolve(jsonResponse({ status: "processing" }))));

    const resultPromise = pollJob("job-x");
    await vi.advanceTimersByTimeAsync(5 * 60 * 1000 + 1000);
    const result = await resultPromise;

    expect(result).toEqual({ kind: "error", text: "Timed out waiting for ingestion to finish." });
  });

  test("returns ok once the job reaches done, without waiting for the full timeout", async () => {
    vi.useFakeTimers();
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(jsonResponse({ status: "queued" }))
      .mockResolvedValueOnce(jsonResponse({ status: "done", result: { documents_ingested: 3 } }));
    vi.stubGlobal("fetch", fetchMock);

    const resultPromise = pollJob("job-y");
    await vi.advanceTimersByTimeAsync(1000);
    const result = await resultPromise;

    expect(result).toEqual({ kind: "ok", text: "3 documents ingested." });
  });

  test("returns an error immediately if the job status lookup itself fails", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(jsonResponse({ error: "job store down" }, 503)));

    const result = await pollJob("job-z");

    expect(result).toEqual({ kind: "error", text: "job store down" });
  });
});
