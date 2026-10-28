import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, test, vi } from "vitest";
import IngestPage from "../app/ingest/page";

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

  test("ingests a single source and shows the result", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        new Response(JSON.stringify({ documents_ingested: 2 }), { status: 200 }),
      ),
    );

    render(<IngestPage />);
    await userEvent.type(
      screen.getByRole("textbox", { name: /source/i }),
      "https://example.com/paper.pdf",
    );
    await userEvent.click(screen.getByRole("button", { name: /^ingest$/i }));

    expect(await screen.findByText(/2 documents ingested/i)).toBeInTheDocument();
  });

  test("shows a visible error when ingesting a source fails", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        new Response(JSON.stringify({ error: "Unsupported file type" }), { status: 400 }),
      ),
    );

    render(<IngestPage />);
    await userEvent.type(screen.getByRole("textbox", { name: /source/i }), "bad.exe");
    await userEvent.click(screen.getByRole("button", { name: /^ingest$/i }));

    expect(await screen.findByText(/Unsupported file type/i)).toBeInTheDocument();
  });

  test("the ingest button is disabled for empty input", () => {
    render(<IngestPage />);
    expect(screen.getByRole("button", { name: /^ingest$/i })).toBeDisabled();
  });

  test("re-indexes everything and shows the result", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        new Response(
          JSON.stringify({ documents_ingested: 10, files_found: 12 }),
          { status: 200 },
        ),
      ),
    );

    render(<IngestPage />);
    await userEvent.click(screen.getByRole("button", { name: /re-index everything/i }));

    expect(await screen.findByText(/10 documents ingested.*12 files found/i)).toBeInTheDocument();
  });

  test("shows a visible indicator while ingesting a source is in flight", async () => {
    let resolveFetch: (value: Response) => void;
    vi.stubGlobal(
      "fetch",
      vi.fn().mockReturnValue(
        new Promise<Response>((resolve) => {
          resolveFetch = resolve;
        }),
      ),
    );

    render(<IngestPage />);
    await userEvent.type(screen.getByRole("textbox", { name: /source/i }), "https://example.com/x.pdf");
    await userEvent.click(screen.getByRole("button", { name: /^ingest$/i }));

    expect(await screen.findByText(/ingesting/i)).toBeInTheDocument();

    resolveFetch!(new Response(JSON.stringify({ documents_ingested: 1 }), { status: 200 }));

    expect(await screen.findByText(/1 documents ingested/i)).toBeInTheDocument();
    expect(screen.queryByText(/ingesting…/i)).not.toBeInTheDocument();
  });

  test("shows a visible indicator while re-indexing is in flight", async () => {
    let resolveFetch: (value: Response) => void;
    vi.stubGlobal(
      "fetch",
      vi.fn().mockReturnValue(
        new Promise<Response>((resolve) => {
          resolveFetch = resolve;
        }),
      ),
    );

    render(<IngestPage />);
    await userEvent.click(screen.getByRole("button", { name: /re-index everything/i }));

    expect(await screen.findByText(/re-indexing/i)).toBeInTheDocument();

    resolveFetch!(
      new Response(JSON.stringify({ documents_ingested: 1, files_found: 1 }), { status: 200 }),
    );

    expect(await screen.findByText(/1 documents ingested/i)).toBeInTheDocument();
  });

  test("shows a visible error when re-indexing fails", async () => {
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new Error("network down")));

    render(<IngestPage />);
    await userEvent.click(screen.getByRole("button", { name: /re-index everything/i }));

    expect(await screen.findByText(/couldn't reach the server/i)).toBeInTheDocument();
  });
});
