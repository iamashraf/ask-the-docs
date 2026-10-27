import { afterEach, describe, expect, test, vi } from "vitest";
import { POST } from "../app/api/ingest/route";

function makeRequest(body: unknown) {
  return new Request("http://localhost/api/ingest", {
    method: "POST",
    body: JSON.stringify(body),
  });
}

describe("POST /api/ingest", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  test("proxies a successful ingest response", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        new Response(JSON.stringify({ documents_ingested: 3 }), { status: 200 }),
      ),
    );

    const res = await POST(makeRequest({ source: "https://example.com/paper.pdf" }));

    expect(res.status).toBe(200);
    expect(await res.json()).toEqual({ documents_ingested: 3 });
  });

  test("surfaces a backend non-2xx as an error", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        new Response(JSON.stringify({ detail: "Unsupported file type" }), { status: 400 }),
      ),
    );

    const res = await POST(makeRequest({ source: "bad.exe" }));

    expect(res.status).toBe(400);
    expect((await res.json()).error).toBe("Unsupported file type");
  });

  test("returns a 502 error when the backend is unreachable", async () => {
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new Error("fetch failed")));

    const res = await POST(makeRequest({ source: "x" }));

    expect(res.status).toBe(502);
    expect((await res.json()).error).toBeTruthy();
  });
});
