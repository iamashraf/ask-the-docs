import { afterEach, describe, expect, test, vi } from "vitest";
import { POST } from "../app/api/ask/route";

function makeRequest(body: unknown) {
  return new Request("http://localhost/api/ask", {
    method: "POST",
    body: JSON.stringify(body),
  });
}

describe("POST /api/ask", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  test("proxies a successful backend response", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        new Response(JSON.stringify({ answer: "A", citations: ["x"] }), {
          status: 200,
        }),
      ),
    );

    const res = await POST(makeRequest({ query: "q" }));

    expect(res.status).toBe(200);
    expect(await res.json()).toEqual({ answer: "A", citations: ["x"] });
  });

  test("surfaces a backend non-2xx as an error with the same status", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        new Response(JSON.stringify({ detail: "Ollama unreachable" }), {
          status: 503,
        }),
      ),
    );

    const res = await POST(makeRequest({ query: "q" }));

    expect(res.status).toBe(503);
    expect((await res.json()).error).toBe("Ollama unreachable");
  });

  test("returns a 502 error when the backend is unreachable", async () => {
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new Error("fetch failed")));

    const res = await POST(makeRequest({ query: "q" }));

    expect(res.status).toBe(502);
    expect((await res.json()).error).toBeTruthy();
  });

  test("returns a 502 error when the backend's body isn't valid JSON", async () => {
    // e.g. an unhandled exception in FastAPI that falls through to
    // Starlette's default plain-text 500 handler.
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(new Response("Internal Server Error", { status: 500 })),
    );

    const res = await POST(makeRequest({ query: "q" }));

    expect(res.status).toBe(502);
    expect((await res.json()).error).toBeTruthy();
  });

  test("falls back to a generic message when a non-2xx detail isn't a string", async () => {
    // FastAPI's 422 validation errors shape `detail` as an array of
    // objects, not a string -- must not be passed through as-is (it would
    // crash the page trying to render an object/array as text).
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        new Response(JSON.stringify({ detail: [{ msg: "field required" }] }), {
          status: 422,
        }),
      ),
    );

    const res = await POST(makeRequest({ query: "q" }));

    expect(res.status).toBe(422);
    expect(typeof (await res.json()).error).toBe("string");
  });
});
