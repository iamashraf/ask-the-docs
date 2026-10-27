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
});
