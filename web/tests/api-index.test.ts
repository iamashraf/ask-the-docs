import { afterEach, describe, expect, test, vi } from "vitest";
import { POST } from "../app/api/index/route";

describe("POST /api/index", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  test("proxies a successful bulk-index response", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        new Response(
          JSON.stringify({ documents_ingested: 5, files_found: 7 }),
          { status: 200 },
        ),
      ),
    );

    const res = await POST();

    expect(res.status).toBe(200);
    expect(await res.json()).toEqual({ documents_ingested: 5, files_found: 7 });
  });

  test("returns a 502 error when the backend is unreachable", async () => {
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new Error("fetch failed")));

    const res = await POST();

    expect(res.status).toBe(502);
    expect((await res.json()).error).toBeTruthy();
  });
});
