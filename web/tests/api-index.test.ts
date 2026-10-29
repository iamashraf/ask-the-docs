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
        new Response(JSON.stringify({ job_id: "job-1", status: "queued" }), { status: 202 }),
      ),
    );

    const res = await POST();

    expect(res.status).toBe(202);
    expect(await res.json()).toEqual({ job_id: "job-1", status: "queued" });
  });

  test("returns a 502 error when the backend is unreachable", async () => {
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new Error("fetch failed")));

    const res = await POST();

    expect(res.status).toBe(502);
    expect((await res.json()).error).toBeTruthy();
  });
});
