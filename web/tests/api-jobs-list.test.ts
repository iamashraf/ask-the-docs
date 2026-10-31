import { afterEach, describe, expect, test, vi } from "vitest";
import { GET } from "../app/api/jobs/route";

describe("GET /api/jobs", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  test("proxies the job list response", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ jobs: [{ job_id: "job-1", status: "done" }] }), { status: 200 }),
    );
    vi.stubGlobal("fetch", fetchMock);

    const res = await GET(new Request("http://localhost/api/jobs"));

    expect(res.status).toBe(200);
    expect((await res.json()).jobs).toHaveLength(1);
    expect(fetchMock).toHaveBeenCalledWith("http://localhost:8000/v1/jobs", { method: "GET" });
  });

  test("passes through a limit query param", async () => {
    const fetchMock = vi.fn().mockResolvedValue(new Response(JSON.stringify({ jobs: [] }), { status: 200 }));
    vi.stubGlobal("fetch", fetchMock);

    await GET(new Request("http://localhost/api/jobs?limit=5"));

    expect(fetchMock).toHaveBeenCalledWith("http://localhost:8000/v1/jobs?limit=5", { method: "GET" });
  });

  test("surfaces a backend error", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response(JSON.stringify({ detail: "bad limit" }), { status: 422 })));

    const res = await GET(new Request("http://localhost/api/jobs?limit=0"));

    expect(res.status).toBe(422);
  });
});
