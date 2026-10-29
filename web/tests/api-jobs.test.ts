import { afterEach, describe, expect, test, vi } from "vitest";
import { GET } from "../app/api/jobs/[id]/route";

function makeParams(id: string) {
  return { params: Promise.resolve({ id }) };
}

describe("GET /api/jobs/[id]", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  test("proxies a job status response", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        new Response(JSON.stringify({ job_id: "job-1", status: "done", result: { documents_ingested: 2 } }), {
          status: 200,
        }),
      ),
    );

    const res = await GET(new Request("http://localhost/api/jobs/job-1"), makeParams("job-1"));

    expect(res.status).toBe(200);
    expect((await res.json()).status).toBe("done");
  });

  test("surfaces a 404 for an unknown job", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(new Response(JSON.stringify({ detail: "Job not found" }), { status: 404 })),
    );

    const res = await GET(new Request("http://localhost/api/jobs/nope"), makeParams("nope"));

    expect(res.status).toBe(404);
  });
});
