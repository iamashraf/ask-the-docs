const BACKEND_URL = process.env.BACKEND_URL ?? "http://localhost:8000";

// Shared proxy logic for every /api/* route that forwards to the FastAPI
// backend: a fetch failure or a non-JSON backend body (an unhandled
// exception falls through to a plain-text 500) must degrade to a visible
// JSON error, never an uncaught throw -- see api-ask's own fix history for
// why this matters.
export async function proxyToBackend(path: string, init: RequestInit): Promise<Response> {
  let backendResponse: Response;
  try {
    backendResponse = await fetch(`${BACKEND_URL}${path}`, init);
  } catch {
    return Response.json(
      { error: "Couldn't reach the ask-the-docs backend." },
      { status: 502 },
    );
  }

  let data: unknown;
  try {
    data = await backendResponse.json();
  } catch {
    return Response.json(
      { error: "The backend returned an unexpected response." },
      { status: 502 },
    );
  }

  if (!backendResponse.ok) {
    const detail = (data as { detail?: unknown }).detail;
    return Response.json(
      { error: typeof detail === "string" ? detail : "Request failed." },
      { status: backendResponse.status },
    );
  }

  return Response.json(data, { status: backendResponse.status });
}
