const BACKEND_URL = process.env.BACKEND_URL ?? "http://localhost:8000";

// FastAPI sends a plain string for errors it raises itself, and a list of
// {loc, msg} objects for request validation (422). Show the first problem as
// "field: reason" so the user can see what to change.
function errorMessage(detail: unknown): string {
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) {
    const first = detail[0] as { loc?: unknown; msg?: unknown } | undefined;
    if (typeof first?.msg === "string") {
      const field = Array.isArray(first.loc) ? first.loc[first.loc.length - 1] : undefined;
      return typeof field === "string" ? `${field}: ${first.msg}` : first.msg;
    }
  }
  return "Request failed.";
}

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
    return Response.json(
      { error: errorMessage((data as { detail?: unknown }).detail) },
      { status: backendResponse.status },
    );
  }

  return Response.json(data, { status: backendResponse.status });
}
