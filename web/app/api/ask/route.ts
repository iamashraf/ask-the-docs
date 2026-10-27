const BACKEND_URL = process.env.BACKEND_URL ?? "http://localhost:8000";

export async function POST(request: Request) {
  const body = await request.text();

  let backendResponse: Response;
  try {
    backendResponse = await fetch(`${BACKEND_URL}/v1/ask`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body,
    });
  } catch {
    return Response.json(
      { error: "Couldn't reach the research-assistant backend." },
      { status: 502 },
    );
  }

  let data: unknown;
  try {
    data = await backendResponse.json();
  } catch {
    // An unhandled backend exception falls through to a plain-text 500,
    // not JSON -- degrade to a visible error rather than letting this
    // throw and produce no response at all.
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
