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

  const data = await backendResponse.json();

  if (!backendResponse.ok) {
    return Response.json({ error: data.detail ?? "Request failed." }, {
      status: backendResponse.status,
    });
  }

  return Response.json(data, { status: backendResponse.status });
}
