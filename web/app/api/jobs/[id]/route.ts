import { proxyToBackend } from "../../../../lib/backend";

export async function GET(request: Request, { params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  return proxyToBackend(`/v1/jobs/${id}`, { method: "GET" });
}
