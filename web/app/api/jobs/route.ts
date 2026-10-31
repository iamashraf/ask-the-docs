import { proxyToBackend } from "../../../lib/backend";

export async function GET(request: Request) {
  const { search } = new URL(request.url);
  return proxyToBackend(`/v1/jobs${search}`, { method: "GET" });
}
