import { proxyToBackend } from "../../../lib/backend";

export async function POST() {
  return proxyToBackend("/v1/index", { method: "POST" });
}
