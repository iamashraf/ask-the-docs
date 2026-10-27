import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Produces a minimal, self-contained server bundle (.next/standalone) so
  // the production image doesn't need the full node_modules tree.
  output: "standalone",
};

export default nextConfig;
