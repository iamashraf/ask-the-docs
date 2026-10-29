import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Produces a minimal, self-contained server bundle (.next/standalone) so
  // the production image doesn't need the full node_modules tree.
  output: "standalone",
  // The OTel SDK's instrumentation packages do dynamic requires internally
  // (to monkey-patch Node's http module), which the standalone bundler's
  // static file-tracing can't follow -- it misses transitive deps like
  // forwarded-parse. Listing them here makes Next require them normally
  // from node_modules at runtime instead, where its tracer correctly
  // copies their real dependency tree into .next/standalone/node_modules.
  serverExternalPackages: [
    "@opentelemetry/sdk-node",
    "@opentelemetry/instrumentation-http",
    "@opentelemetry/instrumentation-undici",
    "@opentelemetry/exporter-trace-otlp-http",
  ],
  // @opentelemetry/instrumentation-http's require("forwarded-parse") lives
  // inside a compiled .mjs file, which the standalone bundler's static file
  // tracer (@vercel/nft) fails to follow even with the package marked
  // external above -- force-include it explicitly.
  outputFileTracingIncludes: {
    "/**": ["./node_modules/forwarded-parse/**"],
  },
};

export default nextConfig;
