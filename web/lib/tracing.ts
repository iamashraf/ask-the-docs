import { NodeSDK } from "@opentelemetry/sdk-node";
import { OTLPTraceExporter } from "@opentelemetry/exporter-trace-otlp-http";
import { HttpInstrumentation } from "@opentelemetry/instrumentation-http";
import { UndiciInstrumentation } from "@opentelemetry/instrumentation-undici";
import { resourceFromAttributes } from "@opentelemetry/resources";
import { ATTR_SERVICE_NAME } from "@opentelemetry/semantic-conventions";

export function registerTracing(): void {
  const sdk = new NodeSDK({
    resource: resourceFromAttributes({ [ATTR_SERVICE_NAME]: "ask-the-docs-web" }),
    traceExporter: new OTLPTraceExporter({
      url: process.env.OTEL_EXPORTER_ENDPOINT ?? "http://jaeger:4318/v1/traces",
    }),
    // HttpInstrumentation only patches Node's legacy http/https modules;
    // proxyToBackend's fetch() calls run on undici, which needs its own
    // instrumentation to inject the traceparent header.
    instrumentations: [new HttpInstrumentation(), new UndiciInstrumentation()],
  });
  sdk.start();
}
