import { NodeSDK } from "@opentelemetry/sdk-node";
import { OTLPTraceExporter } from "@opentelemetry/exporter-trace-otlp-http";
import { HttpInstrumentation } from "@opentelemetry/instrumentation-http";
import { resourceFromAttributes } from "@opentelemetry/resources";
import { ATTR_SERVICE_NAME } from "@opentelemetry/semantic-conventions";

export function registerTracing(): void {
  const sdk = new NodeSDK({
    resource: resourceFromAttributes({ [ATTR_SERVICE_NAME]: "ask-the-docs-web" }),
    traceExporter: new OTLPTraceExporter({
      url: process.env.OTEL_EXPORTER_ENDPOINT ?? "http://jaeger:4318/v1/traces",
    }),
    instrumentations: [new HttpInstrumentation()],
  });
  sdk.start();
}
