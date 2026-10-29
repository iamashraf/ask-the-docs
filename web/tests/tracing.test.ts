import { afterEach, describe, expect, test, vi } from "vitest";

describe("registerTracing", () => {
  afterEach(() => {
    vi.doUnmock("@opentelemetry/sdk-node");
    vi.resetModules();
  });

  test("starts an SDK configured with the OTLP endpoint and service name", async () => {
    const startMock = vi.fn();
    vi.doMock("@opentelemetry/sdk-node", () => ({
      NodeSDK: class {
        constructor(config: unknown) {
          this.config = config;
        }
        config: unknown;
        start() {
          startMock(this.config);
        }
      },
    }));

    const { registerTracing } = await import("../lib/tracing");
    registerTracing();

    expect(startMock).toHaveBeenCalled();
  });
});

describe("register", () => {
  afterEach(() => {
    vi.doUnmock("../lib/tracing");
    vi.resetModules();
    delete process.env.NEXT_RUNTIME;
  });

  test("skips tracing setup outside the nodejs runtime", async () => {
    // Review Focus: instrumentation.ts must tolerate running (or being
    // imported) outside a real Node server process -- the Edge runtime is
    // the one Next itself can invoke this hook under.
    process.env.NEXT_RUNTIME = "edge";
    const registerTracingMock = vi.fn();
    vi.doMock("../lib/tracing", () => ({ registerTracing: registerTracingMock }));

    const { register } = await import("../instrumentation");
    await register();

    expect(registerTracingMock).not.toHaveBeenCalled();
  });
});
