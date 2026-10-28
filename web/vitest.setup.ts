import { cleanup } from "@testing-library/react";
import "@testing-library/jest-dom/vitest";
import { afterEach } from "vitest";

afterEach(() => {
  cleanup();
});

// jsdom doesn't implement scrollIntoView at all -- stub it globally so
// components that call it don't crash; individual tests spy on it to
// assert it was called.
Element.prototype.scrollIntoView = Element.prototype.scrollIntoView ?? (() => {});
