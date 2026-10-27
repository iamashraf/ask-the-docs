import react from "@vitejs/plugin-react";
import { defineConfig } from "vitest/config";

export default defineConfig({
  plugins: [react()],
  test: {
    environment: "jsdom",
    setupFiles: ["./vitest.setup.ts"],
    // e2e/ holds Playwright specs (real browser, real stack) -- a
    // different test runner with its own test()/expect(), not jsdom
    // component tests. Excluded here so Vitest's default *.spec.ts
    // pattern doesn't try to run them.
    exclude: ["node_modules/**", "e2e/**"],
  },
});
