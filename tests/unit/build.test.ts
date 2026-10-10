/** The shape of the build, asserted rather than assumed. */

import { existsSync, readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import { describe, expect, it } from "vitest";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..", "..");
const DIST = join(ROOT, "dist");
const built = existsSync(join(DIST, "worker.js"));

describe.runIf(built)("the worker bundle can run from a Blob URL", () => {
  const worker = () => readFileSync(join(DIST, "worker.js"), "utf-8");

  it("has no dynamic import of a module we own", () => {
    // `await import(/*@vite-ignore*/ t)` on a URL it computes at runtime.

    expect(worker()).not.toMatch(/\bimport\s*\(\s*["'`]/);
  });

  it("has no bare or relative import", () => {
    expect(worker()).not.toMatch(/^\s*import\s[^;]*?from\s*["'][^."']/m);
    expect(worker()).not.toMatch(/^\s*import\s[^;]*?from\s*["']\.\.?\//m);
  });

  it("has no node builtin and no createRequire shim", () => {
    expect(worker()).not.toMatch(/from\s*["']node:/);
    expect(worker()).not.toContain("createRequire");
  });

  it("actually contains the runtime, not a reference to it", () => {
    expect(worker().length).toBeGreaterThan(200_000);
  });
});

describe.runIf(existsSync(DIST))("the library entries stay external", () => {
  it("does not inline the runtime into index", () => {
    const index = readFileSync(join(DIST, "index.js"), "utf-8");
    expect(index.length).toBeLessThan(200_000);
  });
});
