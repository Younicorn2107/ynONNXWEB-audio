/**
 * The transcription worker decides where its runtime and its models come from once: a later request
 * that names another `wasmPaths` (where onnxruntime-web imports its glue from) or another model
 * source is refused, and a location that is code handed over inline is refused outright (security review
 * review, E.15).
 */

import { afterAll, beforeAll, describe, expect, it, vi } from "vitest";

import { ConfigurationError, TaskClient, transportPair, workerHost } from "@youni/onnxweb-core";

let client: TaskClient;

beforeAll(async () => {
  vi.stubGlobal("addEventListener", () => undefined);
  vi.stubGlobal("removeEventListener", () => undefined);
  vi.stubGlobal("postMessage", () => undefined);
  await import("../../src/worker/entry.js");
  const [hostEnd, clientEnd] = transportPair();
  workerHost.serve(hostEnd);
  client = new TaskClient(clientEnd);
});

afterAll(() => vi.unstubAllGlobals());

const first = { model: { name: "tiny", computeType: "int8" }, resolverBase: "http://127.0.0.1:9/models/", wasmPaths: "http://127.0.0.1:9/ort/" };

describe("the transcription worker's locations", () => {
  it("refuses a wasmPaths that is code rather than a URL", async () => {
    await expect(client.call("resolve", { ...first, wasmPaths: "data:text/javascript,postMessage(1)" })).rejects.toBeInstanceOf(ConfigurationError);
  });

  it("keeps the first request's sources and refuses others", async () => {
    await client.call("resolve", first).catch(() => undefined);
    for (const change of [{ wasmPaths: "https://evil.example/" }, { endpoint: "https://evil.example" }, { resolverBase: "https://evil.example/models/" }, { hub: { repo: "attacker/model" } }]) {
      await expect(client.call("resolve", { ...first, ...change })).rejects.toThrow(/cannot change/);
    }
    await expect(client.call("analyze_audio", { audio: new ArrayBuffer(4), analyses: ["music"], wasmPaths: "https://evil.example/" })).rejects.toThrow(/cannot change/);
  }, 30_000);
});
