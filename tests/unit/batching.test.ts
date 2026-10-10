import { describe, expect, it } from "vitest";

import { DEFAULT_MAX_WINDOWS, maxWindows, sectionBudget, windowBytes } from "../../src/whisper/batching.js";

const geometry = (numLayers: number, numHeads: number) => ({ numLayers, numHeads, headDim: 64, hasCrossAttention: true, dtype: "float32" });
const LARGE_V3 = geometry(32, 20);
const TURBO = geometry(4, 20);
const TINY = geometry(4, 6);

describe("wave size", () => {
  it("costs large-v3 most of a gigabyte per window, the cross cache first", () => {
    const mb = windowBytes(LARGE_V3) / 2 ** 20;
    expect(mb).toBeGreaterThan(800);
    expect(mb).toBeLessThan(900);
  });

  it("holds Python's 8 when memory is not the limit", () => {
    expect(maxWindows({ backend: "wasm", kv: TINY, weightsMb: 40, heapCeilingMb: 4096 })).toBe(DEFAULT_MAX_WINDOWS);
    expect(maxWindows({ backend: "wasm", kv: TURBO, weightsMb: 1100, heapCeilingMb: 4096 })).toBe(DEFAULT_MAX_WINDOWS);
  });

  it("drops to what the WebAssembly heap holds beside the weights", () => {
    expect(maxWindows({ backend: "wasm", kv: LARGE_V3, weightsMb: 1600, heapCeilingMb: 4096 })).toBe(2);
    expect(maxWindows({ backend: "wasm", kv: LARGE_V3, weightsMb: 3000, heapCeilingMb: 4096 })).toBe(1);
  });

  it("keeps a batch's attention scores inside one WebGPU binding", () => {
    expect(maxWindows({ backend: "webgpu", kv: LARGE_V3, weightsMb: 1600, heapCeilingMb: 4096, maxBindingBytes: 2 ** 30 })).toBe(5);
    expect(maxWindows({ backend: "webgpu", kv: LARGE_V3, weightsMb: 1600, heapCeilingMb: 4096, maxBindingBytes: 128 * 2 ** 20 })).toBe(1);
  });

  it("does not size what it cannot see", () => {
    expect(maxWindows({ backend: "wasm", kv: null, weightsMb: 0, heapCeilingMb: 4096 })).toBe(DEFAULT_MAX_WINDOWS);
  });

  it("cuts two sections per audio, no more than a wave holds", () => {
    expect(sectionBudget(8, 1)).toBe(2);
    expect(sectionBudget(8, 10)).toBe(8);
    expect(sectionBudget(1, 3)).toBe(1);
  });
});
