/** How many windows one decoding wave holds, and how many sections to cut. */

import type { KVGeometry } from "@youni/onnxweb-core/core";

/** Ceiling on one wave, as in Python. */
export const DEFAULT_MAX_WINDOWS = 8;

/** Encoder positions after the two stride-2 convolutions: fixed by the architecture. */
export const ENCODER_POSITIONS = 1500;

/** Whisper's decoder length, prompt included. */
const MAX_LENGTH = 448;

/** Share of the budget a wave may take; the rest is what the runtime keeps for itself. */
const MARGIN = 0.9;

/** Loaded weights against their size on disk (session copies, arena). */
const WEIGHT_OVERHEAD = 1.25;

const ELEMENT_BYTES: Readonly<Record<string, number>> = { float32: 4, float16: 2 };

/**
 * Bytes one window costs at its peak: encoder activations, one layer's attention scores, the
 * caches.
 */
export function windowBytes(kv: KVGeometry): number {
  const dModel = kv.numHeads * kv.headDim;
  const element = ELEMENT_BYTES[kv.dtype] ?? 4;

  const activations = ENCODER_POSITIONS * dModel * 4 * 4;
  const attention = kv.numHeads * ENCODER_POSITIONS * ENCODER_POSITIONS * 4;
  const perLayer = 2 * kv.numHeads * kv.headDim * element;
  const selfCache = kv.numLayers * perLayer * MAX_LENGTH;
  const crossCache = kv.hasCrossAttention ? kv.numLayers * perLayer * ENCODER_POSITIONS : 0;
  return activations + attention + selfCache + crossCache;
}

export interface WaveBudget {
  backend: string;
  /** The model's KV geometry; without it the wave is not sized. */
  kv: KVGeometry | null;
  /** The weights' size on disk, in MiB. */
  weightsMb: number;
  /** The WebAssembly heap ceiling, in MiB. */
  heapCeilingMb: number;
  /** WebGPU's largest storage binding, in bytes. */
  maxBindingBytes?: number | null;
}

/** How many windows one wave may hold. */
export function maxWindows(budget: WaveBudget): number {
  if (!budget.kv || budget.kv.numHeads <= 0) return DEFAULT_MAX_WINDOWS;
  const perWindow = windowBytes(budget.kv);
  let fit = DEFAULT_MAX_WINDOWS;
  if (budget.backend === "wasm") {
    const room = budget.heapCeilingMb * MARGIN - budget.weightsMb * WEIGHT_OVERHEAD;
    fit = Math.floor((room * 1024 * 1024) / perWindow);
  } else if (budget.backend === "webgpu" && budget.maxBindingBytes) {
    const scores = budget.kv.numHeads * ENCODER_POSITIONS * ENCODER_POSITIONS * 4;
    fit = Math.floor(budget.maxBindingBytes / scores);
  }
  return Math.max(1, Math.min(DEFAULT_MAX_WINDOWS, fit));
}

/** Sections to cut the audios into: two per audio, as many as one wave holds at most. */
export function sectionBudget(windows: number, audioCount: number): number {
  return Math.max(1, Math.min(windows, audioCount * 2));
}
