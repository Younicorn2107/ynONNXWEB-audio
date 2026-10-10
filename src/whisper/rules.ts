/** Whisper's decoding constraints, as a logits processor of the core's loop. */

import { NEG_INF, type Logits, type LogitsProcessor, type Sequences } from "@youni/onnxweb-core/core";

import { TIME_PRECISION } from "./tokens.js";

/** What the rules need to know of the vocabulary (a `WhisperTokens` has it). */
export interface TimestampVocabulary {
  readonly timestampBegin: number | null;
  readonly eot: number;
  suppressed(options?: { allowTimestamps?: boolean }): number[];
}

export interface TimestampRulesOptions {
  /** The first timestamp may not exceed this (seconds). Python's default: 1.0. */
  maxInitialTimestamp?: number | null;
  withoutTimestamps?: boolean;
  /** Extra ids the caller suppresses, after the structural ones. */
  suppressTokens?: readonly number[];
}

export class TimestampRules implements LogitsProcessor {
  readonly suppress: number[];
  readonly begin: number;
  readonly eot: number;
  readonly withoutTimestamps: boolean;
  readonly maxInitialIndex: number | null;

  constructor(tokens: TimestampVocabulary, options: TimestampRulesOptions = {}) {
    this.withoutTimestamps = options.withoutTimestamps ?? false;
    this.begin = tokens.timestampBegin ?? Number.MAX_SAFE_INTEGER;
    this.eot = tokens.eot;

    this.suppress = [
      ...new Set([
        ...tokens.suppressed({ allowTimestamps: !this.withoutTimestamps }),
        ...(options.suppressTokens ?? []).filter((t) => t >= 0),
      ]),
    ].sort((a, b) => a - b);
    const initial = options.maxInitialTimestamp === undefined ? 1.0 : options.maxInitialTimestamp;
    this.maxInitialIndex = initial !== null && initial > 0 ? Math.round(initial / TIME_PRECISION) : null;
  }

  process(logits: Logits, sequences: Sequences): void {
    const { vocab } = logits;
    const begin = this.begin;
    if (begin >= vocab) return;
    for (let r = 0; r < logits.rows; r += 1) {
      const row = logits.data.subarray(r * vocab, (r + 1) * vocab);
      for (const id of this.suppress) if (id < vocab) row[id] = NEG_INF;
      if (this.withoutTimestamps) {
        row.fill(NEG_INF, begin);
        continue;
      }
      this.#rules(row, sequences[r] ?? [], begin, vocab);
    }
  }

  #rules(row: Float32Array, emitted: readonly number[], begin: number, vocab: number): void {
    const stamps = emitted.filter((t) => t >= begin);
    const lastWasStamp = emitted.length > 0 && emitted[emitted.length - 1]! >= begin;

    const penultimateWasStamp = emitted.length < 2 || emitted[emitted.length - 2]! >= begin;

    if (lastWasStamp) {
      if (penultimateWasStamp) row.fill(NEG_INF, begin);
      else row.fill(NEG_INF, 0, this.eot);
    }

    if (stamps.length > 0) {
      const last = stamps[stamps.length - 1]!;
      const floor = lastWasStamp && !penultimateWasStamp ? last : last + 1;
      if (floor > begin) row.fill(NEG_INF, begin, Math.min(floor, vocab));
    }

    if (emitted.length === 0 && this.maxInitialIndex !== null) {
      const ceiling = begin + this.maxInitialIndex + 1;
      if (ceiling < vocab) row.fill(NEG_INF, ceiling);
    }

    let max = NEG_INF;
    let anyStamp = false;
    for (let i = 0; i < vocab; i += 1) {
      const v = row[i]!;
      if (v > max) max = v;
      if (i >= begin && v !== NEG_INF) anyStamp = true;
    }
    if (!anyStamp || max === NEG_INF) return;
    let total = 0;
    let stampMass = 0;
    let textPeak = 0;
    for (let i = 0; i < vocab; i += 1) {
      const v = row[i]!;
      if (v === NEG_INF) continue;
      const p = Math.exp(v - max);
      total += p;
      if (i >= begin) stampMass += p;
      else if (p > textPeak) textPeak = p;
    }
    if (total > 0 && stampMass / total > textPeak / total) row.fill(NEG_INF, 0, begin);
  }
}
