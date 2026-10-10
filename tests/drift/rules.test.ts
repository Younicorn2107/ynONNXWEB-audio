/** Whisper's timestamp rules suppress exactly what Python's do. */

import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import { describe, expect, it } from "vitest";

import { TimestampRules } from "../../src/whisper/rules.js";

const fixture = JSON.parse(
  readFileSync(join(dirname(fileURLToPath(import.meta.url)), "..", "fixtures", "timestamp_rules.json"), "utf-8"),
) as {
  vocab: number;
  begin: number;
  eot: number;
  structural: number[];
  structural_without: number[];
  cases: { name: string; case: number; stamp_bias: number; without_timestamps: boolean; emitted: number[]; suppressed: [number, number][] }[];
};

const vocabulary = {
  timestampBegin: fixture.begin,
  eot: fixture.eot,
  suppressed: (options?: { allowTimestamps?: boolean }) =>
    options?.allowTimestamps === false ? fixture.structural_without : fixture.structural,
};

function logitsFor(index: number, bias: number): Float32Array {
  const out = new Float32Array(fixture.vocab);
  for (let i = 0; i < fixture.vocab; i += 1) {
    out[i] = Math.fround(3 * Math.sin(0.37 * i + index) + (i >= fixture.begin ? bias : 0));
  }
  return out;
}

function ranges(row: Float32Array): [number, number][] {
  const out: [number, number][] = [];
  let start: number | null = null;
  for (let i = 0; i <= row.length; i += 1) {
    const on = i < row.length && row[i] === Number.NEGATIVE_INFINITY;
    if (on && start === null) start = i;
    else if (!on && start !== null) {
      out.push([start, i]);
      start = null;
    }
  }
  return out;
}

describe("TimestampRules agrees with ynonnx_transcribe's TimestampLogitsProcessor", () => {
  it.each(
    fixture.cases.map((c) => [`${c.name}, bias ${c.stamp_bias}${c.without_timestamps ? ", no timestamps" : ""}`, c] as const),
  )("%s", (_label, c) => {
    const rules = new TimestampRules(vocabulary, { maxInitialTimestamp: 1.0, withoutTimestamps: c.without_timestamps });
    const data = logitsFor(c.case, c.stamp_bias);
    rules.process({ data, rows: 1, vocab: fixture.vocab }, [c.emitted]);
    expect(ranges(data)).toEqual(c.suppressed);
  });
});
