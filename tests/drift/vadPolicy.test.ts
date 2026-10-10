import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import { describe, expect, it } from "vitest";

import { WHISPER_PARAMS_M, applyPadding, clipToSpeech, groupSpeech, restoreTime, shouldUseVad, speechCoverage } from "../../src/audio/vadPolicy.js";

const f = JSON.parse(
  readFileSync(join(dirname(fileURLToPath(import.meta.url)), "..", "fixtures", "vad_policy.json"), "utf-8"),
) as {
  segments: [number, number][];
  grouped: [number, number][];
  padded: [number, number][];
  clipped_length: number;
  clipped_first: [number, number];
  map: [number, number, number][];
  restored: [number, number][];
  coverage: number;
  should: Record<string, boolean>;
};

describe("the VAD policy agrees with ynonnx_transcribe.audio.vad_policy", () => {
  const segments = f.segments.map(([start, end]) => ({ start, end }));

  it("groups, pads and measures coverage", () => {
    expect(groupSpeech(segments)).toEqual(f.grouped);
    const padded = applyPadding(f.grouped, 200, 65);
    padded.forEach(([s, e], i) => {
      expect(s).toBeCloseTo(f.padded[i]![0], 9);
      expect(e).toBeCloseTo(f.padded[i]![1], 9);
    });
    expect(speechCoverage(segments, 65)).toBeCloseTo(f.coverage, 9);
  });

  it("clips to the speech and maps every time back", () => {
    const signal = Float32Array.from({ length: 65 * 16000 }, (_, i) => i / 16000);
    const { clipped, map } = clipToSpeech(signal, f.padded);
    expect(clipped.length).toBe(f.clipped_length);
    expect(clipped[0]).toBeCloseTo(f.clipped_first[0], 4);
    expect(clipped[clipped.length - 1]).toBeCloseTo(f.clipped_first[1], 4);
    map.forEach((m, i) => m.forEach((v, j) => expect(v).toBeCloseTo(f.map[i]![j]!, 9)));
    for (const [t, want] of f.restored) expect(restoreTime(t, map)).toBeCloseTo(want, 6);
  });

  it("is on for large models only, in auto", () => {
    for (const [name, want] of Object.entries(f.should)) expect(shouldUseVad(WHISPER_PARAMS_M[name] ?? null)).toBe(want);
  });
});
