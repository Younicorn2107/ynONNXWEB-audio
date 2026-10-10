import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import { describe, expect, it } from "vitest";

import { DEFAULT_VAD, segmentsFromProbabilities, type VADConfig } from "../../src/audio/vad.js";

const fixture = JSON.parse(
  readFileSync(join(dirname(fileURLToPath(import.meta.url)), "..", "fixtures", "vad_spans.json"), "utf-8"),
) as { probabilities: number[]; samples: number; cases: { name: string; config: Record<string, number>; spans: [number, number][] }[] };

const NAMES: Record<string, keyof VADConfig> = {
  threshold: "threshold",
  min_speech_duration_ms: "minSpeechDurationMs",
  min_silence_duration_ms: "minSilenceDurationMs",
  speech_pad_ms: "speechPadMs",
  max_speech_duration_s: "maxSpeechDurationS",
};

describe("speech spans agree with ynonnx_core's VAD", () => {
  it.each(fixture.cases.map((c) => [c.name, c] as const))("%s", (_name, c) => {
    const config = { ...DEFAULT_VAD } as VADConfig;
    for (const [key, value] of Object.entries(c.config)) (config as unknown as Record<string, number>)[NAMES[key]!] = value;
    const spans = segmentsFromProbabilities(Float32Array.from(fixture.probabilities), fixture.samples, config);
    expect(spans.length).toBe(c.spans.length);
    spans.forEach((span, i) => {
      expect(span.start).toBeCloseTo(c.spans[i]![0], 6);
      expect(span.end).toBeCloseTo(c.spans[i]![1], 6);
    });
  });
});
