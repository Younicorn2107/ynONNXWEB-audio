import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import { describe, expect, it } from "vitest";

import { agglomerative, relabelByFirstAppearance, spectral, symmetricEigen } from "../../src/diarization/clustering.js";
import { assignSpeakers, chunkSpeech, runsToSpans, speakerDurations, toTurns } from "../../src/diarization/diarizer.js";
import { fbank } from "../../src/diarization/fbank.js";

const f = JSON.parse(
  readFileSync(join(dirname(fileURLToPath(import.meta.url)), "..", "fixtures", "diarization.json"), "utf-8"),
) as {
  fbank: { frames: number; first: number[]; last: number[]; mean_abs: number };
  embeddings: number[][];
  agglomerative: { kwargs: { threshold?: number; num_speakers?: number; max_speakers?: number; min_speakers?: number }; labels: number[] }[];
  spectral: number[];
  chunks: [number, number][];
  runs: [number, number][];
  turns: { speaker: string; start: number; end: number }[];
  assign_longest: (string | null)[];
  assign_first: (string | null)[];
  durations: Record<string, number>;
  segments: { start: number; end: number }[];
};

/** The generator's "bright" signal (scripts/gen_fixtures.py, `profile_signal`), one second. */
function bright(seconds = 1): Float32Array {
  const n = seconds * 16000;
  const out = new Float32Array(n);
  let state = 12345;
  for (let i = 0; i < n; i += 1) {
    state = Number((1103515245n * BigInt(state) + 12345n) % 2147483648n);
    const noise = state / 2147483648 - 0.5;
    const t = i / 16000;
    const envelope = 0.5 + 0.5 * Math.sin(2 * Math.PI * 3 * t);
    const voice = 0.3 * Math.sin(2 * Math.PI * 180 * t) + 0.2 * Math.sin(2 * Math.PI * 1200 * t) + 0.1 * Math.sin(2 * Math.PI * 5000 * t);
    out[i] = voice * envelope + 0.01 * noise;
  }
  return out;
}

const embeddings = f.embeddings.map((row) => Float32Array.from(row));

describe("diarization, against ynonnx_transcribe", () => {
  it("computes Kaldi fbank as Python does", () => {
    const out = fbank(bright());
    expect(out.frames).toBe(f.fbank.frames);
    f.fbank.first.forEach((v, b) => expect(out.data[b]).toBeCloseTo(v, 3));
    f.fbank.last.forEach((v, b) => expect(out.data[(out.frames - 1) * 80 + b]).toBeCloseTo(v, 3));
  });

  it("clusters agglomeratively into the same speakers", () => {
    for (const c of f.agglomerative) {
      const labels = agglomerative(embeddings, {
        ...(c.kwargs.threshold !== undefined ? { threshold: c.kwargs.threshold } : {}),
        numSpeakers: c.kwargs.num_speakers ?? null,
        minSpeakers: c.kwargs.min_speakers ?? null,
        maxSpeakers: c.kwargs.max_speakers ?? null,
      });
      expect(relabelByFirstAppearance(labels), JSON.stringify(c.kwargs)).toEqual(c.labels);
    }
  });

  it("finds the same speakers spectrally (count by eigengap)", () => {
    expect(relabelByFirstAppearance(spectral(embeddings))).toEqual(f.spectral);
  });

  it("chunks, runs, turns, assignment and durations", () => {
    expect(chunkSpeech([[0, 4.1], [5, 5.6], [7, 7.5]])).toEqual(f.chunks);
    expect(runsToSpans([0, 0, 1, 1, 1, 2, 0, 0], 0.0169)).toEqual(f.runs);
    const turns = toTurns([[0, 1.5], [0.75, 2.25], [1.5, 3], [3.2, 4], [4.6, 6], [6.2, 7]], [0, 0, 1, 1, 0, 0]);
    expect(turns).toEqual(f.turns);
    const segments = f.segments.map((s) => ({ ...s, speaker: null as string | null }));
    expect(assignSpeakers(segments, turns).map((s) => s.speaker)).toEqual(f.assign_longest);
    expect(assignSpeakers(segments, turns, "first").map((s) => s.speaker)).toEqual(f.assign_first);
    expect(speakerDurations(turns)).toEqual(f.durations);
  });
});

describe("eigen decomposition", () => {
  it("diagonalizes a symmetric matrix", () => {
    const m = [Float64Array.of(2, 1, 0), Float64Array.of(1, 2, 1), Float64Array.of(0, 1, 2)];
    const { values, vectors } = symmetricEigen(m);
    expect(values[0]).toBeCloseTo(2 - Math.SQRT2, 9);
    expect(values[1]).toBeCloseTo(2, 9);
    expect(values[2]).toBeCloseTo(2 + Math.SQRT2, 9);

    const v = vectors.map((row) => row[0]!);
    m.forEach((row, i) => expect(row.reduce((s, x, j) => s + x * v[j]!, 0)).toBeCloseTo(values[0]! * v[i]!, 9));
  });
});
