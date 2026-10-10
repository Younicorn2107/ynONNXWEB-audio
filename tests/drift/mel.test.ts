/** The log-mel front end, against numpy. */

import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import { describe, expect, it } from "vitest";

import {
  RealDFT,
  WHISPER_MEL,
  WHISPER_MEL_V3,
  logMel,
  melBandsFor,
  melConfigFor,
  melFilters,
} from "../../src/audio/mel.js";

const path = join(dirname(fileURLToPath(import.meta.url)), "..", "fixtures", "mel.json");

interface MelFixture {
  sample_rate: number;
  audio_f32_le_base64: string;
  sample_count: number;
  cases: { n_mels: number; frames: number; mel_f32_le_base64: string }[];
  tolerance: number;
}

function decodeFloat32(base64: string): Float32Array {
  const binary = Buffer.from(base64, "base64");
  return new Float32Array(
    binary.buffer.slice(binary.byteOffset, binary.byteOffset + binary.byteLength),
  );
}

describe("logMel reproduces what numpy computed", () => {
  const fixture = JSON.parse(readFileSync(path, "utf-8")) as MelFixture;

  it.each(fixture.cases.map((c) => [c.n_mels, c] as const))("agrees coefficient by coefficient, %i bands", (_bands, melCase) => {
    const audio = decodeFloat32(fixture.audio_f32_le_base64);
    const expected = decodeFloat32(melCase.mel_f32_le_base64);
    expect(audio.length).toBe(fixture.sample_count);

    const mel = logMel(audio, melCase.n_mels === 128 ? WHISPER_MEL_V3 : WHISPER_MEL, null);

    expect(mel.nMels).toBe(melCase.n_mels);
    expect(mel.frames).toBe(melCase.frames);
    expect(mel.data.length).toBe(expected.length);
    expect(expected.length).toBeGreaterThan(0);

    let worst = 0;
    let worstAt = -1;
    for (let i = 0; i < expected.length; i += 1) {
      const delta = Math.abs(mel.data[i]! - expected[i]!);
      if (delta > worst) {
        worst = delta;
        worstAt = i;
      }
    }
    const band = Math.floor(worstAt / mel.frames);
    const frame = worstAt % mel.frames;
    expect(
      worst,
      `worst mismatch ${worst.toExponential(2)} at mel band ${band}, frame ${frame}`,
    ).toBeLessThan(fixture.tolerance);
  });

  it("puts the chirp's energy where the chirp is", () => {
    const audio = decodeFloat32(fixture.audio_f32_le_base64);
    const mel = logMel(audio, WHISPER_MEL, null);

    const energyAt = (frame: number): number => {
      let sum = 0;
      for (let band = 0; band < mel.nMels; band += 1) sum += mel.data[band * mel.frames + frame]!;
      return sum / mel.nMels;
    };
    expect(energyAt(10)).toBeGreaterThan(energyAt(mel.frames - 2));
  });
});

describe("the filterbank", () => {
  it("is Slaney-normalised, not HTK", () => {
    const filters = melFilters(WHISPER_MEL);
    const nFreqs = WHISPER_MEL.nFft / 2 + 1;
    expect(filters.length).toBe(WHISPER_MEL.nMels * nFreqs);

    for (let m = 0; m < WHISPER_MEL.nMels; m += 1) {
      let sum = 0;
      for (let f = 0; f < nFreqs; f += 1) {
        const value = filters[m * nFreqs + f]!;
        expect(value).toBeGreaterThanOrEqual(0);
        sum += value;
      }
      expect(sum, `mel band ${m} is empty`).toBeGreaterThan(0);
    }
  });

  it("is memoised, since it is rebuilt per window otherwise", () => {
    expect(melFilters(WHISPER_MEL)).toBe(melFilters(WHISPER_MEL));
  });
});

describe("windowing", () => {
  it("pads the audio to exactly 3000 frames", () => {
    const mel = logMel(new Float32Array(16000));
    expect(mel.frames).toBe(3000);
  });

  it("handles an empty signal without throwing", () => {
    expect(logMel(new Float32Array(0)).frames).toBe(3000);
  });

  it("trims audio longer than the window", () => {
    const mel = logMel(new Float32Array(16000 * 40));
    expect(mel.frames).toBe(3000);
  });
});

describe("how many mel bands", () => {
  it("believes a static encoder axis", () => {
    expect(melBandsFor({ encoderDims: ["batch_size", 128, 3000] })).toBe(128);
    expect(melBandsFor({ encoderDims: ["batch_size", 80, 3000] })).toBe(80);
  });

  it("prefers the graph over the vocabulary", () => {
    expect(
      melBandsFor({ encoderDims: ["batch_size", 128, 3000], logitsVocabSize: 51865 }),
    ).toBe(128);
  });

  it("falls back to the vocabulary when the axis is dynamic", () => {
    const dynamic = ["batch_size", "feature_size", "encoder_sequence_length"] as const;
    expect(melBandsFor({ encoderDims: dynamic, logitsVocabSize: 51866 })).toBe(128);
    expect(melBandsFor({ encoderDims: dynamic, logitsVocabSize: 51865 })).toBe(80);
  });

  it("does not treat a bigger vocabulary as a newer model", () => {
    const dynamic = ["batch_size", "feature_size", "encoder_sequence_length"] as const;
    expect(melBandsFor({ encoderDims: dynamic, logitsVocabSize: 51896 })).toBe(80);
    expect(melBandsFor({ encoderDims: dynamic, logitsVocabSize: 51897 })).toBe(80);

    expect(
      melBandsFor({ encoderDims: ["batch_size", 128, 3000], logitsVocabSize: 51897 }),
    ).toBe(128);
  });

  it("assumes 80 when nothing is known", () => {
    expect(melBandsFor({})).toBe(80);
    expect(melBandsFor({ encoderDims: [], logitsVocabSize: null })).toBe(80);
  });

  it("builds a config that differs only in the band count", () => {
    expect(melConfigFor({ logitsVocabSize: 51866 })).toEqual(WHISPER_MEL_V3);
    expect(melConfigFor({ logitsVocabSize: 51865 })).toEqual(WHISPER_MEL);
    expect(WHISPER_MEL_V3.nFft).toBe(WHISPER_MEL.nFft);
    expect(WHISPER_MEL_V3.hopLength).toBe(WHISPER_MEL.hopLength);
  });

  it("actually produces 128 rows for a v3 model", () => {
    const mel = logMel(new Float32Array(16000), WHISPER_MEL_V3);
    expect(mel.nMels).toBe(128);
    expect(mel.data.length).toBe(128 * mel.frames);
  });

  it("keeps a separate filterbank per band count", () => {
    expect(melFilters(WHISPER_MEL).length).toBe(80 * (WHISPER_MEL.nFft / 2 + 1));
    expect(melFilters(WHISPER_MEL_V3).length).toBe(128 * (WHISPER_MEL.nFft / 2 + 1));
  });
});

describe("RealDFT", () => {
  it("transforms two frames at once exactly as one at a time", () => {
    const dft = new RealDFT(400);
    const a = Float64Array.from({ length: 400 }, (_, i) => Math.sin(i * 0.37) + 0.2 * Math.cos(i * 1.9));
    const b = Float64Array.from({ length: 400 }, (_, i) => ((i * 7919) % 101) / 101 - 0.5);
    const one = (x: Float64Array) => {
      const re = new Float64Array(201);
      const im = new Float64Array(201);
      dft.run(x, re, im);
      return [re, im];
    };
    const [aRe, aIm] = one(a);
    const [bRe, bIm] = one(b);
    const outA: [Float64Array, Float64Array] = [new Float64Array(201), new Float64Array(201)];
    const outB: [Float64Array, Float64Array] = [new Float64Array(201), new Float64Array(201)];
    dft.runPair(a, b, outA, outB);
    for (let k = 0; k < 201; k += 1) {
      expect(outA[0][k]).toBeCloseTo(aRe![k]!, 9);
      expect(outA[1][k]).toBeCloseTo(aIm![k]!, 9);
      expect(outB[0][k]).toBeCloseTo(bRe![k]!, 9);
      expect(outB[1][k]).toBeCloseTo(bIm![k]!, 9);
    }
  });

  it("matches a direct DFT", () => {
    const dft = new RealDFT(400);
    const x = Float64Array.from({ length: 400 }, (_, i) => Math.cos(i * 0.11) * Math.exp(-i / 300));
    const re = new Float64Array(201);
    const im = new Float64Array(201);
    dft.run(x, re, im);
    for (const k of [0, 1, 17, 100, 200]) {
      let r = 0;
      let s = 0;
      for (let j = 0; j < 400; j += 1) {
        r += x[j]! * Math.cos((-2 * Math.PI * j * k) / 400);
        s += x[j]! * Math.sin((-2 * Math.PI * j * k) / 400);
      }
      expect(re[k]).toBeCloseTo(r, 8);
      expect(im[k]).toBeCloseTo(s, 8);
    }
  });
});
