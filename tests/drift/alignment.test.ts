import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import { describe, expect, it } from "vitest";

import { ctcVocabulary, logSoftmaxRows, normalizeForCtc, normalizeSignal, toTargets, viterbiAlign, wordsFromFrames } from "../../src/alignment/ctc.js";
import { dtwPath, groupIntoWords, medianFilter } from "../../src/alignment/dtw.js";
import { mergePunctuation, proportionalWords, splitWordsIntoSegments, type Word } from "../../src/alignment/punctuation.js";

const load = <T>(name: string) =>
  JSON.parse(readFileSync(join(dirname(fileURLToPath(import.meta.url)), "..", "fixtures", name), "utf-8")) as T;

const a = load<{
  viterbi: { logits: number[][]; targets: number[]; frames: number[] };
  dtw: { cost: number[][]; tokens: number[]; frames: number[] };
  median: { input: number[][]; output: number[][] };
  punctuation: { cases: Word[][]; merged: Word[][] };
  proportional: { segment: { start: number; end: number; text: string }; words: Word[] };
}>("alignment.json");
const c = load<{ vocab: Record<string, number>; cases: { text: string; normalized: string; targets: number[]; spans: [number, number][] }[] }>("ctc_text.json");

describe("alignment, against ynonnx_transcribe", () => {
  it("Viterbi picks the same frames", () => {
    const frames = a.viterbi.logits.length;
    const vocab = a.viterbi.logits[0]!.length;
    const emission = logSoftmaxRows(Float32Array.from(a.viterbi.logits.flat()), frames, vocab);
    expect(viterbiAlign(emission, frames, vocab, a.viterbi.targets, 0)).toEqual(a.viterbi.frames);
  });

  it("DTW walks the same path", () => {
    const { tokenPath, framePath } = dtwPath(a.dtw.cost.flat(), a.dtw.cost.length, a.dtw.cost[0]!.length);
    expect(tokenPath).toEqual(a.dtw.tokens);
    expect(framePath).toEqual(a.dtw.frames);
  });

  it("filters like numpy's median over edge-padded windows", () => {
    const rows = a.median.input.length;
    const cols = a.median.input[0]!.length;
    const out = medianFilter(Float32Array.from(a.median.input.flat()), rows, cols, 7);
    a.median.output.flat().forEach((v, i) => expect(out[i]).toBeCloseTo(v, 6));
  });

  it("glues punctuation to its word", () => {
    a.punctuation.cases.forEach((words, i) => expect(mergePunctuation(words)).toEqual(a.punctuation.merged[i]));
  });

  it("splits a segment proportionally to word length", () => {
    expect(proportionalWords(a.proportional.segment)).toEqual(a.proportional.words);
  });

  it("spells text for the CTC vocabulary the same way", () => {
    const vocabulary = ctcVocabulary(c.vocab);
    for (const k of c.cases) {
      expect(normalizeForCtc(k.text)).toBe(k.normalized);
      const { targets, spans } = toTargets(k.text, vocabulary);
      expect(targets).toEqual(k.targets);
      expect(spans).toEqual(k.spans);
    }
  });
});

describe("alignment helpers", () => {
  it("times the original words from the chosen frames", () => {
    const words = wordsFromFrames("Hi, you", [[0, 2], [3, 6]], [10, 12, 14, 20, 22, 30, 31], 1.5);
    expect(words).toEqual([
      { word: "Hi,", start: 1.7, end: 1.76, probability: 0 },
      { word: "you", start: 1.9, end: 2.12, probability: 0 },
    ]);
  });

  it("groups pieces into words and keeps them monotonic", () => {
    const words = groupIntoWords([" Hel", "lo", " world"], [[0, 0.2], [0.2, 0.4], [0.3, 0.8]]);
    expect(words).toEqual([
      { word: "Hello", start: 0, end: 0.4, probability: 0 },
      { word: "world", start: 0.4, end: 0.8, probability: 0 },
    ]);
  });

  it("gives each segment the words whose middle falls in it", () => {
    const segments: { end: number; words?: Word[] | null }[] = [{ end: 1 }, { end: 2 }];
    const w = (word: string, start: number, end: number) => ({ word, start, end, probability: 0 });
    splitWordsIntoSegments([w("a", 0, 0.5), w("b", 0.9, 1.2), w("c", 1.5, 1.9)], segments);
    expect(segments.map((s) => s.words?.map((x) => x.word))).toEqual([["a", "b"], ["c"]]);
  });

  it("normalizes the input as wav2vec2's feature extractor does", () => {
    const out = normalizeSignal(Float32Array.of(1, 2, 3, 4));
    expect(out.reduce((s, x) => s + x, 0)).toBeCloseTo(0, 6);
    expect(out.reduce((s, x) => s + x * x, 0) / 4).toBeCloseTo(1, 4);
  });
});
