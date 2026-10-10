/** The pure half of the pipeline: segments, scheduling, quality, tokenizer. */

import { describe, expect, it, vi } from "vitest";

import { SchedulingError } from "../../src/errors.js";
import { Tokenizer, TokenizerError } from "@youni/onnxweb-core/core";
import { WhisperTokens } from "../../src/whisper/tokens.js";
import { DEFAULT_MAX_CONTEXT, buildPrompt, withoutContext } from "../../src/whisper/prompt.js";
import { MAX_TARGET_POSITIONS } from "../../src/whisper/transcriber.js";
import {
  DEFAULT_QUALITY,
  TEMPERATURE_SCALE,
  avgLogprob,
  avgLogprobFromSum,
  compressionRatioSync,
  isDegenerate,
  isSilent,
  withTemperatureFallback,
} from "../../src/whisper/quality.js";
import { extractSegments, mergeSegments } from "../../src/whisper/segments.js";
import {
  MIN_SECTION_S,
  SectionScheduler,
  WaveScheduler,
  type WindowResult,
} from "../../src/whisper/schedule.js";

function makeTokenizer(): Tokenizer {
  const vocab: Record<string, number> = { hello: 0, "Ġworld": 1, "Ġagain": 2, a: 3 };
  const added = [
    { id: 100, content: "<|endoftext|>", special: true },
    { id: 101, content: "<|startoftranscript|>", special: true },
    { id: 102, content: "<|startofprev|>", special: true },
    { id: 103, content: "<|notimestamps|>", special: true },
    { id: 104, content: "<|transcribe|>", special: true },
    { id: 105, content: "<|translate|>", special: true },
    { id: 106, content: "<|nospeech|>", special: true },
    { id: 107, content: "<|en|>", special: true },
    { id: 108, content: "<|fr|>", special: true },
  ];

  for (let step = 0; step < 60; step += 1) {
    added.push({
      id: 200 + step,
      content: `<|${(step * 0.02).toFixed(2)}|>`,
      special: true,
    });
  }
  return Tokenizer.fromJson({
    model: { type: "BPE", vocab, merges: [] },
    added_tokens: added.map((t) => ({ ...t, single_word: false, lstrip: false, rstrip: false, normalized: false })),
    pre_tokenizer: { type: "ByteLevel", add_prefix_space: false, trim_offsets: true, use_regex: true },
    decoder: { type: "ByteLevel", add_prefix_space: false, trim_offsets: true, use_regex: true },
    normalizer: null,
    post_processor: null,
  });
}

const tokenizer = makeTokenizer();

const tokens = new WhisperTokens(tokenizer, { multilingual: true });

describe("the tokenizer", () => {
  it("decodes byte-level BPE back to text", () => {
    expect(tokenizer.decode([0, 1])).toBe("hello world");
  });

  it("drops control tokens from the transcript", () => {
    expect(tokenizer.decode([101, 107, 104, 0, 1, 100])).toBe("hello world");
  });

  it("can be asked to keep them", () => {
    expect(tokenizer.decode([101, 0], { skipSpecialTokens: false })).toContain("<|startoftranscript|>");
  });

  it("refuses a file that is not a tokenizer rather than decoding to nothing", () => {
    expect(() => Tokenizer.fromJson("{not json")).toThrow(TokenizerError);
  });

  it("knows an English-only checkpoint by its vocabulary size, not by its language tokens", () => {
    expect(new WhisperTokens(tokenizer, { vocabSize: 51864 }).isMultilingual).toBe(false);
    expect(new WhisperTokens(tokenizer, { vocabSize: 51864 }).languages.size).toBe(0);
    expect(new WhisperTokens(tokenizer, { vocabSize: 51865 }).isMultilingual).toBe(true);
  });
});

describe("the special tokens", () => {
  it("resolves them from the tokenizer, not from hardcoded ids", () => {
    expect(tokens.sot).toBe(101);
    expect(tokens.eot).toBe(100);
    expect(tokens.noTimestamps).toBe(103);
  });

  it("finds the language tokens", () => {
    expect(tokens.isMultilingual).toBe(true);
    expect(tokens.language("fr")).toBe(108);
    expect(tokens.language("zz")).toBeNull();
  });

  it("maps timestamps to seconds on the 20 ms grid", () => {
    expect(tokens.timeOf(200)).toBeCloseTo(0);
    expect(tokens.timeOf(210)).toBeCloseTo(0.2);
    expect(tokens.timeOf(0)).toBeNull();
  });

  it("suppresses <|notimestamps|> by default", () => {
    expect(tokens.suppressed()).toContain(103);
    expect(tokens.suppressed({ allowTimestamps: false })).not.toContain(103);
  });
});

describe("the prompt", () => {
  it("opens with the transcript marker", () => {
    expect(buildPrompt(tokens)[0]).toBe(tokens.sot);
  });

  it("carries the language and the task on a multilingual model", () => {
    expect(buildPrompt(tokens, { language: "fr" })).toEqual([101, 108, 104]);
  });

  it("prepends previous tokens behind <|startofprev|>", () => {
    const context = Array.from({ length: 32 }, (_, i) => i % 4);
    const prompt = buildPrompt(tokens, { context, language: "en" });
    expect(prompt).toEqual([102, ...context, 101, 107, 104]);
  });

  it("rounds the context down onto the ladder, keeping its end", () => {
    const context = Array.from({ length: 40 }, (_, i) => i);
    const prompt = buildPrompt(tokens, { context, language: "en" });
    expect(prompt.slice(1, -3)).toEqual(context.slice(-32));
  });

  it("carries no context shorter than the lowest step", () => {
    expect(buildPrompt(tokens, { context: [0, 1], language: "en" })).toEqual([101, 107, 104]);
  });

  it("drops the context entirely when the budget is zero", () => {
    expect(
      buildPrompt(tokens, { language: "fr", context: [0, 1], maxContext: 0 }),
    ).toEqual([tokens.sot, tokens.languages.get("fr"), tokens.transcribe]);
  });

  it("strips a previous window's context so it does not compound", () => {
    const prompt = buildPrompt(tokens, { context: [0, 1] });
    expect(withoutContext(tokens, prompt)[0]).toBe(tokens.sot);
  });
});

describe("segments", () => {
  const ts = (seconds: number): number => 200 + Math.round(seconds / 0.02);

  it("reads adjacent timestamp pairs", () => {
    const ids = [ts(0), 0, ts(0.2), ts(0.2), 1, ts(0.4)];
    const parsed = extractSegments(ids, tokens, tokenizer);
    expect(parsed.segments).toHaveLength(2);
    expect(parsed.segments[0]).toMatchObject({ start: 0, end: 0.2, text: "hello" });
    expect(parsed.segments[1]?.text).toBe("world");
  });

  it("offsets every time by the window start", () => {
    const parsed = extractSegments([ts(0), 0, ts(0.2)], tokens, tokenizer, 30);
    expect(parsed.segments[0]?.start).toBeCloseTo(30);
    expect(parsed.segments[0]?.end).toBeCloseTo(30.2);
  });

  it("reports the last CLOSED end, which is where the seek lands", () => {
    const parsed = extractSegments([ts(0), 0, ts(0.4)], tokens, tokenizer);
    expect(parsed.lastEnd).toBeCloseTo(0.4);
  });

  it("keeps a trailing segment the model never closed", () => {
    const parsed = extractSegments([ts(0), 0, 1], tokens, tokenizer);
    expect(parsed.segments).toHaveLength(1);
    expect(parsed.lastEnd).toBeNull();
  });

  it("keeps the text when the model emitted no timestamps", () => {
    const parsed = extractSegments([0, 1], tokens, tokenizer);
    expect(parsed.segments).toHaveLength(1);
    expect(parsed.text).toBe("hello world");
    expect(parsed.lastEnd).toBeNull();
  });

  it("drops a duplicate produced by the window overlap", () => {
    const merged = mergeSegments([
      [{ start: 0, end: 2, text: "hello world", tokens: [] }],
      [{ start: 1.5, end: 2, text: "hello world", tokens: [] }],
    ]);
    expect(merged).toHaveLength(1);
  });

  it("keeps genuinely different speech that happens to overlap", () => {
    const merged = mergeSegments([
      [{ start: 0, end: 2, text: "hello", tokens: [] }],
      [{ start: 1.5, end: 3, text: "something else", tokens: [] }],
    ]);
    expect(merged).toHaveLength(2);
  });
});

describe("quality", () => {
  it("divides by n+1, the faster-whisper formula", () => {
    expect(avgLogprob([-1, -1, -1])).toBeCloseTo(-0.75);
    expect(avgLogprobFromSum(-3, 3)).toBeCloseTo(-0.75);
    expect(avgLogprob([])).toBe(0);
  });

  it("scores a repetition loop as far more compressible than prose", () => {
    const loop = "I I I I I I I I I I I I I I I I I I I I I I I I I I I I";
    const prose =
      "Our planet is covered in oceans, and yet we have chosen to call it the Earth.";
    expect(compressionRatioSync(loop)).toBeGreaterThan(DEFAULT_QUALITY.compressionRatioThreshold);
    expect(compressionRatioSync(prose)).toBeLessThan(DEFAULT_QUALITY.compressionRatioThreshold);
  });

  it("calls a window degenerate when it is too compressible OR too improbable", () => {
    expect(isDegenerate({ text: "", avgLogprob: -0.1, compressionRatio: 3, noSpeechProb: 0 })).toBe(true);
    expect(isDegenerate({ text: "", avgLogprob: -2, compressionRatio: 1, noSpeechProb: 0 })).toBe(true);
    expect(isDegenerate({ text: "", avgLogprob: -0.1, compressionRatio: 1, noSpeechProb: 0 })).toBe(false);
  });

  it("needs BOTH conditions to call a window silent", () => {
    expect(isSilent(0.9, -2)).toBe(true);
    expect(isSilent(0.9, -0.1)).toBe(false);
    expect(isSilent(0.1, -2)).toBe(false);
  });
});

describe("the temperature ladder", () => {
  it("stops at the first rung that is not degenerate", async () => {
    const attempt = vi.fn(async (temperature: number) => ({
      text: "x",
      avgLogprob: temperature > 0 ? -0.1 : -5,
      compressionRatio: 1,
      noSpeechProb: 0,
    }));
    const { temperature, attempts } = await withTemperatureFallback(attempt);
    expect(temperature).toBe(TEMPERATURE_SCALE[1]);
    expect(attempts).toBe(2);
  });

  it("returns the last attempt rather than failing when every rung is bad", async () => {
    const { result, attempts } = await withTemperatureFallback(async () => ({
      text: "bad",
      avgLogprob: -9,
      compressionRatio: 9,
      noSpeechProb: 0,
    }));
    expect(result.text).toBe("bad");
    expect(attempts).toBe(TEMPERATURE_SCALE.length);
  });

  it("honours a pinned temperature", async () => {
    const attempt = vi.fn(async () => ({
      text: "x",
      avgLogprob: -9,
      compressionRatio: 9,
      noSpeechProb: 0,
    }));
    await withTemperatureFallback(attempt, { scale: [0.4] });
    expect(attempt).toHaveBeenCalledTimes(1);
  });
});

describe("sectioning", () => {
  const scheduler = new SectionScheduler();

  it("never cuts below 90 seconds, whatever the batch asks for", () => {
    const sections = scheduler.plan([{ id: "a", duration: 60 }], { totalBatch: 8 });
    expect(sections).toHaveLength(1);
  });

  it("cuts a long file into at most duration/90 sections", () => {
    const sections = scheduler.plan([{ id: "a", duration: 600 }], { totalBatch: 20 });
    expect(sections.length).toBeLessThanOrEqual(Math.trunc(600 / MIN_SECTION_S));
  });

  it("overlaps sections so the merger can cut cleanly", () => {
    const sections = scheduler.plan([{ id: "a", duration: 600 }], { totalBatch: 3 });
    if (sections.length > 1) {
      expect(sections[0]!.end).toBeGreaterThan(sections[1]!.start);
    }
  });

  it("shares the budget by duration, not equally", () => {
    const sections = scheduler.plan(
      [
        { id: "long", duration: 1800 },
        { id: "short", duration: 95 },
      ],
      { totalBatch: 10 },
    );
    const long = sections.filter((s) => s.audioId === "long").length;
    const short = sections.filter((s) => s.audioId === "short").length;
    expect(long).toBeGreaterThan(short);
  });
});

describe("waves", () => {
  const result = (start: number, end: number | null): WindowResult => ({
    audioId: "a",
    sectionIndex: 0,
    windowStart: start,
    tokens: [1],
    text: "x",
    avgLogprob: -0.2,
    compressionRatio: 1.2,
    noSpeechProb: 0.01,
    isSilent: false,
    lastSegmentEnd: end,
    temperature: 0,
  });

  it("terminates when the caller advances", () => {
    const waves = new WaveScheduler();
    const states = waves.states(
      new SectionScheduler().plan([{ id: "a", duration: 120 }]),
    );
    let rounds = 0;
    for (const wave of waves.waves(states, 4)) {
      rounds += 1;
      expect(rounds).toBeLessThan(50);
      for (const state of wave) waves.advance(state, result(state.seek, null));
    }
    expect(states.every((s) => s.done)).toBe(true);
  });

  it("terminates when the caller skips", () => {
    const waves = new WaveScheduler();
    const states = waves.states(new SectionScheduler().plan([{ id: "a", duration: 120 }]));
    let rounds = 0;
    for (const wave of waves.waves(states, 4)) {
      rounds += 1;
      expect(rounds).toBeLessThan(50);
      for (const state of wave) waves.skip(state);
    }
    expect(states.every((s) => s.done)).toBe(true);
  });

  it("refuses to loop for ever when the caller does neither", () => {
    const waves = new WaveScheduler();
    const states = waves.states(new SectionScheduler().plan([{ id: "a", duration: 120 }]));
    const stream = waves.waves(states, 4);
    expect(() => {
      for (let i = 0; i < 50; i += 1) stream.next();
    }).toThrow(SchedulingError);
  });

  it("lands the seek on the last closed segment, not mid-word", () => {
    const waves = new WaveScheduler();
    const [state] = waves.states(new SectionScheduler().plan([{ id: "a", duration: 120 }]));
    waves.advance(state!, result(0, 25));
    expect(state!.seek).toBeCloseTo(25);
  });

  it("advances a full window when nothing closed", () => {
    const waves = new WaveScheduler();
    const [state] = waves.states(new SectionScheduler().plan([{ id: "a", duration: 120 }]));
    waves.advance(state!, result(0, null));
    expect(state!.seek).toBeCloseTo(30);
  });

  it("ignores a closing stamp too early to be progress", () => {
    const waves = new WaveScheduler();
    const [state] = waves.states(new SectionScheduler().plan([{ id: "a", duration: 120 }]));
    waves.advance(state!, result(0, 3));
    expect(state!.seek).toBeCloseTo(30);
  });

  it("carries the tokens forward, but not from a silent window", () => {
    const waves = new WaveScheduler();
    const [state] = waves.states(new SectionScheduler().plan([{ id: "a", duration: 120 }]));
    waves.advance(state!, { ...result(0, null), isSilent: true });
    expect(state!.context).toEqual([]);
    waves.advance(state!, result(30, null));
    expect(state!.context).toEqual([1]);
  });
});

describe("the 448 positions", () => {
  it("is the total, prompt included", () => {
    expect(MAX_TARGET_POSITIONS).toBe(448);
  });

  it("leaves the same budget the Python engine computes", () => {
    const budget = (prompt: number) => Math.max(1, MAX_TARGET_POSITIONS - prompt);
    expect(budget(4)).toBe(444);
    expect(budget(DEFAULT_MAX_CONTEXT + 4)).toBe(220);
    expect(budget(500)).toBe(1);
  });
});

describe("markers in the transcript", () => {
  it("drops generated language and task tokens", () => {
    const tokenizer = makeTokenizer();
    const tokens = new WhisperTokens(tokenizer, { multilingual: true });
    const language = tokens.languages.get("fr");
    expect(language).toBeDefined();

    const { text } = extractSegments(
      [language!, tokens.noTimestamps!, 200, 0, 1],
      tokens,
      tokenizer,
    );
    expect(text).toBe("hello world");
  });
});

describe("an unpinned language", () => {
  it("leaves the slot empty rather than filling it with the task", () => {
    const built = buildPrompt(tokens, { language: null });
    expect(built).toEqual([tokens.sot]);
    expect(built).not.toContain(tokens.transcribe);
  });

  it("still emits both once a language is known", () => {
    expect(buildPrompt(tokens, { language: "fr" })).toEqual([
      tokens.sot,
      tokens.languages.get("fr"),
      tokens.transcribe,
    ]);
  });

  it("leaves the slot empty for a language the model does not have", () => {
    expect(buildPrompt(tokens, { language: "zz" })).toEqual([tokens.sot]);
  });
});
