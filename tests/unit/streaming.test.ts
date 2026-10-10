import { describe, expect, it } from "vitest";

import { StreamSession, type StreamUpdate, type StreamingTranscriber } from "../../src/streaming.js";
import type { TranscribeOptions, TranscriptionResult } from "../../src/whisper/transcriber.js";

const RATE = 16000;

/**
 * A transcriber that reads its "words" from the audio: every full second holds one value k and
 * comes out as the segment "wk"; a trailing partial second comes out as "wk?" — provisional, until
 * its right context arrives.
 */
function fake(name: string, calls: { start: number; length: number }[] = []): StreamingTranscriber {
  return {
    isMultilingual: false,
    modelName: name,
    async detectLanguage() {
      return { language: "en" };
    },
    async transcribe(audio: Float32Array, options?: TranscribeOptions): Promise<TranscriptionResult> {
      calls.push({ start: audio[0] ?? -1, length: audio.length / RATE });
      const seconds = audio.length / RATE;
      const segments = [];
      for (let s = 0; s < Math.ceil(seconds - 1e-6); s += 1) {
        const value = audio[Math.min(audio.length - 1, s * RATE + 10)]!;
        const whole = s + 1 <= seconds + 1e-6;
        segments.push({ id: s, start: s, end: Math.min(s + 1, seconds), text: ` ${name === "large" ? "W" : "w"}${value}${whole ? "" : "?"}`, avgLogprob: -0.1 });
      }
      void options;
      return { segments, text: segments.map((x) => x.text).join(""), language: "en" } as unknown as TranscriptionResult;
    },
  };
}

const second = (k: number, length = 1) => new Float32Array(Math.round(length * RATE)).fill(k);

function collect(session: StreamSession): StreamUpdate[] {
  const updates: StreamUpdate[] = [];
  session.on((u) => updates.push(u));
  return updates;
}

/** Revisions only grow, and nothing comes after a final one. */
function checkProtocol(updates: readonly StreamUpdate[]): void {
  const last = new Map<string, StreamUpdate>();
  for (const u of updates) {
    const before = last.get(u.segmentId);
    if (before) {
      expect(before.final, `${u.segmentId} revised after final`).toBe(false);
      expect(u.revision).toBeGreaterThan(before.revision);
    } else expect(u.revision).toBe(0);
    last.set(u.segmentId, u);
  }
}

describe("streaming, simple", () => {
  it("revises a segment as its right context arrives, and finalizes everything at close", async () => {
    const session = new StreamSession(fake("one"), { mode: "simple", stepS: 0.5 });
    const updates = collect(session);
    session.feed(second(1));
    session.feed(second(2, 0.5));
    await session.flush();
    expect(session.segments.map((s) => s.text)).toEqual(["w1", "w2?"]);
    session.feed(second(2, 0.5));
    session.feed(second(3));
    session.feed(second(4));
    const result = await session.close();
    expect(result.text).toBe("w1 w2 w3 w4");
    expect(result.segments.every((s) => !s.text.endsWith("?"))).toBe(true);
    checkProtocol(updates);

    const w2 = updates.filter((u) => u.segment?.text.startsWith("w2"));
    expect(new Set(w2.map((u) => u.segmentId)).size).toBe(1);
    expect(w2.map((u) => u.segment!.text)).toContain("w2?");

    const finals = new Map(updates.filter((u) => u.final).map((u) => [u.segmentId, u]));
    expect([...finals.values()].filter((u) => u.segment).map((u) => u.segment!.text)).toEqual(["w1", "w2", "w3", "w4"]);
    expect(updates.filter((u) => !u.final).every((u) => u.producer === "single")).toBe(true);
    expect(session.plan.finalizationPolicy).toMatch(/close/);
  });

  it("refuses audio after close, and keeps chunk boundaries out of segment boundaries", async () => {
    const session = new StreamSession(fake("one"), { stepS: 10 });
    session.feed(second(7, 0.3));
    session.feed(second(7, 0.7));
    const result = await session.close();
    expect(result.segments.map((s) => [s.text, s.start, s.end])).toEqual([["w7", 0, 1]]);
    expect(() => session.feed(second(1))).toThrow(/closed/);
  });
});

describe("streaming, post", () => {
  it("replaces every provisional segment with the full pass at close", async () => {
    const session = new StreamSession(fake("one"), { mode: "post", stepS: 0.5 });
    const updates = collect(session);
    session.feed(second(1));
    session.feed(second(2));
    await session.flush();
    session.feed(second(3));
    const result = await session.close();
    checkProtocol(updates);
    const provisional = new Set(updates.filter((u) => u.producer === "single").map((u) => u.segmentId));

    for (const id of provisional) {
      const last = updates.filter((u) => u.segmentId === id).at(-1)!;
      expect(last).toMatchObject({ final: true, segment: null });
    }
    const finals = updates.filter((u) => u.producer === "post_final");
    expect(finals.every((u) => u.final && u.revision === 0)).toBe(true);
    expect(finals.map((u) => u.segment!.text)).toEqual(["w1", "w2", "w3"]);
    expect(result.final).not.toBeNull();
    expect(result.plan.audioPersistence).toBe(true);
  });
});

describe("streaming, double", () => {
  it("publishes fast segments from windows of at most 3 s, then the large model replaces them", async () => {
    const fastCalls: { start: number; length: number }[] = [];
    const session = new StreamSession(fake("large"), { mode: "double", stepS: 1 }, fake("fast", fastCalls));
    const updates = collect(session);
    for (let k = 1; k <= 12; k += 1) {
      session.feed(second(k));
      await session.flush();
    }
    expect(fastCalls.every((c) => c.length <= 3 + 1e-6)).toBe(true);
    expect(updates.some((u) => u.producer === "fast")).toBe(true);
    expect(updates.some((u) => u.producer === "large")).toBe(true);
    const result = await session.close();
    checkProtocol(updates);

    const texts = result.segments.map((s) => s.text.toLowerCase().replace("?", ""));
    expect(texts).toEqual(Array.from({ length: 12 }, (_, i) => `w${i + 1}`));
    expect(result.segments.some((s) => s.text.startsWith("W"))).toBe(true);
    expect(result.plan.modelRoles.fast?.maxChunkDurationS).toBe(3);
    expect(result.plan.modelRoles.large?.maxChunkDurationS).toBe(30);
  });
});

describe("cleanSegments", () => {
  it("keeps a short window's segments inside its audio, without loops or duplicates", async () => {
    const { cleanSegments } = await import("../../src/streaming.js");
    const segment = (start: number, end: number, text: string, compressionRatio = 1.2) => ({ id: 0, start, end, text, tokens: [], compressionRatio, avgLogprob: -0.2 });
    const out = cleanSegments(
      [
        segment(0, 2, " Hello there."),
        segment(2, 2, " Hello there."),
        segment(2, 9, " General Kenobi."),
        segment(3, 4, " Without a card. Without a card. Without a card. Without a card."),
        segment(9, 12, " past the end"),
      ],
      10,
      14,
    );
    expect(out.map((s) => [s.start, s.end, s.text])).toEqual([
      [10, 12, "Hello there."],
      [12, 14, "General Kenobi."],
    ]);
    expect(cleanSegments([segment(0, 3, " fine"), segment(3, 5, " a b c d e f g h i j", 3.1)], 0, 10).map((s) => s.text)).toEqual(["fine"]);
  });
});

describe("sections inside language portions", () => {
  it("cuts each portion like an audio of its own, windows stopping at the portion's end", async () => {
    const { SectionScheduler } = await import("../../src/whisper/schedule.js");
    const sections = new SectionScheduler().planSpans({ id: "a", duration: 600 }, [{ start: 0, end: 400 }, { start: 400, end: 600 }], 0, { budget: 6 });
    const first = sections.filter((s) => s.limit === 400);
    const second = sections.filter((s) => s.limit === 600);
    expect(first.length).toBe(4);
    expect(second.length).toBe(2);

    expect(first[0]!.end).toBeGreaterThan(first[1]!.start);
    expect(Math.max(...first.map((s) => s.end))).toBe(400);
    expect(second[0]!.start).toBe(400);
    expect(sections.map((s) => s.index)).toEqual([0, 1, 2, 3, 4, 5]);
  });
});
