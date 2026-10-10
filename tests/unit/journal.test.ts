import { describe, expect, it } from "vitest";

import { MemoryKeyValueStore } from "@youni/onnxweb-core/core";

import { GAP, WaveJournal } from "../../src/whisper/journal.js";
import { SectionScheduler, WaveScheduler, type WindowResult } from "../../src/whisper/schedule.js";

const window = (audioId: string, sectionIndex: number, windowStart: number, text: string): WindowResult => ({
  audioId,
  sectionIndex,
  windowStart,
  tokens: [1, 2, 3],
  text,
  avgLogprob: -0.2,
  compressionRatio: 1.2,
  noSpeechProb: 0.01,
  isSilent: false,
  lastSegmentEnd: windowStart + 28,
  temperature: 0,
});

function plan() {
  const sections = new SectionScheduler().plan([{ id: "a", duration: 200 }], { totalBatch: 2 });
  return new WaveScheduler(30).states(sections);
}

describe("the wave journal", () => {
  it("resumes each section where the last run left it", async () => {
    const store = new MemoryKeyValueStore();
    const first = plan();
    const journal = new WaveJournal(store, "transcribe/x");
    first[0]!.results.push(window("a", 0, 0, " One."));
    first[0]!.seek = 28;
    first[0]!.context = [7, 8];
    await journal.save(first);
    first[0]!.results.push(window("a", 0, 28, " Two."));
    first[0]!.seek = 56;
    await journal.save(first);

    await journal.save(first);
    expect((await store.keys("transcribe/x/entry/")).length).toBe(2);

    const second = plan();
    const resumed = await new WaveJournal(store, "transcribe/x").restore(second);
    expect(resumed).toBe(1);
    expect(second[0]!.seek).toBe(56);
    expect(second[0]!.context).toEqual([7, 8]);
    expect(second[0]!.results.map((r) => r.text)).toEqual([" One.", " Two."]);
    expect(second[1]!.seek).toBe(second[1]!.section.start);
  });

  it("never resumes another fingerprint", async () => {
    const store = new MemoryKeyValueStore();
    const states = plan();
    states[0]!.results.push(window("a", 0, 0, " One."));
    states[0]!.seek = 28;
    await new WaveJournal(store, "transcribe/x").save(states);
    expect(await new WaveJournal(store, "transcribe/y").restore(plan())).toBe(0);
  });

  it("writes the text so far, gaps marked, and goes away at the end", async () => {
    const store = new MemoryKeyValueStore();
    const states = plan();
    states[0]!.results.push(window("a", 0, 0, " One."));
    states[0]!.seek = 28;
    const journal = new WaveJournal(store, "transcribe/x");
    await journal.save(states);
    const text = await journal.text();
    expect(text).toContain(`[a 00:00:00–`);
    expect(text).toContain(`One. ${GAP}`);
    await journal.complete();
    expect(await store.keys("transcribe/x/")).toEqual([]);
  });
});
