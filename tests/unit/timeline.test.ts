import { describe, expect, it } from "vitest";

import { renderCsv, renderMarkdown, renderResult } from "../../src/render.js";
import { Timeline, checkAnalyses } from "../../src/timeline.js";

describe("the timeline", () => {
  it("sorts by (start, end, kind, id), numbers ids per kind, and serializes with its schema version", () => {
    const t = new Timeline();
    t.add({ kind: "word", start: 4, end: 6, label: "b", source: "m", granularity: "word" });
    t.add({ kind: "speaker", start: 0, end: 5, label: "SPEAKER_00", speakerId: "SPEAKER_00", source: "w", granularity: "segment" });
    t.add({ kind: "word", start: 1, end: 3, label: "a", source: "m", granularity: "word" });
    t.add({ kind: "language", start: 0, end: 5, label: "fr", source: "w", granularity: "clip" });
    const json = JSON.parse(JSON.stringify(t)) as { schemaVersion: string; annotations: { id: string; kind: string; start: number }[] };
    expect(json.schemaVersion).toBe("1.1");
    expect(json.annotations.map((a) => a.id)).toEqual(["language-0", "speaker-0", "word-1", "word-0"]);
    expect(json.annotations[1]).toMatchObject({ kind: "speaker", value: null, confidence: null, speakerId: "SPEAKER_00" });
    expect(t.during("word", 2, 4.5).map((a) => a.label)).toEqual(["a", "b"]);
  });

  it("keeps half-open, non-empty intervals and object values", () => {
    const t = new Timeline();
    expect(() => t.add({ kind: "speech", start: 3, end: 2, source: "v", granularity: "segment" })).toThrow(/start < end/);
    t.add({ kind: "speech", start: 2, end: 2, source: "v", granularity: "segment" });
    expect(t.annotations).toHaveLength(0);
    t.add({ kind: "profile", start: 0, end: 10, label: "profile", value: { snrDb: 30, clipping: 0 }, source: "profile", granularity: "clip" });
    expect(t.toJSON().annotations[0]!.value).toEqual({ snrDb: 30, clipping: 0 });
  });

  it("refuses unknown analyses with the close ones, and unshipped ones saying so", () => {
    expect(checkAnalyses(["speakers", "words", "speakers"])).toEqual(["speakers", "words"]);
    expect(() => checkAnalyses(["speaker"])).toThrow(/did you mean 'speakers'/);
    expect(checkAnalyses(["emotion", "events", "music"])).toEqual(["emotion", "events", "music"]);
    expect(checkAnalyses(undefined)).toEqual([]);
  });
});

describe("renderings", () => {
  const result = {
    text: "Hello there. Hi! How are you?",
    language: "en",
    segments: [
      { id: 0, start: 0, end: 1.5, text: " Hello there.", speaker: "SPEAKER_00" },
      { id: 1, start: 1.5, end: 2.5, text: " Hi!", speaker: "SPEAKER_01" },
      { id: 2, start: 2.5, end: 4, text: " How are you?", speaker: "SPEAKER_01" },
    ],
    timeline: [],
  };

  it("writes Markdown by speaker turn, an unknown voice as Speaker N, an enrolled one by name", () => {
    expect(renderMarkdown(result)).toBe("_00:04 · en · 2 speakers_\n\n## Speaker 1 · 00:00\nHello there.\n\n## Speaker 2 · 00:02\nHi!\nHow are you?\n");
    const named = { ...result, speakerInfo: [{ speakerId: "SPEAKER_01", name: "Loïc", similarity: 0.9 }] };
    expect(renderMarkdown(named)).toContain("## Loïc · 00:02\nHi!");
  });

  it("puts the speaker at the head of each subtitle line, nothing else", () => {
    expect(renderResult(result, "srt")).toContain("00:00:01,500 --> 00:00:02,500\nSpeaker 2: Hi!");
    expect(renderResult(result, "vtt")).toContain("Speaker 1: Hello there.");
  });

  it("writes one CSV row per segment with what is known", () => {
    expect(renderCsv(result).split("\r\n")[0]).toBe("id,start,end,speaker,language,text");
  });

  it("keeps the text alone in TXT", () => {
    expect(renderResult(result, "txt")).toBe("Hello there. Hi! How are you?\n");
    expect(() => renderResult(result, "pdf")).toThrow(/Unknown format/);
  });
});
