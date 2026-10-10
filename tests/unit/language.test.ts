import { describe, expect, it } from "vitest";

import { dominantLanguage, mergePortions, portionAt } from "../../src/whisper/transcriber.js";

const p = (language: string, start: number, probability: number) => ({ language, start, end: start + 30, probability });

describe("the language over time", () => {
  it("merges windows of one language and lets an unsure one follow its neighbour", () => {
    const portions = mergePortions([p("fr", 0, 0.9), p("fr", 30, 0.8), p("es", 60, 0.3), p("en", 90, 0.95), p("en", 120, 0.9)]);
    expect(portions.map((x) => [x.language, x.start, x.end])).toEqual([
      ["fr", 0, 90],
      ["en", 90, 150],
    ]);
    expect(portions[0]!.probability).toBeCloseTo((0.9 + 0.8 + 0.3) / 3, 9);
  });

  it("takes the next window's language for an unsure first one", () => {
    expect(mergePortions([p("de", 0, 0.2), p("en", 30, 0.9)]).map((x) => x.language)).toEqual(["en"]);
  });

  it("names the audio after its longest language, and finds the portion of a time", () => {
    const portions = mergePortions([p("fr", 0, 0.9), p("en", 30, 0.9), p("en", 60, 0.7)]);
    expect(dominantLanguage(portions)).toEqual({ language: "en", probability: 0.8 });
    expect(portionAt(portions, 10)).toBe("fr");
    expect(portionAt(portions, 45)).toBe("en");
    expect(portionAt(null, 10)).toBeNull();
  });
});
