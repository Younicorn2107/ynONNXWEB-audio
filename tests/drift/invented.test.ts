import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import { describe, expect, it } from "vitest";

import { isCreditLine, isInventedLine, isNonSpeechLine } from "../../src/whisper/invented.js";

/** The Python library's verdicts on edge cases (Unicode `\w`, accents, digits, dashes). */
const cases = JSON.parse(
  readFileSync(join(dirname(fileURLToPath(import.meta.url)), "..", "fixtures", "invented_lines.json"), "utf-8"),
) as { text: string; credit: boolean; nonSpeech: boolean; invented: boolean }[];

describe("invented lines, against Python's verdicts", () => {
  it(`gives the same ${cases.length} verdicts`, () => {
    for (const c of cases) {
      expect({ text: c.text, credit: isCreditLine(c.text), nonSpeech: isNonSpeechLine(c.text), invented: isInventedLine(c.text) }).toEqual(c);
    }
  });
});
