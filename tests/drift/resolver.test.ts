import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import { describe, expect, it } from "vitest";

import {
  choose,
  explain,
  label,
  ladder,
  levelFor,
  parseAuto,
  resolveSettings,
  retryRung,
  type Level,
  type Measurements,
} from "../../src/whisper/resolver.js";

interface Choice {
  level: Level;
  language: string;
  memory: number | null;
  model: string | null;
  noisy: boolean;
  chosen: { model: string; precision: { encoder: string; decoder: string }; rung: number; wer: number; rtf: number; estimated: boolean } | null;
  reason: string;
}

const f = JSON.parse(
  readFileSync(join(dirname(fileURLToPath(import.meta.url)), "..", "fixtures", "resolver.json"), "utf-8"),
) as {
  tables: Measurements;
  choices: Choice[];
  retries: { model: string; current: string | { encoder: string; decoder: string }; memory: number | null; rung: unknown }[];
  ladder: { encoder: string; decoder: string }[];
};

describe("the resolver, against Python's on the same tables (CPU read as WebAssembly)", () => {
  it("orders the ladder the same way", () => {
    expect(ladder("wasm")).toEqual(f.ladder);
  });

  it(`makes the same ${f.choices.length} choices`, () => {
    for (const c of f.choices) {
      const got = choose(c.level, { devices: [{ device: "wasm", memoryMb: c.memory }] }, f.tables, {
        language: c.language,
        model: c.model,
        noisy: c.noisy,
      });
      const where = `${c.level} ${c.language} ${c.memory} ${c.model} noisy=${c.noisy}`;
      if (c.chosen === null) {
        expect(got.candidate, where).toBeNull();
        continue;
      }
      expect(got.candidate, where).not.toBeNull();
      const { model, precision, rung, wer, rtf, estimated } = got.candidate!;
      expect({ model, precision, rung, wer, rtf, estimated }, where).toEqual(c.chosen);

      expect(got.reason, where).toBe(c.reason);
    }
  });

  it("retries at the same rungs", () => {
    for (const r of f.retries) {
      expect(retryRung(r.model, r.current, "wasm", "en", r.memory, f.tables), `${r.model} ${label(r.current)} ${r.memory}`).toEqual(r.rung);
    }
  });
});

describe("auto values", () => {
  it("reads levels, refuses unknown ones", () => {
    expect(parseAuto("auto")).toBe("");
    expect(parseAuto("auto:fast")).toBe("fast");
    expect(parseAuto("small")).toBeNull();
    expect(() => parseAuto("auto:fastest")).toThrow(/unknown automatic level/);
    expect(levelFor("auto", "balanced")).toBe("balanced");
    expect(levelFor("int8", "balanced")).toBeNull();
  });
});

describe("resolveSettings", () => {
  const hardware = { devices: [{ device: "webgpu" as const, memoryMb: null }, { device: "wasm" as const, memoryMb: 4096 }] };

  it("keeps explicit values and chooses the rest around them", () => {
    const r = resolveSettings({ model: "small", language: "en", hardware, measurements: f.tables });
    expect(r.decisions.model).toEqual({ value: "small", source: "explicit", reason: "" });
    expect(r.decisions.computeType.source).toBe("auto:quality");
    expect(r.decisions.vad.value).toBe("off");
    expect(explain(r)).toContain("model = \"small\"  [explicit]");
  });

  it("turns VAD on for a large model", () => {
    const r = resolveSettings({ model: "large-v3-turbo", language: "fr", hardware, measurements: f.tables });
    expect(r.decisions.vad.value).toBe("on");
  });

  it("serves the smallest measured model when nothing is measured for this device", () => {
    const r = resolveSettings({ language: "en", hardware: { devices: [{ device: "wasm", memoryMb: 10 }] }, measurements: f.tables });
    expect(r.warnings.join()).toMatch(/no measured model fits/);
  });
});

describe("AudioTranscriber.resolve", () => {
  it("decides before loading, on the browser tables, and says why", async () => {
    const { AudioTranscriber } = await import("../../src/whisper/transcriber.js");
    const resolution = await AudioTranscriber.resolve({ ort: {} as never, Tensor: {} as never, model: "auto", language: "en", backends: "wasm" });
    expect(resolution).not.toBeNull();
    expect(resolution!.decisions.backend.value).toBe("wasm");
    expect(resolution!.decisions.model.source).toBe("auto:quality");
    expect(explain(resolution!)).toMatch(/computeType = .*rung/);
  });

  it("leaves explicit settings alone", async () => {
    const { AudioTranscriber } = await import("../../src/whisper/transcriber.js");
    expect(await AudioTranscriber.resolve({ ort: {} as never, Tensor: {} as never, model: "small", computeType: "int8" })).toBeNull();
  });
});
