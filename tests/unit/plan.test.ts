import { describe, expect, it } from "vitest";

import { buildPlan, explainPlan, licenseOf } from "../../src/plan.js";

const base = {
  model: "whisper-large-v3-turbo_int8",
  device: "webgpu",
  precision: "float16",
  languageKnown: false,
  vad: true,
  analyses: [] as const,
  wordTimestamps: false,
  ctcModel: null,
  diarization: { embeddingModel: "onnx-community/wespeaker-voxceleb-resnet34-LM", segmentationModel: "onnx-community/pyannote-segmentation-3.0" },
  resolution: null,
};

describe("the execution plan", () => {
  it("orders the steps by dependency", () => {
    const plan = buildPlan({ ...base, analyses: ["speakers", "words", "profile"], ctcModel: "onnx-community/wav2vec2-base-960h-ONNX" });
    expect(plan.steps.map((s) => s.step)).toEqual(["speech", "profile", "language", "transcribe", "words", "speakers"]);
    expect(plan.steps.find((s) => s.step === "transcribe")!.dependsOn).toEqual(["speech", "language"]);
    expect(plan.steps.find((s) => s.step === "speakers")!.dependsOn).toEqual(["transcribe"]);
    expect(explainPlan(plan)).toContain("words: onnx-community/wav2vec2-base-960h-ONNX @ int8 on wasm [Apache-2.0] after transcribe");
  });

  it("drops what is not needed, and says when words are only a split", () => {
    const plan = buildPlan({ ...base, languageKnown: true, vad: false, wordTimestamps: true });
    expect(plan.steps.map((s) => s.step)).toEqual(["transcribe", "words"]);
    expect(plan.steps[1]!.reason).toMatch(/proportional split/);
  });

  it("keeps a step outside the licence policy, with a warning", () => {
    const plan = buildPlan({ ...base, analyses: ["speakers"], licenses: { permissive: false } });
    expect(plan.warnings.length).toBeGreaterThan(0);
    expect(licenseOf("small").license).toBe("MIT");
    expect(licenseOf("someone/thing").category).toBe("unknown");
  });
});

describe("enrolled speakers", () => {
  it("names the closest cluster above the threshold, one name per cluster", async () => {
    const { matchEnrolled } = await import("../../src/diarization/diarizer.js");
    const e = (x: number, y: number) => Float32Array.of(x, y);
    const embeddings = [e(1, 0), e(0.9, 0.1), e(0, 1), e(0.1, 0.9), e(-1, 0)];
    const labels = [0, 0, 1, 1, 2];
    const names = matchEnrolled(embeddings, labels, { anna: [1, 0], bob: [0, 1], carl: [0.8, 0.6] }, 0.5);
    expect(names.get(0)).toBe("anna");
    expect(names.get(1)).toBe("bob");

    expect(names.has(2)).toBe(false);
  });
});
