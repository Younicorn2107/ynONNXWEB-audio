import { describe, expect, it } from "vitest";

import { displayName, speakerInfoOf } from "../../src/diarization/diarizer.js";
import { StreamingError } from "../../src/errors.js";
import { buildPlan, explainEntries } from "../../src/plan.js";

const inputs = {
  model: "whisper-small",
  device: "webgpu",
  precision: "float16",
  languageKnown: false,
  vad: true,
  analyses: ["speakers"] as const,
  wordTimestamps: false,
  ctcModel: null,
  diarization: { embeddingModel: "onnx-community/wespeaker-voxceleb-resnet34-LM", segmentationModel: null },
  resolution: null,
};

describe("the plan says where every step comes from (spec § 7)", () => {
  it("marks steps asked for, decided by auto, and needed by another", () => {
    const plan = buildPlan({ ...inputs, analyses: ["speakers"] });
    expect(plan.schemaVersion).toBe("1.1");
    expect(plan.network).toBe("offline");
    expect(Object.fromEntries(plan.steps.map((s) => [s.step, s.origin]))).toEqual({ speech: "auto", language: "dependency", transcribe: "caller", speakers: "caller" });
    expect(buildPlan({ ...inputs, analyses: ["speech", "language"] }).steps.filter((s) => s.step !== "transcribe").map((s) => s.origin)).toEqual(["caller", "caller"]);
  });

  it("explains each setting as parameter, value, origin and reason", () => {
    const entries = explainEntries(buildPlan({ ...inputs, analyses: ["speakers"], network: "allowed" }));
    expect(entries.find((e) => e.parameter === "step:language")).toMatchObject({ origin: "dependency" });
    expect(entries.find((e) => e.parameter === "network")).toMatchObject({ value: "allowed", origin: "caller" });
  });
});

describe("speakers: an identity, and a name apart (spec § 6.1)", () => {
  it("lists each speaker once, in order, with its enrolled name and similarity", () => {
    const turns = [
      { speaker: "SPEAKER_00", start: 0, end: 2, name: "Loïc" },
      { speaker: "SPEAKER_01", start: 2, end: 3, name: null },
      { speaker: "SPEAKER_00", start: 3, end: 5, name: "Loïc" },
    ];
    expect(speakerInfoOf(turns, new Map([["SPEAKER_00", 0.91]]))).toEqual([
      { speakerId: "SPEAKER_00", name: "Loïc", similarity: 0.91 },
      { speakerId: "SPEAKER_01", name: null, similarity: null },
    ]);
  });

  it("shows an unknown voice as Speaker N and an enrolled one by name", () => {
    expect(displayName("SPEAKER_00")).toBe("Speaker 1");
    expect(displayName("SPEAKER_11", null)).toBe("Speaker 12");
    expect(displayName("SPEAKER_01", "Zira")).toBe("Zira");
  });

  it("has a streaming error that keeps its name across a boundary", () => {
    expect(new StreamingError("revision after final").code).toBe("StreamingError");
  });
});
