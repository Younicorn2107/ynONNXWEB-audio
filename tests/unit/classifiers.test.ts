import { describe, expect, it } from "vitest";

import { audioSetKind, canonicalEmotion, canonicalEvent } from "../../src/analysis/classifiers.js";
import { buildPlan } from "../../src/plan.js";

describe("emotion, events and music", () => {
  it("maps every published emotion vocabulary onto the canonical one", () => {
    for (const label of ["angry", "calm", "disgust", "fearful", "happy", "sad", "surprised"]) expect(canonicalEmotion(label)).toBe(label);
    expect(["ANG", "CAL", "DIS", "FEA", "HAP", "NEU", "SAD", "SUR"].map(canonicalEmotion)).toEqual(["angry", "calm", "disgust", "fearful", "happy", "neutral", "sad", "surprised"]);
    expect(["SAD", "ANGRY", "DISGUST", "FEAR", "HAPPY", "NEUTRAL"].map(canonicalEmotion)).toEqual(["sad", "angry", "disgust", "fearful", "happy", "neutral"]);
    expect(canonicalEmotion("boredom")).toBeNull();
  });

  it("keeps speech, music and the room's acoustics out of events", () => {
    expect([0, 7, 16, 67, 137, 282, 300, 500, 511, 513].map(audioSetKind)).toEqual(["speech", "speech", "event", "event", "music", "music", "event", "ambience", "ambience", "event"]);
    expect(canonicalEvent("Applause")).toBe("applause");
  });

  it("plans emotion after the speakers, or after speech spans it asks for itself", () => {
    const base = { model: "whisper-small", device: "wasm", precision: "int8", languageKnown: true, vad: false, wordTimestamps: false, ctcModel: null, diarization: { embeddingModel: "w", segmentationModel: null }, resolution: null };
    const alone = buildPlan({ ...base, analyses: ["emotion"] });
    expect(alone.steps.map((s) => [s.step, s.origin])).toEqual([["transcribe", "caller"], ["speech", "dependency"], ["emotion", "caller"]]);
    expect(alone.steps.find((s) => s.step === "emotion")!.dependsOn).toEqual(["speech"]);
    const withSpeakers = buildPlan({ ...base, analyses: ["speakers", "emotion", "events", "music"] });
    expect(withSpeakers.steps.find((s) => s.step === "emotion")!.dependsOn).toEqual(["speakers"]);
    expect(withSpeakers.steps.filter((s) => s.step === "events" || s.step === "music").map((s) => s.license?.category)).toEqual(["permissive", "permissive"]);

    const analysis = buildPlan({ ...base, languageKnown: false, transcribe: false, analyses: ["speakers", "emotion"] });
    expect(analysis.steps.map((s) => s.step)).toEqual(["speakers", "emotion"]);
    expect(analysis.steps[0]!.dependsOn).toEqual([]);
  });
});
