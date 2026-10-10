/**
 * The execution plan `auto` builds before running anything (spec § 7), and the licences of the
 * models it uses (spec § 8).
 */

import type { Analysis } from "./timeline.js";
import type { Resolution } from "./whisper/resolver.js";

export type LicenseCategory = "permissive" | "copyleft" | "non_commercial" | "research_only" | "unknown";

export interface ModelLicense {
  license: string;
  category: LicenseCategory;
}

/** The licence of every model this library can load. */
export const MODEL_LICENSES: Readonly<Record<string, ModelLicense>> = {
  whisper: { license: "MIT", category: "permissive" },
  "distil-whisper": { license: "MIT", category: "permissive" },
  moonshine: { license: "MIT", category: "permissive" },
  "onnx-community/silero-vad": { license: "MIT", category: "permissive" },
  "onnx-community/wav2vec2-base-960h-ONNX": { license: "Apache-2.0", category: "permissive" },
  "onnx-community/wespeaker-voxceleb-resnet34-LM": { license: "CC-BY-4.0", category: "permissive" },
  "onnx-community/pyannote-segmentation-3.0": { license: "MIT", category: "permissive" },
  "onnx-community/wav2vec2-emotion-recognition-ONNX": { license: "MIT", category: "permissive" },
  "onnx-community/ast-finetuned-audioset-10-10-0.4593-ONNX": { license: "BSD-3-Clause", category: "permissive" },
};

export function licenseOf(model: string): ModelLicense {
  if (MODEL_LICENSES[model]) return MODEL_LICENSES[model]!;
  if (/moonshine/i.test(model)) return MODEL_LICENSES["moonshine"]!;
  if (/distil/i.test(model)) return MODEL_LICENSES["distil-whisper"]!;
  if (/whisper|^(tiny|base|small|medium|large)/i.test(model)) return MODEL_LICENSES["whisper"]!;
  return { license: "unknown", category: "unknown" };
}

/**
 * Which licence categories a caller accepts: everything by default; `{ non_commercial: false }`
 * refuses those. A model the caller NAMED is loaded anyway, with a warning.
 */
export type LicensePolicy = Partial<Record<LicenseCategory, boolean>>;

export function allowed(license: ModelLicense, policy: LicensePolicy = {}): boolean {
  return policy[license.category] !== false;
}

export type StepName = "speech" | "profile" | "language" | "transcribe" | "words" | "speakers" | "emotion" | "events" | "music";

/**
 * Where a choice came from (spec § 7): asked by the caller, decided by `auto`, a default, or needed
 * by another step.
 */
export type Origin = "caller" | "auto" | "default" | "dependency";

/** The plan's schema version: what a reader of a serialized plan can expect. */
export const PLAN_SCHEMA_VERSION = "1.1";

/** Whether anything may leave for the network (spec § 8.2); downloading a model is not counted. */
export type NetworkPolicy = "offline" | "allowed";

export interface PlanStep {
  step: StepName;
  /** The model, or null for a measurement that needs none. */
  model: string | null;
  device: string;
  precision: string | null;
  license: ModelLicense | null;
  /** Steps that must finish first. */
  dependsOn: StepName[];
  /** Why the step is there: asked for, decided by `auto`, or needed by another one. */
  origin: Origin;
  reason: string;
}

export interface ExecutionPlan {
  schemaVersion: string;
  steps: PlanStep[];
  /** No service is used offline; none is configured today either way. */
  network: NetworkPolicy;
  /** How steps share the device: one page drives one GPU, so steps run in order. */
  workers: "sequential" | number;
  resolution: Resolution | null;
  warnings: string[];
}

export interface PlanInputs {
  model: string;
  device: string;
  precision: string;
  /** Whether the language is already known (given, or an English-only model). */
  languageKnown: boolean;
  /** Whether silence is removed before decoding. */
  vad: boolean;
  analyses: readonly Analysis[];
  wordTimestamps: boolean;
  ctcModel: string | null;
  diarization: { embeddingModel: string; segmentationModel: string | null };
  /** False for `analyze()`: the analyses alone, no transcript. */
  transcribe?: boolean;
  /** The emotion and AudioSet models, when those analyses are asked for. */
  emotionModel?: string;
  eventsModel?: string;
  resolution: Resolution | null;
  /**
   * Licence categories accepted; the transcription model is always the caller's choice or `auto`'s.
   */
  licenses?: LicensePolicy;
  workers?: "auto" | "sequential" | number;
  network?: NetworkPolicy;
}

/** The plan, from what is known before anything loads. */
export function buildPlan(inputs: PlanInputs): ExecutionPlan {
  const steps: PlanStep[] = [];
  const warnings: string[] = [];
  const policy = inputs.licenses ?? {};
  const add = (step: PlanStep) => {
    if (step.license && !allowed(step.license, policy)) {
      warnings.push(`${step.model} (${step.license.license}, ${step.license.category}) is outside the licence policy; kept, since the step was asked for`);
    }
    steps.push(step);
  };
  const silero = "onnx-community/silero-vad";
  if (inputs.vad || inputs.analyses.includes("speech")) {
    const asked = inputs.analyses.includes("speech");
    add({ step: "speech", model: silero, device: "wasm", precision: "float32", license: licenseOf(silero), dependsOn: [], origin: asked ? "caller" : "auto", reason: asked ? "speech was asked for" : "silences are removed before decoding (a model of 500 M parameters or more)" });
  }
  if (inputs.analyses.includes("profile")) {
    add({ step: "profile", model: null, device: "cpu", precision: null, license: null, dependsOn: [], origin: "caller", reason: "signal measurements, no model" });
  }
  const transcribing = inputs.transcribe ?? true;
  if (transcribing ? !inputs.languageKnown : inputs.analyses.includes("language")) {
    const asked = inputs.analyses.includes("language");
    add({ step: "language", model: inputs.model, device: inputs.device, precision: inputs.precision, license: licenseOf(inputs.model), dependsOn: [], origin: asked ? "caller" : "dependency", reason: asked ? "the language was asked for" : "three probes and a vote, before the prompt can name the language" });
  }
  const before = steps.filter((s) => s.step === "speech" || s.step === "language").map((s) => s.step);
  if (transcribing) add({
    step: "transcribe",
    model: inputs.model,
    device: inputs.device,
    precision: inputs.precision,
    license: licenseOf(inputs.model),
    dependsOn: before,
    origin: "caller",
    reason: inputs.resolution?.decisions.model.reason || "as asked",
  });
  if (inputs.wordTimestamps || inputs.analyses.includes("words")) {
    add({
      step: "words",
      model: inputs.ctcModel,
      device: "wasm",
      precision: inputs.ctcModel ? "int8" : null,
      license: inputs.ctcModel ? licenseOf(inputs.ctcModel) : null,
      dependsOn: ["transcribe"],
      origin: "caller",
      reason: inputs.ctcModel ? "CTC forced alignment on each segment" : "no CTC model configured: a proportional split, not an alignment",
    });
  }
  if (inputs.analyses.includes("speakers")) {
    const { embeddingModel, segmentationModel } = inputs.diarization;
    add({ step: "speakers", model: embeddingModel, device: "wasm", precision: "int8", license: licenseOf(embeddingModel), dependsOn: transcribing ? ["transcribe"] : [], origin: "caller", reason: segmentationModel ? `segmented by ${segmentationModel}, embedded, clustered` : "speech sliced into overlapping chunks, embedded, clustered" });
    if (segmentationModel) {
      const license = licenseOf(segmentationModel);
      if (!allowed(license, policy)) warnings.push(`${segmentationModel} (${license.license}) is outside the licence policy; kept, since speakers were asked for`);
    }
  }
  const emotionModel = inputs.emotionModel ?? "onnx-community/wav2vec2-emotion-recognition-ONNX";
  const eventsModel = inputs.eventsModel ?? "onnx-community/ast-finetuned-audioset-10-10-0.4593-ONNX";
  if (inputs.analyses.includes("emotion")) {
    const after: StepName[] = inputs.analyses.includes("speakers") ? ["speakers"] : [];
    if (!inputs.analyses.includes("speakers") && !steps.some((s) => s.step === "speech")) {
      add({ step: "speech", model: silero, device: "wasm", precision: "float32", license: licenseOf(silero), dependsOn: [], origin: "dependency", reason: "emotion is read on speech spans" });
    }
    if (!after.length) after.push("speech");
    add({ step: "emotion", model: emotionModel, device: "webgpu", precision: "float16", license: licenseOf(emotionModel), dependsOn: after, origin: "caller", reason: "one emotion per region of speech (not yet measured: model and thresholds wait for a labelled corpus)" });
  }
  for (const kind of ["events", "music"] as const) {
    if (!inputs.analyses.includes(kind)) continue;
    add({ step: kind, model: eventsModel, device: "webgpu", precision: "float16", license: licenseOf(eventsModel), dependsOn: [], origin: "caller", reason: `AudioSet classes on 10-second windows${kind === "music" ? "; no title (no online service configured)" : ""}` });
  }
  const workers = typeof inputs.workers === "number" ? Math.max(1, Math.trunc(inputs.workers)) : "sequential";
  return { schemaVersion: PLAN_SCHEMA_VERSION, steps, network: inputs.network ?? "offline", workers, resolution: inputs.resolution, warnings };
}

/** The plan as text: one step per line, its model, device and reason. */
export function explainPlan(plan: ExecutionPlan): string {
  const lines = plan.steps.map((s) => {
    const after = s.dependsOn.length > 0 ? ` after ${s.dependsOn.join(", ")}` : "";
    const model = s.model ? `${s.model}${s.precision ? ` @ ${s.precision}` : ""} on ${s.device}` : `no model, ${s.device}`;
    const license = s.license ? ` [${s.license.license}]` : "";
    return `${s.step}: ${model}${license}${after} — ${s.reason}`;
  });
  lines.push(`workers: ${plan.workers}`);
  lines.push(...plan.warnings.map((w) => `warning: ${w}`));
  return lines.join("\n");
}

/** One line of `explain()` (spec § 7): a setting, its value, where it came from and why. */
export interface PlanEntry {
  parameter: string;
  value: unknown;
  origin: Origin;
  reason: string;
}

const originOf = (source: string): Origin => (source === "explicit" ? "caller" : source.startsWith("auto") ? "auto" : "default");

/** Every setting of a plan with its origin: `auto`'s decisions, then each step. */
export function explainEntries(plan: ExecutionPlan): PlanEntry[] {
  const entries: PlanEntry[] = [];
  for (const [parameter, decision] of Object.entries(plan.resolution?.decisions ?? {})) {
    if (decision) entries.push({ parameter, value: decision.value, origin: originOf(decision.source), reason: decision.reason });
  }
  for (const step of plan.steps) {
    entries.push({ parameter: `step:${step.step}`, value: step.model ?? "no model", origin: step.origin, reason: step.reason });
  }
  entries.push({ parameter: "network", value: plan.network, origin: plan.network === "offline" ? "default" : "caller", reason: "no online service is configured" });
  return entries;
}
