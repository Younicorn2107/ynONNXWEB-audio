/** Automatic settings: say what you want rather than how to get it. */

import { ConfigurationError } from "@youni/onnxweb-core/core";

import type { AudioProfile } from "../audio/profile.js";
import { LARGE_MODEL_PARAMS_M, WHISPER_PARAMS_M, shouldUseVad } from "../audio/vadPolicy.js";
import type { DecodingProfile } from "./profiles.js";
import { asRoles, type HalfPrecision } from "./published.js";

export const LEVELS = ["quality", "balanced", "fast", "max_speed"] as const;
export type Level = (typeof LEVELS)[number];

/** What bare `"auto"` means when the session says nothing. */
export const DEFAULT_LEVEL: Level = "quality";

/** WER points tolerated above the best candidate, per level. */
export const WER_BUDGET_POINTS: Readonly<Record<Level, number>> = {
  quality: 0.5,
  balanced: 1.5,
  fast: 4,
  max_speed: Number.POSITIVE_INFINITY,
};

/** The measured decoding profile each level uses. */
export const DECODING_ROLE: Readonly<Record<Level, DecodingProfile>> = {
  quality: "quality",
  balanced: "default",
  fast: "speed",
  max_speed: "speed",
};

/** Below this SNR estimate the audio is noisy: no int4 half when another rung fits. */
export const NOISY_SNR_DB = 12.5;

/** Share of the memory a model may take. */
export const MEMORY_MARGIN = 0.9;

/** Loaded weights against their size on disk (ynonnx_core's figure). */
const INFERENCE_OVERHEAD = 1.25;

/** float32 weights per million parameters, in MB. */
const MB_PER_MILLION_PARAMS = 4;

export type Device = "webgpu" | "wasm";

/** The precisions of each half the published variants offer, per device. */
export const PRECISION_LADDER: Readonly<Record<Device, readonly { encoder: string; decoder: string }[]>> = {
  webgpu: ["float16", "int4"].flatMap((encoder) => ["float32", "int4"].map((decoder) => ({ encoder, decoder }))),
  wasm: ["float16", "int8", "int4"].flatMap((encoder) => ["float32", "int8", "int4"].map((decoder) => ({ encoder, decoder }))),
};

const WIDTH: Readonly<Record<string, number>> = { float32: 32, float16: 16, int8: 8, int4: 4 };

export interface ModelMeasurements {
  /** `wer[device][corpus][label]`, in %. `en` is English, `other` every other language (French). */
  wer?: Partial<Record<Device, Partial<Record<"en" | "other", Record<string, number>>>>>;
  /** `rtf[device][label]`, times real time. */
  rtf?: Partial<Record<Device, Record<string, number>>>;
  /** `size_mb[label]`: the graphs actually loaded. */
  size_mb?: Record<string, number>;
  /** `failed[device]`: defective there (loops, lost passages, a wrong kernel). */
  failed?: Partial<Record<Device, string[]>>;
}

export type Measurements = Readonly<Record<string, ModelMeasurements>>;

/** Whether a checkpoint is multilingual (`.en` and the distil checkpoints are not). */
export function isMultilingualModel(model: string): boolean {
  return !model.endsWith(".en") && !model.startsWith("distil-");
}

/** The level an `auto` value carries: `""` for bare `auto`, `null` when the value is explicit. */
export function parseAuto(value: unknown): Level | "" | null {
  if (typeof value !== "string") return null;
  const text = value.trim().toLowerCase();
  if (text === "auto") return "";
  if (!text.startsWith("auto:")) return null;
  const level = text.slice("auto:".length);
  if (!(LEVELS as readonly string[]).includes(level)) {
    throw new ConfigurationError(
      `unknown automatic level: '${value}' — expected auto or ${LEVELS.map((l) => `auto:${l}`).join(", ")}`,
    );
  }
  return level as Level;
}

export function checkLevel(level: string): Level {
  if (level === "auto") return DEFAULT_LEVEL;
  if (!(LEVELS as readonly string[]).includes(level)) {
    throw new ConfigurationError(`unknown profile: '${level}' — expected ${LEVELS.join(", ")}`);
  }
  return level as Level;
}

/** The level that decides this value, or `null` when it is explicit. */
export function levelFor(value: unknown, session: Level): Level | null {
  const level = parseAuto(value);
  if (level === null) return null;
  return level || session;
}

/** The stable name of a precision, the one the tables use: `mix_decoder-int4_encoder-int8`. */
export function label(precision: HalfPrecision): string {
  if (typeof precision === "string") return precision;
  const parts = Object.entries(precision)
    .filter((entry): entry is [string, string] => typeof entry[1] === "string")
    .sort(([a], [b]) => a.localeCompare(b))
    .map(([role, value]) => `${role}-${value}`);
  return `mix_${parts.join("_")}`;
}

function width(precision: HalfPrecision): number {
  if (typeof precision === "string") return WIDTH[precision] ?? 8;
  const values = Object.values(precision).filter((v): v is string => typeof v === "string");
  return values.reduce((s, p) => s + (WIDTH[p] ?? 8), 0) / Math.max(values.length, 1);
}

function cellWidth(key: string): number {
  if (!key.startsWith("mix_")) return WIDTH[key] ?? 8;
  const roles = Object.fromEntries(key.slice(4).split("_").map((part) => part.split("-", 2) as [string, string]));
  return width(roles);
}

/** A device's rungs, widest first. */
export function ladder(device: Device): { encoder: string; decoder: string }[] {
  return [...PRECISION_LADDER[device]].sort(
    (a, b) => width(b) - width(a) || (WIDTH[b.encoder] ?? 8) - (WIDTH[a.encoder] ?? 8),
  );
}

function bucket(language: string | null | undefined): "en" | "other" {
  return (language ?? "").split("-")[0]!.toLowerCase() === "en" ? "en" : "other";
}

/** The WER cells that apply on a device: its own, completed by the other device's. */
export function werCells(measured: ModelMeasurements, device: Device, corpus: "en" | "other"): Record<string, number> {
  const cells: Record<string, number> = {};
  for (const [table, corpora] of Object.entries(measured.wer ?? {})) {
    if (table !== device) Object.assign(cells, corpora?.[corpus] ?? {});
  }
  Object.assign(cells, measured.wer?.[device]?.[corpus] ?? {});
  return cells;
}

function nearest(cells: Record<string, number>, key: string): [number, string] | null {
  const keys = Object.keys(cells);
  if (keys.length === 0) return null;
  if (key in cells) return [cells[key]!, key];
  const target = cellWidth(key);
  const best = keys.reduce((a, b) => {
    const da = Math.abs(cellWidth(a) - target);
    const db = Math.abs(cellWidth(b) - target);
    if (da !== db) return da < db ? a : b;
    return cellWidth(a) >= cellWidth(b) ? a : b;
  });
  return [cells[best]!, best];
}

/** Memory a loaded model takes, in MB, or null when it is not in the catalogue. */
export function modelMemoryMb(model: string, precision: HalfPrecision, measurements: Measurements): number | null {
  const params = WHISPER_PARAMS_M[model];
  if (params === undefined) return null;
  const key = label(asRoles(precision));
  const measured = measurements[model]?.size_mb?.[key];
  if (measured !== undefined) return measured * INFERENCE_OVERHEAD;
  const roles = asRoles(precision);
  const widest = Math.max(WIDTH[roles.encoder] ?? 32, WIDTH[roles.decoder] ?? 32);
  return params * MB_PER_MILLION_PARAMS * (widest / 32) * INFERENCE_OVERHEAD;
}

/** Whether the model fits; true when the memory or its size is unknown. */
export function fits(model: string, precision: HalfPrecision, memoryMb: number | null, measurements: Measurements): boolean {
  const needed = modelMemoryMb(model, precision, measurements);
  if (needed === null || memoryMb === null) return true;
  return needed <= memoryMb * MEMORY_MARGIN;
}

export interface Candidate {
  model: string;
  device: Device;
  precision: { encoder: string; decoder: string };
  /** Position in the device's ladder: 0 is the preferred precision. */
  rung: number;
  wer: number;
  rtf: number;
  /** Whether the figures were borrowed from another precision. */
  estimated: boolean;
}

/** The measurable triples that fit on this device. */
export function candidates(
  device: Device,
  memoryMb: number | null,
  language: string | null | undefined,
  models: readonly string[],
  precisions: readonly HalfPrecision[],
  measurements: Measurements,
): Candidate[] {
  const corpus = bucket(language);
  const kept: Candidate[] = [];
  for (const name of models) {
    const measured = measurements[name];
    if (!measured || WHISPER_PARAMS_M[name] === undefined) continue;
    if (!isMultilingualModel(name) && corpus !== "en") continue;
    const broken = new Set(measured.failed?.[device] ?? []);
    precisions.forEach((precision, rung) => {
      const roles = asRoles(precision);
      if (!PRECISION_LADDER[device].some((r) => r.encoder === roles.encoder && r.decoder === roles.decoder)) return;
      const key = label(roles);
      if (broken.has(key) || !fits(name, roles, memoryMb, measurements)) return;
      const wer = nearest(werCells(measured, device, corpus), key);
      const rtf = nearest(measured.rtf?.[device] ?? {}, key);
      if (!wer || !rtf) return;
      kept.push({ model: name, device, precision: roles, rung, wer: wer[0], rtf: rtf[0], estimated: wer[1] !== key || rtf[1] !== key });
    });
  }
  return kept;
}

/** A level's candidate, and why. */
export function pick(level: Level, found: readonly Candidate[], noisy = false): [Candidate, string] {
  const robust = noisy ? found.filter((c) => c.precision.encoder !== "int4" && c.precision.decoder !== "int4") : found;
  const pool = robust.length > 0 ? robust : found;
  const best = Math.min(...pool.map((c) => c.wer));
  const budget = WER_BUDGET_POINTS[level];
  const eligible = pool.filter((c) => c.wer <= best + budget);
  const chosen = eligible.reduce((a, b) => {
    if (a.rtf !== b.rtf) return a.rtf > b.rtf ? a : b;
    if (a.wer !== b.wer) return a.wer < b.wer ? a : b;
    return a.rung <= b.rung ? a : b;
  });
  let reason = Number.isFinite(budget)
    ? `fastest (${chosen.rtf}x real time) within ${budget} point of the best WER (${best.toFixed(2)} %, this one ${chosen.wer.toFixed(2)} %)`
    : `fastest (${chosen.rtf}x real time)`;
  if (noisy && robust.length > 0) reason += "; noisy audio: no int4 half";
  if (chosen.estimated) {
    reason += `; ${label(chosen.precision)} not measured for ${chosen.model}, figures from the nearest precision`;
  }
  return [chosen, reason];
}

/** What a browser offers, and how much memory each device has (null: unknown). */
export interface Hardware {
  /** Devices to try, in order. */
  devices: { device: Device; memoryMb: number | null }[];
}

/** The hardware a page can use, from the core's capabilities. */
export function browserHardware(options: {
  webgpu: boolean;
  wasmHeapMb: number;
  /** `navigator.deviceMemory` in GB, when the browser tells (capped at 8 by the API). */
  deviceMemoryGb?: number | null;
}): Hardware {
  const devices: Hardware["devices"] = [];
  if (options.webgpu) devices.push({ device: "webgpu", memoryMb: options.deviceMemoryGb ? options.deviceMemoryGb * 1024 : null });
  devices.push({ device: "wasm", memoryMb: options.wasmHeapMb });
  return { devices };
}

/**
 * The single precision for a model outside the tables: int4 on the CPU, float16 on a GPU (the fp16
 * variant: float16 encoder, float32 decoder).
 */
export function scalarFallback(device: Device): string {
  return device === "wasm" ? "int4" : "float16";
}

/**
 * A level's triple under memory, the explicit values held fixed. Null when nothing measured
 * matches.
 */
export function choose(
  level: Level,
  hardware: Hardware,
  measurements: Measurements,
  constraints: { language?: string | null; model?: string | null; precision?: HalfPrecision | null; noisy?: boolean } = {},
): { candidate: Candidate | null; reason: string; warnings: string[] } {
  const warnings: string[] = [];
  const models = constraints.model ? [constraints.model] : Object.keys(measurements);
  const first = hardware.devices[0]?.device;
  for (const { device, memoryMb } of hardware.devices) {
    const precisions = constraints.precision ? [constraints.precision] : ladder(device);
    const found = candidates(device, memoryMb, constraints.language, models, precisions, measurements);
    if (found.length > 0) {
      if (device !== first) warnings.push(`nothing measured usable fits on ${first}: falling back to ${device}`);
      const [candidate, reason] = pick(level, found, constraints.noisy ?? false);
      return { candidate, reason, warnings };
    }
  }
  return { candidate: null, reason: "", warnings };
}

/**
 * The rung a weak section is re-decoded at: the most accurate one measured on this device, when it
 * beats the current one by the `quality` budget and fits in the memory left now. A wider rung is
 * not a better one.
 */
export function retryRung(
  model: string,
  current: HalfPrecision,
  device: Device,
  language: string | null | undefined,
  memoryMb: number | null,
  measurements: Measurements,
): { encoder: string; decoder: string } | null {
  const measured = measurements[model];
  if (!measured) return null;
  const cells = werCells(measured, device, bucket(language));
  const currentKey = label(asRoles(current));
  if (!(currentKey in cells)) return null;
  const broken = new Set(measured.failed?.[device] ?? []);
  const target = cells[currentKey]! - WER_BUDGET_POINTS.quality;
  let best: { encoder: string; decoder: string } | null = null;
  for (const rung of ladder(device)) {
    const key = label(rung);
    if (broken.has(key) || !(key in cells) || cells[key]! > target) continue;
    if (!fits(model, rung, memoryMb, measurements)) continue;
    if (best === null || cells[key]! < cells[label(best)]!) best = { ...rung };
  }
  return best;
}

export interface Decision<T = unknown> {
  value: T;
  /** `"explicit"`, `"auto:<level>"` or `"session"`. */
  source: string;
  reason: string;
}

export interface Resolution {
  hardware: Hardware;
  decisions: {
    profile: Decision<Level>;
    model: Decision<string>;
    computeType: Decision<HalfPrecision>;
    backend: Decision<Device | "auto">;
    vad: Decision<"on" | "off">;
    decodingProfile: Decision<DecodingProfile>;
    language?: Decision<string>;
  };
  warnings: string[];
  audio: AudioProfile | null;
}

export interface ResolveOptions {
  model?: string;
  computeType?: HalfPrecision;
  backend?: Device | "auto";
  profile?: string;
  language?: string | null;
  /** The audio's measurements, when the audio is known. */
  audio?: AudioProfile | null;
  hardware: Hardware;
  measurements: Measurements;
}

/** Decide the settings BEFORE loading anything; the result can be read and passed on. */
export function resolveSettings(options: ResolveOptions): Resolution {
  const session = checkLevel(options.profile ?? DEFAULT_LEVEL);
  const model = options.model ?? "auto";
  const computeType = options.computeType ?? "auto";
  const noisy = options.audio !== undefined && options.audio !== null && options.audio.snrEstimateDb < NOISY_SNR_DB;
  const warnings: string[] = [];

  const modelLevel = levelFor(model, session);
  const precisionLevel = typeof computeType === "string" ? levelFor(computeType, session) : null;
  const pinnedModel = modelLevel === null ? model : null;
  const pinnedPrecision = precisionLevel === null ? computeType : null;
  const level = modelLevel ?? precisionLevel ?? session;

  const hardware: Hardware =
    options.backend && options.backend !== "auto"
      ? { devices: options.hardware.devices.filter((d) => d.device === options.backend) }
      : options.hardware;
  const chosen = choose(level, hardware, options.measurements, {
    language: options.language ?? null,
    model: pinnedModel,
    precision: pinnedPrecision,
    noisy,
  });
  warnings.push(...chosen.warnings);

  let name: string;
  let device: Device;
  let precision: HalfPrecision;
  let why: string;
  if (chosen.candidate === null) {
    const last = hardware.devices[hardware.devices.length - 1]?.device ?? "wasm";
    device = last;
    precision = pinnedPrecision ?? scalarFallback(last);
    why = "outside the measured tables";
    if (pinnedModel === null) {
      const measured = Object.keys(options.measurements).filter((m) => WHISPER_PARAMS_M[m] !== undefined);
      name = measured.reduce((a, b) => (WHISPER_PARAMS_M[a]! <= WHISPER_PARAMS_M[b]! ? a : b), measured[0] ?? "tiny");
      warnings.push(`no measured model fits; serving the smallest, ${name}`);
    } else {
      name = pinnedModel;
      if (options.measurements[name]) {
        const needed = modelMemoryMb(name, precision, options.measurements) ?? 0;
        warnings.push(`${name} fits nowhere (about ${needed.toFixed(0)} MB): kept, since it was requested explicitly`);
      }
    }
  } else {
    ({ model: name, device, precision } = chosen.candidate);
    why = chosen.reason;
  }

  const rungs = ladder(device);
  const roles = asRoles(precision);
  const rung = rungs.findIndex((r) => r.encoder === roles.encoder && r.decoder === roles.decoder);
  const large = shouldUseVad(WHISPER_PARAMS_M[name] ?? null, "auto");

  const decisions: Resolution["decisions"] = {
    profile: { value: session, source: "session", reason: "" },
    model: pinnedModel !== null
      ? { value: name, source: "explicit", reason: "" }
      : { value: name, source: `auto:${modelLevel}`, reason: why },
    computeType: pinnedPrecision !== null
      ? { value: pinnedPrecision, source: "explicit", reason: "" }
      : {
          value: precision,
          source: `auto:${precisionLevel}`,
          reason: rung >= 0 ? `${device}, rung ${rung + 1} of ${rungs.length} — ${why}` : why,
        },
    backend: options.backend && options.backend !== "auto"
      ? { value: options.backend, source: "explicit", reason: "" }
      : {
          value: device,
          source: `auto:${session}`,
          reason: device === hardware.devices[0]?.device ? `the first device that computes — ${device}` : `nothing measured fits on ${hardware.devices[0]?.device}`,
        },
    vad: {
      value: large ? "on" : "off",
      source: `auto:${session}`,
      reason: `${large ? "from" : "below"} ${LARGE_MODEL_PARAMS_M} M parameters, ${large ? "silences are removed before decoding" : "removing silences costs words"}`,
    },
    decodingProfile: { value: DECODING_ROLE[level], source: `auto:${level}`, reason: `the ${level} level's decoding` },
  };
  if (options.language) decisions.language = { value: options.language, source: "explicit", reason: "" };
  return { hardware, decisions, warnings, audio: options.audio ?? null };
}

/** The resolution as text, one decision per line. */
export function explain(resolution: Resolution): string {
  const lines = [
    `hardware: ${resolution.hardware.devices.map((d) => `${d.device}${d.memoryMb ? ` ${d.memoryMb.toFixed(0)} MB` : ""}`).join(", ")}`,
  ];
  for (const [name, d] of Object.entries(resolution.decisions)) {
    lines.push(`${name} = ${JSON.stringify(d.value)}  [${d.source}]${d.reason ? ` — ${d.reason}` : ""}`);
  }
  lines.push(...resolution.warnings.map((w) => `warning: ${w}`));
  return lines.join("\n");
}
