/**
 * Emotion, sound events and music (spec § 2.2): audio classifiers whose answers become timeline
 * annotations on the original audio.
 */

import { huggingFaceResolver, loadModel, type LoadOptions, type ModelBundle } from "@youni/onnxweb-core";
import { ConfigurationError, type TensorFactory } from "@youni/onnxweb-core/core";

import { fbank } from "../diarization/fbank.js";

export const DEFAULT_EMOTION_MODEL = "onnx-community/wav2vec2-emotion-recognition-ONNX";
export const DEFAULT_EVENTS_MODEL = "onnx-community/ast-finetuned-audioset-10-10-0.4593-ONNX";

/** The canonical emotions (timeline schema 1.1), and how model labels map onto them. */
export const EMOTIONS = ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"] as const;
export type Emotion = (typeof EMOTIONS)[number];
const EMOTION_ALIASES: Readonly<Record<string, Emotion>> = {
  neu: "neutral", neutral: "neutral",
  cal: "calm", calm: "calm",
  hap: "happy", happy: "happy", joy: "happy",
  sad: "sad", sadness: "sad",
  ang: "angry", angry: "angry", anger: "angry",
  fea: "fearful", fear: "fearful", fearful: "fearful",
  dis: "disgust", disgust: "disgust",
  sur: "surprised", surprise: "surprised", surprised: "surprised", ps: "surprised",
};

/** A model label as a canonical emotion, or null when it maps onto none. */
export function canonicalEmotion(label: string): Emotion | null {
  return EMOTION_ALIASES[label.trim().toLowerCase()] ?? null;
}

/**
 * AudioSet classes that are not events: speech itself (0–7), the music family (137–282), silence
 * and the room (500, 506–512).
 */
export function audioSetKind(index: number): "speech" | "music" | "ambience" | "event" {
  if (index <= 7) return "speech";
  if (index >= 137 && index <= 282) return "music";
  if (index === 500 || (index >= 506 && index <= 512)) return "ambience";
  return "event";
}

/** A canonical event label: the AudioSet name, lowercase. */
export function canonicalEvent(name: string): string {
  return name.trim().toLowerCase();
}

export interface Detection {
  start: number;
  end: number;
  label: string;
  sourceLabel: string;
  confidence: number;
  speakerId: string | null;
}

interface Loaded {
  bundle: ModelBundle;
  labels: string[];
  config: Record<string, unknown>;
}

async function load(repo: string, options: ClassifierOptions): Promise<Loaded> {
  const { Tensor: _t, computeType, endpoint, ...rest } = options;
  const bundle = await loadModel({
    extraFiles: ["config.json", "preprocessor_config.json"],
    ...rest,
    resolver: huggingFaceResolver({ repo, subfolder: "onnx", manifest: false, ...(endpoint ? { endpoint } : {}) }),
    model: { name: repo, computeType: computeType ?? "int8" },
  });
  const config = JSON.parse(bundle.assetText("config.json") ?? "{}") as { id2label?: Record<string, string> };
  const ids = Object.keys(config.id2label ?? {}).sort((a, b) => Number(a) - Number(b));
  if (ids.length === 0) {
    await bundle.close();
    throw new ConfigurationError(`${repo} names no labels (config.json id2label)`);
  }
  const pre = JSON.parse(bundle.assetText("preprocessor_config.json") ?? "{}") as Record<string, unknown>;
  return { bundle, labels: ids.map((id) => config.id2label![id]!), config: pre };
}

export interface ClassifierOptions extends Omit<LoadOptions, "model" | "computeType" | "resolver"> {
  Tensor: TensorFactory;
  /** `int8` by default (the graphs are small; WASM runs them well). */
  computeType?: string;
  endpoint?: string;
}

function softmax(values: ArrayLike<number>): Float32Array {
  let max = -Infinity;
  for (let i = 0; i < values.length; i += 1) max = Math.max(max, values[i]!);
  const out = new Float32Array(values.length);
  let sum = 0;
  for (let i = 0; i < values.length; i += 1) sum += out[i] = Math.exp(values[i]! - max);
  for (let i = 0; i < out.length; i += 1) out[i] = out[i]! / sum;
  return out;
}

const sigmoid = (x: number) => 1 / (1 + Math.exp(-x));
const round3 = (x: number) => Math.round(x * 1000) / 1000;

/** Emotion of speech regions, with a wav2vec2 classifier. */
export class EmotionClassifier {
  /** Seconds per chunk; a region longer than that is averaged over its chunks. */
  static readonly CHUNK_S = 8;
  static readonly MIN_S = 1;

  private constructor(
    readonly model: string,
    readonly loaded: Loaded,
    readonly Tensor: TensorFactory,
  ) {}

  static async load(options: ClassifierOptions & { model?: string }): Promise<EmotionClassifier> {
    const model = options.model ?? DEFAULT_EMOTION_MODEL;
    return new EmotionClassifier(model, await load(model, options), options.Tensor);
  }

  /** Probabilities over the model's labels for one chunk of 16 kHz audio. */
  async probabilities(chunk: Float32Array): Promise<Float32Array> {
    let input = chunk;
    if (this.loaded.config["do_normalize"] !== false) {
      let mean = 0;
      for (const v of chunk) mean += v;
      mean /= chunk.length || 1;
      let variance = 0;
      for (const v of chunk) variance += (v - mean) ** 2;
      const scale = 1 / Math.sqrt(variance / (chunk.length || 1) + 1e-7);
      input = Float32Array.from(chunk, (v) => (v - mean) * scale);
    }
    const session = this.loaded.bundle.only;
    const name = session.inputs[0]?.name ?? "input_values";
    const out = (await session.run({ [name]: new this.Tensor("float32", input, [1, input.length]) })) as Record<string, { data: Float32Array }>;
    const logits = out[session.outputs[0]!.name]!.data;
    return softmax(logits);
  }

  /** One emotion per region: the mean of its chunks' probabilities. */
  async classify(audio: Float32Array, regions: readonly { start: number; end: number; speakerId?: string | null }[]): Promise<Detection[]> {
    const out: Detection[] = [];
    for (const region of regions) {
      if (region.end - region.start < EmotionClassifier.MIN_S) continue;
      const sum = new Float32Array(this.loaded.labels.length);
      let chunks = 0;
      for (let t = region.start; t < region.end - EmotionClassifier.MIN_S / 2; t += EmotionClassifier.CHUNK_S) {
        const end = Math.min(region.end, t + EmotionClassifier.CHUNK_S);
        if (end - t < EmotionClassifier.MIN_S && chunks > 0) break;
        const p = await this.probabilities(audio.subarray(Math.round(t * 16000), Math.round(end * 16000)));
        p.forEach((v, i) => (sum[i] = sum[i]! + v));
        chunks += 1;
      }
      if (chunks === 0) continue;
      let best = 0;
      for (let i = 1; i < sum.length; i += 1) if (sum[i]! > sum[best]!) best = i;
      const sourceLabel = this.loaded.labels[best]!;
      out.push({
        start: round3(region.start),
        end: round3(region.end),
        label: canonicalEmotion(sourceLabel) ?? sourceLabel.toLowerCase(),
        sourceLabel,
        confidence: round3(sum[best]! / chunks),
        speakerId: region.speakerId ?? null,
      });
    }
    return out;
  }

  async close(): Promise<void> {
    await this.loaded.bundle.close();
  }
}

/** Sound events and music, with AST on AudioSet. */
export class AudioSetTagger {
  /** The window AST reads: 1024 frames of 10 ms. */
  static readonly WINDOW_S = 10.24;

  private constructor(
    readonly model: string,
    readonly loaded: Loaded,
    readonly Tensor: TensorFactory,
  ) {}

  static async load(options: ClassifierOptions & { model?: string }): Promise<AudioSetTagger> {
    const model = options.model ?? DEFAULT_EVENTS_MODEL;
    return new AudioSetTagger(model, await load(model, options), options.Tensor);
  }

  /**
   * AST's input for one window: Kaldi fbank (Hann window, 128 bands), 1024 frames, normalized as
   * the extractor does.
   */
  features(window: Float32Array): Float32Array {
    const c = this.loaded.config;
    const bins = Number(c["num_mel_bins"] ?? 128);
    const frames = Number(c["max_length"] ?? 1024);
    const mean = Number(c["mean"] ?? -4.2677393);
    const std = Number(c["std"] ?? 4.5689974);
    const f = fbank(window, 16000, bins, false, "hanning");
    const out = new Float32Array(frames * bins);
    out.set(f.data.subarray(0, Math.min(f.data.length, out.length)));
    for (let i = 0; i < out.length; i += 1) out[i] = (out[i]! - mean) / (2 * std);
    return out;
  }

  /**
   * Each class's probability (sigmoid: AudioSet is multi-label) for the window starting at `start`.
   */
  async scores(audio: Float32Array, start: number): Promise<Float32Array> {
    const from = Math.round(start * 16000);
    const window = audio.subarray(from, Math.min(audio.length, from + Math.round(AudioSetTagger.WINDOW_S * 16000)));
    const c = this.loaded.config;
    const bins = Number(c["num_mel_bins"] ?? 128);
    const frames = Number(c["max_length"] ?? 1024);
    const session = this.loaded.bundle.only;
    const name = session.inputs[0]?.name ?? "input_values";
    const out = (await session.run({ [name]: new this.Tensor("float32", this.features(window), [1, frames, bins]) })) as Record<string, { data: Float32Array }>;
    return Float32Array.from(out[session.outputs[0]!.name]!.data, sigmoid);
  }

  /**
   * Events and music over the audio: one window every `interval` seconds; a class above `threshold`
   * (music: `musicThreshold` on the "Music" class) becomes a detection, neighbouring windows of one
   * class merged.
   */
  async tag(audio: Float32Array, options: { interval?: number; threshold?: number; musicThreshold?: number; want?: ReadonlySet<"events" | "music"> } = {}): Promise<{ events: Detection[]; music: Detection[] }> {
    const duration = audio.length / 16000;
    const interval = options.interval ?? 5;
    const threshold = options.threshold ?? 0.3;
    const musicThreshold = options.musicThreshold ?? 0.4;
    const want = options.want ?? new Set(["events", "music"] as const);
    const events = new Map<string, Detection[]>();
    const music: Detection[] = [];
    const musicIndex = this.loaded.labels.indexOf("Music");
    const push = (list: Detection[], start: number, end: number, label: string, sourceLabel: string, confidence: number) => {
      const last = list[list.length - 1];
      if (last && start <= last.end + 1e-3) {
        last.end = Math.max(last.end, end);
        last.confidence = Math.max(last.confidence, confidence);
      } else list.push({ start: round3(start), end: round3(end), label, sourceLabel, confidence: round3(confidence), speakerId: null });
    };
    for (let start = 0; start < Math.max(duration - 0.5, 0.001); start += interval) {
      const end = Math.min(duration, start + AudioSetTagger.WINDOW_S);
      const p = await this.scores(audio, start);
      if (want.has("music") && musicIndex >= 0 && p[musicIndex]! >= musicThreshold) push(music, start, end, "music", "Music", p[musicIndex]!);
      if (want.has("events")) {
        p.forEach((score, index) => {
          if (score < threshold || audioSetKind(index) !== "event") return;
          const sourceLabel = this.loaded.labels[index]!;
          const label = canonicalEvent(sourceLabel);
          if (!events.has(label)) events.set(label, []);
          push(events.get(label)!, start, end, label, sourceLabel, score);
        });
      }
      if (end >= duration) break;
    }
    for (const list of events.values()) for (const d of list) d.end = round3(d.end);
    return { events: [...events.values()].flat().sort((a, b) => a.start - b.start || a.label.localeCompare(b.label)), music };
  }

  async close(): Promise<void> {
    await this.loaded.bundle.close();
  }
}
