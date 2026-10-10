/** Silero voice-activity detection. */

import {
  VADError,
  huggingFaceResolver,
  loadModel,
  type LoadOptions,
  type ModelBundle,
  type ModelSession,
} from "@youni/onnxweb-core";
import type { TensorFactory } from "@youni/onnxweb-core/core";

export const SILERO_REPO = "onnx-community/silero-vad";

export interface VADConfig {
  threshold: number;
  sampleRate: 8000 | 16000;
  windowSizeSamples: number;
  minSpeechDurationMs: number;
  minSilenceDurationMs: number;
  speechPadMs: number;
  maxSpeechDurationS: number;
}

/** Silero's defaults, as `ynonnx_core.types.configs.VADConfig`. */
export const DEFAULT_VAD: VADConfig = {
  threshold: 0.5,
  sampleRate: 16000,
  windowSizeSamples: 512,
  minSpeechDurationMs: 250,
  minSilenceDurationMs: 100,
  speechPadMs: 100,
  maxSpeechDurationS: Number.POSITIVE_INFINITY,
};

export interface SpeechSegment {
  start: number;
  end: number;
}

const WINDOWS: Readonly<Record<number, number>> = { 8000: 256, 16000: 512 };
const CONTEXT: Readonly<Record<number, number>> = { 8000: 32, 16000: 64 };

/**
 * A probability curve turned into padded speech spans, Python's rules in Python's order: a window
 * is speech at or above the threshold; a span survives a gap shorter than the minimum silence but
 * ends at its last speech window; spans are padded and clamped; spans shorter than the minimum
 * speech are dropped; spans longer than the maximum are split; spans that STRICTLY overlap are
 * merged (touching ones are not, so splits stay split).
 */
export function segmentsFromProbabilities(
  probabilities: ArrayLike<number>,
  samples: number,
  config: VADConfig = DEFAULT_VAD,
  threshold = config.threshold,
): SpeechSegment[] {
  if (probabilities.length === 0) return [];
  const windowS = config.windowSizeSamples / config.sampleRate;
  const totalS = samples / config.sampleRate;
  const minSilence = Math.max(1, Math.round(config.minSilenceDurationMs / 1000 / windowS));
  const minSpeechS = config.minSpeechDurationMs / 1000;
  const padS = config.speechPadMs / 1000;

  const raw: [number, number][] = [];
  let start: number | null = null;
  let lastSpeech = -1;
  for (let i = 0; i < probabilities.length; i += 1) {
    if (probabilities[i]! >= threshold) {
      start ??= i;
      lastSpeech = i;
    } else if (start !== null && i - lastSpeech >= minSilence) {
      raw.push([start, lastSpeech]);
      start = null;
    }
  }
  if (start !== null) raw.push([start, lastSpeech]);

  const spans: [number, number][] = [];
  for (const [first, last] of raw) {
    let begin = Math.max(0, first * windowS - padS);
    const end = Math.min(totalS, (last + 1) * windowS + padS);
    if (end - begin < minSpeechS) continue;
    while (Number.isFinite(config.maxSpeechDurationS) && end - begin > config.maxSpeechDurationS) {
      spans.push([begin, begin + config.maxSpeechDurationS]);
      begin += config.maxSpeechDurationS;
    }
    spans.push([begin, end]);
  }

  const merged: [number, number][] = [];
  for (const [begin, end] of spans) {
    const previous = merged[merged.length - 1];
    if (previous && begin < previous[1]) previous[1] = Math.max(previous[1], end);
    else merged.push([begin, end]);
  }
  return merged.map(([s, e]) => ({ start: s, end: e }));
}

export interface VADLoadOptions extends Omit<LoadOptions, "model"> {
  Tensor: TensorFactory;
  config?: Partial<VADConfig>;
}

export class VAD {
  private constructor(
    readonly bundle: ModelBundle,
    private readonly session: ModelSession,
    private readonly Tensor: TensorFactory,
    readonly config: VADConfig,
  ) {}

  /** Load Silero (from the Hub unless a resolver says otherwise), on the CPU by default. */
  static async load(options: VADLoadOptions): Promise<VAD> {
    const config = { ...DEFAULT_VAD, ...options.config };
    if (WINDOWS[config.sampleRate] === undefined) {
      throw new VADError(`Silero VAD supports 8000 and 16000 Hz, not ${config.sampleRate}: resample first`);
    }
    if (config.windowSizeSamples !== WINDOWS[config.sampleRate]) {
      throw new VADError(
        `windowSizeSamples=${config.windowSizeSamples} is not what the model was exported for at ` +
          `${config.sampleRate} Hz (${WINDOWS[config.sampleRate]})`,
      );
    }
    const bundle = await loadModel({
      backends: "wasm",
      resolver: huggingFaceResolver({ repo: SILERO_REPO, subfolder: "onnx", manifest: false }),
      ...options,
      model: { name: SILERO_REPO, computeType: "float32" },
    });
    return new VAD(bundle, bundle.only, options.Tensor, config);
  }

  /** One speech probability per window, before any threshold: the raw curve. */
  async speechProbabilities(audio: Float32Array): Promise<Float32Array> {
    const window = this.config.windowSizeSamples;
    const contextSize = CONTEXT[this.config.sampleRate]!;
    const out = new Float32Array(Math.ceil(audio.length / window));
    let state: unknown = new this.Tensor("float32", new Float32Array(2 * 128), [2, 1, 128]);
    const rate = new this.Tensor("int64", BigInt64Array.of(BigInt(this.config.sampleRate)), []);
    let context = new Float32Array(contextSize);
    try {
      for (let start = 0, i = 0; start < audio.length; start += window, i += 1) {
        const chunk = new Float32Array(window);
        chunk.set(audio.subarray(start, Math.min(start + window, audio.length)));
        const fed = new Float32Array(contextSize + window);
        fed.set(context);
        fed.set(chunk, contextSize);
        const result = await this.session.run({
          input: new this.Tensor("float32", fed, [1, contextSize + window]),
          state,
          sr: rate,
        });
        out[i] = ((result["output"] as { data: Float32Array }).data)[0]!;
        state = result["stateN"];
        context = chunk.slice(window - contextSize);
      }
    } catch (cause) {
      if (cause instanceof VADError) throw cause;
      throw new VADError(`Silero VAD inference failed: ${cause instanceof Error ? cause.message : String(cause)}`, { cause });
    }
    return out;
  }

  /** Speech intervals, in seconds. */
  async detectSpeech(audio: Float32Array, threshold?: number): Promise<SpeechSegment[]> {
    const probabilities = await this.speechProbabilities(audio);
    return segmentsFromProbabilities(probabilities, audio.length, this.config, threshold ?? this.config.threshold);
  }

  async close(): Promise<void> {
    await this.bundle.close();
  }
}
