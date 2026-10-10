/** Forced alignment against a CTC acoustic model. */

import { huggingFaceResolver, loadModel, type LoadOptions, type ModelBundle, type ModelSession } from "@youni/onnxweb-core";
import type { TensorFactory } from "@youni/onnxweb-core/core";

import { AlignmentError } from "../errors.js";
import type { Word } from "./punctuation.js";

/** CTC models run at 50 frames per second (20 ms stride) at 16 kHz. */
export const FRAMES_PER_SECOND = 50;

/** wav2vec2-base-960h: 94 M parameters, characters, English. */
export const DEFAULT_CTC_MODEL = "onnx-community/wav2vec2-base-960h-ONNX";

const NEG_INF = -1e30;

/**
 * The most likely CTC path spelling `targets`: one frame index per target (the first frame its
 * state is occupied). The target is interleaved with blanks; the path may stay, advance one, or —
 * between DIFFERENT symbols — advance two, which forbids collapsing a doubled letter.
 */
export function viterbiAlign(emission: Float32Array | Float64Array, frames: number, vocab: number, targets: readonly number[], blank = 0): number[] {
  if (targets.length === 0) return [];
  const extended = [blank];
  for (const target of targets) extended.push(target, blank);
  const length = extended.length;
  if (frames < Math.floor(length / 2)) {
    throw new AlignmentError(`${frames} frames cannot spell ${targets.length} symbols; the audio is shorter than the transcript`);
  }
  let previous = new Float64Array(length).fill(NEG_INF);
  let current = new Float64Array(length);
  const back = new Uint8Array(frames * length);
  previous[0] = emission[extended[0]!]!;
  if (length > 1) previous[1] = emission[extended[1]!]!;
  for (let frame = 1; frame < frames; frame += 1) {
    const row = frame * vocab;
    for (let state = 0; state < length; state += 1) {
      let best = previous[state]!;
      let move = 0;
      if (state > 0 && previous[state - 1]! > best) {
        best = previous[state - 1]!;
        move = 1;
      }
      if (state > 1 && extended[state] !== blank && extended[state] !== extended[state - 2] && previous[state - 2]! > best) {
        best = previous[state - 2]!;
        move = 2;
      }
      current[state] = best + emission[row + extended[state]!]!;
      back[frame * length + state] = move;
    }
    [previous, current] = [current, previous];
  }
  let state = length - 1;
  if (length > 1 && previous[length - 2]! > previous[length - 1]!) state = length - 2;
  const path = new Int32Array(frames);
  for (let frame = frames - 1; frame >= 0; frame -= 1) {
    path[frame] = state;
    state -= back[frame * length + state]!;
  }
  const out: number[] = [];
  for (let index = 0; index < targets.length; index += 1) {
    const hit = path.indexOf(2 * index + 1);
    out.push(hit >= 0 ? hit : 0);
  }
  return out;
}

/** Log-softmax over the last axis, in place on a copy. */
export function logSoftmaxRows(logits: Float32Array, frames: number, vocab: number): Float64Array {
  const out = new Float64Array(frames * vocab);
  for (let f = 0; f < frames; f += 1) {
    let max = Number.NEGATIVE_INFINITY;
    for (let v = 0; v < vocab; v += 1) max = Math.max(max, logits[f * vocab + v]!);
    let sum = 0;
    for (let v = 0; v < vocab; v += 1) sum += Math.exp(logits[f * vocab + v]! - max);
    const log = Math.log(sum);
    for (let v = 0; v < vocab; v += 1) out[f * vocab + v] = logits[f * vocab + v]! - max - log;
  }
  return out;
}

/**
 * Zero mean, unit variance: what Wav2Vec2FeatureExtractor does when its config says `do_normalize`.
 * (ynonnx_transcribe feeds the raw signal; the model was trained on normalized input.)
 */
export function normalizeSignal(audio: Float32Array): Float32Array {
  let mean = 0;
  for (const x of audio) mean += x;
  mean /= Math.max(audio.length, 1);
  let variance = 0;
  for (const x of audio) variance += (x - mean) ** 2;
  variance /= Math.max(audio.length, 1);
  const scale = 1 / Math.sqrt(variance + 1e-7);
  return audio.map((x) => (x - mean) * scale);
}

export interface CTCVocabulary {
  symbols: Record<string, number>;
  blank: number;
  wordDelimiter: number;
}

/**
 * The vocabulary as the aligner reads it: upper-cased keys, the blank and the word delimiter found.
 */
export function ctcVocabulary(vocab: Record<string, number>): CTCVocabulary {
  const symbols = Object.fromEntries(Object.entries(vocab).map(([k, v]) => [k.toUpperCase(), v]));
  return {
    symbols,
    blank: symbols["<PAD>"] ?? symbols["<BLANK>"] ?? 0,
    wordDelimiter: symbols["|"] ?? symbols[" "] ?? -1,
  };
}

/** Accents and punctuation the CTC vocabulary cannot spell, removed; upper case. */
export function normalizeForCtc(text: string): string {
  const stripped = text.normalize("NFKD").replace(/\p{M}/gu, "");
  return stripped.replace(/[^\p{L}\p{N}_\s']/gu, " ").toUpperCase();
}

/** Symbol ids, and one [start, end) span per word. */
export function toTargets(text: string, vocabulary: CTCVocabulary): { targets: number[]; spans: [number, number][] } {
  const targets: number[] = [];
  const spans: [number, number][] = [];
  for (const word of normalizeForCtc(text).split(/\s+/).filter(Boolean)) {
    const start = targets.length;
    for (const char of word) {
      const symbol = vocabulary.symbols[char];
      if (symbol !== undefined) targets.push(symbol);
    }
    if (targets.length > start) {
      spans.push([start, targets.length]);
      if (vocabulary.wordDelimiter >= 0) targets.push(vocabulary.wordDelimiter);
    }
  }
  return { targets, spans };
}

/** Word spans from the frames Viterbi chose, with the ORIGINAL words. */
export function wordsFromFrames(text: string, spans: readonly [number, number][], frames: readonly number[], offset = 0): Word[] {
  const original = text.split(/\s+/).filter(Boolean);
  const normalized = normalizeForCtc(text).split(/\s+/).filter(Boolean);
  const round3 = (x: number) => Math.round(x * 1000) / 1000;
  return spans.slice(0, normalized.length).map(([start, end], index) => {
    const first = frames[start]!;
    const last = frames[Math.min(end, frames.length) - 1]!;
    return {
      word: original[index] ?? normalized[index]!,
      start: round3(offset + first / FRAMES_PER_SECOND),
      end: round3(offset + (last + 1) / FRAMES_PER_SECOND),
      probability: 0,
    };
  });
}

export interface ForcedAlignerOptions extends Omit<LoadOptions, "model"> {
  Tensor: TensorFactory;
  /** A Hub repository publishing an ONNX CTC model and its `vocab.json`. */
  model?: string;
}

export class ForcedAligner {
  private constructor(
    readonly bundle: ModelBundle,
    private readonly session: ModelSession,
    private readonly Tensor: TensorFactory,
    readonly vocabulary: CTCVocabulary,
    /** Zero mean, unit variance on the input, as the feature extractor declares (`do_normalize`). */
    readonly normalize: boolean,
  ) {}

  /**
   * Load a CTC model and its vocabulary. The vocabulary is not optional: CTC emits indices into one
   * symbol table, and aligning against another gives a path that is monotonic, plausible, and wrong
   * everywhere.
   */
  static async load(options: ForcedAlignerOptions): Promise<ForcedAligner> {
    const repo = options.model ?? DEFAULT_CTC_MODEL;
    const { Tensor, model: _model, ...rest } = options;

    const bundle = await loadModel({
      backends: "wasm",
      resolver: huggingFaceResolver({ repo, subfolder: "onnx", manifest: false }),
      extraFiles: ["vocab.json", "preprocessor_config.json"],
      ...rest,
      model: { name: repo, computeType: "int8" },
    });
    const vocab = bundle.assetText("vocab.json");
    if (vocab === null) throw new AlignmentError(`${repo} publishes no vocab.json; a CTC model needs its symbol table`);
    const preprocessor = bundle.assetText("preprocessor_config.json");
    const normalize = preprocessor !== null && (JSON.parse(preprocessor) as { do_normalize?: boolean }).do_normalize === true;
    return new ForcedAligner(bundle, bundle.only, Tensor, ctcVocabulary(JSON.parse(vocab) as Record<string, number>), normalize);
  }

  /** Word spans for `text` over `audio` (mono 16 kHz), shifted by `offset` seconds. */
  async align(audio: Float32Array, text: string, offset = 0): Promise<Word[]> {
    const { targets, spans } = toTargets(text, this.vocabulary);
    if (targets.length === 0) return [];
    const input = this.session.inputs[0]?.name ?? "input_values";
    const signal = this.normalize ? normalizeSignal(audio) : audio;
    const outputs = await this.session.run({ [input]: new this.Tensor("float32", signal, [1, signal.length]) });
    const first = Object.values(outputs)[0] as { data: Float32Array; dims: readonly number[] };
    const dims = first.dims;
    const vocab = dims[dims.length - 1]!;
    const frames = dims[dims.length - 2]!;
    const emission = logSoftmaxRows(first.data, frames, vocab);
    const chosen = viterbiAlign(emission, frames, vocab, targets, this.vocabulary.blank);
    return wordsFromFrames(text, spans, chosen, offset);
  }

  async close(): Promise<void> {
    await this.bundle.close();
  }
}
