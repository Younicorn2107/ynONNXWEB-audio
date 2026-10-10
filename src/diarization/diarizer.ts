/** Who spoke when. */

import { huggingFaceResolver, loadModel, type LoadOptions, type ModelBundle, type ModelSession } from "@youni/onnxweb-core";
import type { TensorFactory } from "@youni/onnxweb-core/core";

import { DiarizationError, EnrollmentError } from "../errors.js";
import { agglomerative, relabelByFirstAppearance, spectral } from "./clustering.js";
import { fbank } from "./fbank.js";

export const DEFAULT_EMBEDDING_MODEL = "onnx-community/wespeaker-voxceleb-resnet34-LM";
export const DEFAULT_SEGMENTATION_MODEL = "onnx-community/pyannote-segmentation-3.0";

/** Chunks shorter than this carry too little voice to place a speaker. */
export const MIN_CHUNK_SECONDS = 0.5;
/** Chunk length and stride when slicing speech spans without a segmentation model. */
export const CHUNK_SECONDS = 1.5;
export const CHUNK_STRIDE = 0.75;
/** The segmentation model's "no speaker" class (powerset class 0). */
export const SILENCE_CLASS = 0;
/**
 * A speaker's pause shorter than this does not end the turn: sentences are separated by about 0.8
 * s.
 */
export const MERGE_GAP_SECONDS = 1.0;
/** The input length pyannote-segmentation-3.0 was trained on. */
export const SEGMENTATION_WINDOW_S = 10;

export interface SpeakerTurn {
  /** The stable identity in this result: `SPEAKER_00`, `SPEAKER_01`… by first appearance. */
  speaker: string;
  start: number;
  end: number;
  /** The enrolled name, when the voice matched a print; the identity stays `speaker`. */
  name?: string | null;
}

/** What is known of each speaker of a result (spec § 6.1): an enrolled name is not an identity. */
export interface SpeakerInfo {
  speakerId: string;
  name: string | null;
  /** Cosine similarity to the enrolled print that named it, or null. */
  similarity: number | null;
}

/** The speaker info of a set of turns, in order of first appearance. */
export function speakerInfoOf(turns: readonly SpeakerTurn[], similarities: ReadonlyMap<string, number> = new Map()): SpeakerInfo[] {
  const seen = new Map<string, SpeakerInfo>();
  for (const turn of turns) {
    if (!seen.has(turn.speaker)) seen.set(turn.speaker, { speakerId: turn.speaker, name: turn.name ?? null, similarity: similarities.get(turn.speaker) ?? null });
  }
  return [...seen.values()];
}

/** How a turn is shown: its enrolled name, else "Speaker N" (N = rank + 1). */
export function displayName(speakerId: string, name?: string | null): string {
  if (name) return name;
  const match = /^SPEAKER_(\d+)$/.exec(speakerId);
  return match ? `Speaker ${Number(match[1]) + 1}` : speakerId;
}

/**
 * Cosine similarity from which a cluster is taken for an enrolled speaker. Measured on
 * wespeaker-voxceleb-resnet34-LM (see the speakers system test).
 */
export const ENROLLED_MATCH_THRESHOLD = 0.5;

/**
 * Where agglomerative clustering stops: 0.5 average cosine similarity. The prints here are of short
 * overlapping windows (`chunkSpeech`), noisier than the long speaker-change chunks the Python side
 * embeds with 0.7: at 0.7 one voice came out as 22 speakers (a 70 s talk; Python: 2) and a
 * three-voice dialogue as 12; at 0.5, 2 and 4, every turn of the dialogue still pure.
 */
export const CLUSTER_THRESHOLD = 0.5;

export interface DiarizationConfig {
  /**
   * Enrolled speakers' prints (L2-normalized), by name: a cluster close enough to one is named
   * after it.
   */
  knownSpeakers?: Readonly<Record<string, ArrayLike<number>>>;
  /** The similarity a cluster needs to an enrolled print to take its name. */
  matchThreshold?: number;
  /**
   * A Hub repository, `"auto"` (the default), or `""` to slice speech spans instead (segmentation
   * only).
   */
  segmentationModel?: string;
  embeddingModel?: string;
  clustering?: "agglomerative" | "spectral";
  /** Agglomerative clustering: merge while the average similarity stays above this (0.5). */
  threshold?: number;
  /**
   * With enrolled speakers: `chunk` names each chunk after the closest print above `matchThreshold`
   * and clusters only the rest (two similar voices are not merged first); `cluster` clusters
   * everything, then names clusters.
   */
  enrolledAssignment?: "chunk" | "cluster";
  /** Pauses of one speaker shorter than this stay inside the turn (seconds). */
  mergeGap?: number;
  /** Seconds per embedded chunk, and between two chunk starts. */
  chunkSeconds?: number;
  chunkStride?: number;
  numSpeakers?: number | null;
  minSpeakers?: number | null;
  maxSpeakers?: number | null;
  assignmentStrategy?: "longest" | "first";
}

const round3 = (x: number) => Math.round(x * 1000) / 1000;

/** Runs of one owner, as spans: `step` seconds per frame. */
export function runsToSpans(owners: ArrayLike<number>, step: number): [number, number][] {
  const spans: [number, number][] = [];
  if (owners.length === 0) return spans;
  let start = 0;
  for (let index = 1; index <= owners.length; index += 1) {
    if (index === owners.length || owners[index] !== owners[start]) {
      spans.push([round3(start * step), round3(index * step)]);
      start = index;
    }
  }
  return spans;
}

/**
 * Speech spans sliced into overlapping chunks, so a turn change inside one is also seen cleanly by
 * a neighbour.
 */
export function chunkSpeech(spans: readonly [number, number][], chunkS = CHUNK_SECONDS, strideS = CHUNK_STRIDE, minS = chunkS * 0.5): [number, number][] {
  const out: [number, number][] = [];
  for (const [start, end] of spans) {
    let cursor = start;
    while (cursor < end) {
      const stop = Math.min(cursor + chunkS, end);
      if (stop - cursor >= minS) out.push([round3(cursor), round3(stop)]);
      if (stop >= end) break;
      cursor += strideS;
    }
  }
  return out;
}

/**
 * (span, label) pairs merged into ordered turns: consecutive spans of one speaker within half a
 * second join.
 */
export function toTurns(spans: readonly [number, number][], labels: readonly number[], names: ReadonlyMap<number, string> = new Map(), mergeGap = 0.5): SpeakerTurn[] {
  const paired = spans.map((span, i) => [span, labels[i]!] as const).sort((a, b) => a[0][0] - b[0][0]);
  const turns: SpeakerTurn[] = [];
  for (const [[start, end], label] of paired) {
    const speaker = names.get(label) ?? `SPEAKER_${String(label).padStart(2, "0")}`;
    const last = turns[turns.length - 1];
    if (last && last.speaker === speaker && start - last.end <= mergeGap) last.end = Math.max(last.end, end);
    else turns.push({ speaker, start, end });
  }
  return turns;
}

export function overlapSeconds(a: readonly [number, number], b: readonly [number, number]): number {
  return Math.max(0, Math.min(a[1], b[1]) - Math.max(a[0], b[0]));
}

/**
 * Each segment labelled by whoever spoke most of it (`longest`) or started it (`first`). A segment
 * overlapping no turn keeps no speaker: visibly unlabelled beats wrongly labelled.
 */
export function assignSpeakers<S extends { start: number; end: number; speaker?: string | null }>(
  segments: readonly S[],
  turns: readonly SpeakerTurn[],
  strategy: "longest" | "first" = "longest",
): S[] {
  if (turns.length === 0) return [...segments];
  return segments.map((segment) => {
    const candidates = turns
      .map((turn) => [turn, overlapSeconds([segment.start, segment.end], [turn.start, turn.end])] as const)
      .filter(([, overlap]) => overlap > 0);
    if (candidates.length === 0) return segment;
    const chosen =
      strategy === "first"
        ? candidates.reduce((a, b) => (b[0].start < a[0].start ? b : a))[0]
        : candidates.reduce((a, b) => (b[1] > a[1] ? b : a))[0];
    return { ...segment, speaker: chosen.speaker };
  });
}

function normalize(vector: ArrayLike<number>): Float32Array {
  const out = Float32Array.from(vector);
  const norm = Math.max(Math.sqrt(out.reduce((s, x) => s + x * x, 0)), 1e-12);
  return out.map((x) => x / norm);
}

const dot = (a: ArrayLike<number>, b: ArrayLike<number>) => {
  let sum = 0;
  for (let k = 0; k < a.length; k += 1) sum += a[k]! * b[k]!;
  return sum;
};

/**
 * Names for clusters, from enrolled prints: each cluster's centroid against each print, the best
 * pairs taken first, one name per cluster and one cluster per name, only above the threshold. A
 * cluster left without a name keeps its SPEAKER_xx: an unknown voice stays visibly unknown.
 */
export function matchEnrolled(
  embeddings: readonly Float32Array[],
  labels: readonly number[],
  known: Readonly<Record<string, ArrayLike<number>>>,
  threshold = ENROLLED_MATCH_THRESHOLD,
): Map<number, string> {
  const centroids = new Map<number, Float32Array>();
  labels.forEach((label, i) => {
    const sum = centroids.get(label) ?? new Float32Array(embeddings[i]!.length);
    embeddings[i]!.forEach((x, k) => (sum[k] = sum[k]! + x));
    centroids.set(label, sum);
  });
  const pairs: [number, string, number][] = [];
  for (const [label, sum] of centroids) {
    const centroid = normalize(sum);
    for (const [name, print] of Object.entries(known)) pairs.push([label, name, dot(centroid, normalize(print))]);
  }
  pairs.sort((a, b) => b[2] - a[2]);
  const names = new Map<number, string>();
  const used = new Set<string>();
  for (const [label, name, similarity] of pairs) {
    if (similarity < threshold) break;
    if (names.has(label) || used.has(name)) continue;
    names.set(label, name);
    used.add(name);
  }
  return names;
}

/** How long each speaker held the floor, longest first. */
export function speakerDurations(turns: readonly SpeakerTurn[]): Record<string, number> {
  const totals = new Map<string, number>();
  for (const turn of turns) totals.set(turn.speaker, (totals.get(turn.speaker) ?? 0) + Math.max(0, turn.end - turn.start));
  return Object.fromEntries([...totals].sort((a, b) => b[1] - a[1]));
}

export interface DiarizerOptions extends Omit<LoadOptions, "model"> {
  Tensor: TensorFactory;
  config?: DiarizationConfig;
}

async function loadGraph(repo: string, options: Omit<DiarizerOptions, "config" | "Tensor">): Promise<ModelBundle> {
  return loadModel({
    backends: "wasm",
    resolver: huggingFaceResolver({ repo, subfolder: "onnx", manifest: false }),
    ...options,
    model: { name: repo, computeType: "int8" },
  });
}

export class SpeakerDiarizer {
  private constructor(
    private readonly embedder: ModelBundle,
    private readonly segmenter: ModelBundle | null,
    private readonly Tensor: TensorFactory,
    readonly config: DiarizationConfig,
  ) {}

  static async load(options: DiarizerOptions): Promise<SpeakerDiarizer> {
    const { Tensor, config = {}, ...rest } = options;
    const pick = (value: string | undefined, fallback: string) => (value === undefined || value === "auto" || value === "default" ? fallback : value);
    const embedder = await loadGraph(pick(config.embeddingModel, DEFAULT_EMBEDDING_MODEL), rest);
    const segmentation = pick(config.segmentationModel, DEFAULT_SEGMENTATION_MODEL);
    const segmenter = segmentation === "" ? null : await loadGraph(segmentation, rest);
    return new SpeakerDiarizer(embedder, segmenter, Tensor, config);
  }

  get #embedSession(): ModelSession {
    return this.embedder.only;
  }

  /**
   * One L2-normalized vector for one chunk: the waveform, or Kaldi fbank when the model's input is
   * 3-D.
   */
  async embed(chunk: Float32Array): Promise<Float32Array> {
    if (chunk.length < MIN_CHUNK_SECONDS * 16000) {
      throw new DiarizationError(`a chunk of ${(chunk.length / 16000).toFixed(2)}s is too short to embed (minimum ${MIN_CHUNK_SECONDS}s)`);
    }
    const session = this.#embedSession;
    const input = session.inputs[0];
    const wantsFeatures = (input?.dims?.length ?? 0) >= 3;
    let tensor: unknown;
    if (wantsFeatures) {
      const features = fbank(chunk);
      if (features.frames === 0) throw new DiarizationError("a chunk produced no frame");
      tensor = new this.Tensor("float32", features.data, [1, features.frames, features.bins]);
    } else {
      tensor = new this.Tensor("float32", chunk, [1, chunk.length]);
    }
    const outputs = await session.run({ [input?.name ?? "input"]: tensor });
    const vector = Float32Array.from((Object.values(outputs)[0] as { data: Float32Array }).data);
    const norm = Math.max(Math.sqrt(vector.reduce((s, x) => s + x * x, 0)), 1e-12);
    return vector.map((x) => x / norm);
  }

  /**
   * Speech regions from the segmentation model, run on the 10-second windows it was trained on (on
   * a longer input its frames drift into its "nobody speaks" class mid-sentence). Only speech
   * against silence is kept: the model's speaker classes are local to each window, and their
   * frame-to-frame flips cut a turn into runs too short to embed — turns come from the overlapping
   * chunks of `chunkSpeech` instead.
   */
  async segment(audio: Float32Array): Promise<[number, number][]> {
    if (!this.segmenter) throw new DiarizationError("no segmentation model was loaded");
    const session = this.segmenter.only;
    const window = SEGMENTATION_WINDOW_S * 16000;
    const speech: number[] = [];
    let step = 0;
    for (let offset = 0; offset < audio.length; offset += window) {
      const part = audio.subarray(offset, Math.min(offset + window, audio.length));
      const padded = part.length === window ? part : (() => {
        const out = new Float32Array(window);
        out.set(part);
        return out;
      })();
      const outputs = await session.run({ [session.inputs[0]?.name ?? "input"]: new this.Tensor("float32", padded, [1, 1, window]) });
      const first = Object.values(outputs)[0] as { data: Float32Array; dims: readonly number[] };
      const classes = first.dims[first.dims.length - 1]!;
      const frames = first.data.length / classes;
      step = SEGMENTATION_WINDOW_S / frames;
      const real = Math.min(frames, Math.round((part.length / window) * frames));
      for (let f = 0; f < real; f += 1) {
        let best = 0;
        for (let c = 1; c < classes; c += 1) if (first.data[f * classes + c]! > first.data[f * classes + best]!) best = c;
        speech.push(best === SILENCE_CLASS ? 0 : 1);
      }
    }
    return runsToSpans(speech, step).filter(([start]) => speech[Math.min(speech.length - 1, Math.round(start / step))] === 1);
  }

  /** Merged speaker turns over the whole audio. */
  async diarize(audio: Float32Array, speechSpans?: readonly [number, number][]): Promise<SpeakerTurn[]> {
    const duration = audio.length / 16000;
    const chunks = chunkSpeech(this.segmenter ? await this.segment(audio) : (speechSpans ?? [[0, duration]]), this.config.chunkSeconds ?? CHUNK_SECONDS, this.config.chunkStride ?? CHUNK_STRIDE, MIN_CHUNK_SECONDS);
    if (chunks.length === 0) return [];
    const embeddings: Float32Array[] = [];
    const kept: [number, number][] = [];
    for (const [start, end] of chunks) {
      const begin = Math.max(0, Math.trunc(start * 16000));
      const finish = Math.min(audio.length, Math.trunc(end * 16000));
      if (finish - begin < Math.trunc(MIN_CHUNK_SECONDS * 16000)) continue;
      embeddings.push(await this.embed(audio.subarray(begin, finish)));
      kept.push([start, end]);
    }
    if (kept.length === 0) {
      throw new DiarizationError("no chunk was long enough to embed; the audio may be silent or the speech spans too fragmented");
    }
    const c = this.config;
    const known = c.knownSpeakers && Object.keys(c.knownSpeakers).length > 0 ? c.knownSpeakers : null;
    if (known && (c.enrolledAssignment ?? "chunk") === "chunk") return this.#assignToPrints(kept, embeddings, known, c.matchThreshold ?? ENROLLED_MATCH_THRESHOLD);
    if (kept.length === 1) return this.#finish(kept, [0], embeddings, known ? matchEnrolled(embeddings, [0], known, c.matchThreshold) : new Map(), known);
    const labels =
      c.clustering === "spectral"
        ? spectral(embeddings, { numSpeakers: c.numSpeakers ?? null, maxSpeakers: c.maxSpeakers || 8 })
        : agglomerative(embeddings, {
            threshold: c.threshold ?? CLUSTER_THRESHOLD,
            numSpeakers: c.numSpeakers ?? null,
            minSpeakers: c.minSpeakers ?? null,
            maxSpeakers: c.maxSpeakers ?? null,
          });
    return this.#finish(kept, labels, embeddings, known ? matchEnrolled(embeddings, labels, known, c.matchThreshold) : new Map(), known);
  }

  /** The similarity of each named speaker to its print, from the last `diarize()`. */
  similarities: Map<string, number> = new Map();

  /**
   * Labels into turns: identities by first appearance (`SPEAKER_00`…), the enrolled names apart
   * (`name`), with each name's similarity to its print.
   */
  #finish(kept: [number, number][], labels: readonly number[], embeddings: readonly Float32Array[], names: ReadonlyMap<number, string>, known: Readonly<Record<string, ArrayLike<number>>> | null): SpeakerTurn[] {
    const relabelled = relabelByFirstAppearance(labels);
    const idOf = (label: number) => `SPEAKER_${String(label).padStart(2, "0")}`;
    const nameById = new Map<string, string>();
    const centroids = new Map<string, Float32Array>();
    labels.forEach((label, i) => {
      const id = idOf(relabelled[i]!);
      const name = names.get(label);
      if (name) nameById.set(id, name);
      const sum = centroids.get(id) ?? new Float32Array(embeddings[i]!.length);
      embeddings[i]!.forEach((x, k) => (sum[k] = sum[k]! + x));
      centroids.set(id, sum);
    });
    this.similarities = new Map();
    for (const [id, name] of nameById) {
      if (known?.[name]) this.similarities.set(id, Math.round(dot(normalize(centroids.get(id)!), normalize(known[name]!)) * 1000) / 1000);
    }
    return toTurns(kept, relabelled, new Map(), this.config.mergeGap ?? MERGE_GAP_SECONDS).map((turn) => ({ ...turn, name: nameById.get(turn.speaker) ?? null }));
  }

  /** Each chunk named after its closest print; chunks matching none clustered among themselves. */
  #assignToPrints(kept: [number, number][], embeddings: Float32Array[], known: Readonly<Record<string, ArrayLike<number>>>, threshold: number): SpeakerTurn[] {
    const names = Object.keys(known);
    const prints = names.map((n) => normalize(known[n]!));
    const labels: number[] = [];
    const unknown: number[] = [];
    embeddings.forEach((e, i) => {
      const v = normalize(e);
      let best = -1;
      let score = -Infinity;
      prints.forEach((p, k) => {
        const s = dot(v, p);
        if (s > score) [best, score] = [k, s];
      });
      if (score >= threshold) labels[i] = best;
      else {
        labels[i] = -1;
        unknown.push(i);
      }
    });
    const nameOf = new Map<number, string>(names.map((n, k) => [k, n]));
    if (unknown.length > 0) {
      const c = this.config;
      const rest = unknown.length === 1 ? [0] : agglomerative(unknown.map((i) => embeddings[i]!), { threshold: c.threshold ?? CLUSTER_THRESHOLD, maxSpeakers: c.maxSpeakers ?? null });
      unknown.forEach((i, k) => (labels[i] = names.length + rest[k]!));
    }
    return this.#finish(kept, labels, embeddings, nameOf, known);
  }

  /**
   * A speaker's print: the mean of the embeddings of the speech, in overlapping chunks,
   * L2-normalized. Refused when there is too little speech to embed.
   */
  async print(audio: Float32Array): Promise<Float32Array> {
    const chunks = chunkSpeech([[0, audio.length / 16000]]);
    const vectors: Float32Array[] = [];
    for (const [start, end] of chunks) {
      const begin = Math.trunc(start * 16000);
      const finish = Math.min(audio.length, Math.trunc(end * 16000));
      if (finish - begin < Math.trunc(MIN_CHUNK_SECONDS * 16000)) continue;
      vectors.push(await this.embed(audio.subarray(begin, finish)));
    }
    if (vectors.length === 0) {
      throw new EnrollmentError(`${(audio.length / 16000).toFixed(2)} s of audio holds no chunk long enough to make a print (${MIN_CHUNK_SECONDS} s at least)`);
    }
    const sum = new Float32Array(vectors[0]!.length);
    for (const v of vectors) v.forEach((x, k) => (sum[k] = sum[k]! + x));
    return normalize(sum);
  }

  async close(): Promise<void> {
    await this.embedder.close();
    await this.segmenter?.close();
  }
}
