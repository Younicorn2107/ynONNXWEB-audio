/**
 * The analyses that need the audio only — `speech`, `profile`, `speakers`, `emotion`, `events`,
 * `music` — apart from Whisper.
 */

import { defaultLogger, type Logger } from "@youni/onnxweb-core";
import type { OrtLike } from "@youni/onnxweb-core";
import type { TensorFactory } from "@youni/onnxweb-core/core";

import { profileAudio } from "../audio/profile.js";
import { SILERO_REPO, VAD, type VADConfig } from "../audio/vad.js";
import { DEFAULT_EMBEDDING_MODEL, SpeakerDiarizer, assignSpeakers, speakerInfoOf, type DiarizationConfig, type SpeakerInfo, type SpeakerTurn } from "../diarization/diarizer.js";
import { Timeline, type Analysis, type TimelineData } from "../timeline.js";
import { AudioSetTagger, EmotionClassifier, type Detection } from "./classifiers.js";

/** The analyses that read the audio alone. */
export const AUDIO_ONLY_ANALYSES: ReadonlySet<Analysis> = new Set(["speech", "profile", "speakers", "emotion", "events", "music"]);

/** What the audio-only analyses take from the transcription options. */
export interface AudioAnalysisOptions {
  diarization?: DiarizationConfig;
  vadConfig?: Partial<VADConfig>;
  /**
   * `backends`: where the classifier runs — WebGPU, then WebAssembly, by default; `computeType`
   * float16 there (int8 for events, int4 for emotion when `backends` is "wasm"). 70 s of speech:
   * AST 26.7 s on WebAssembly (int8) against 2.7 s on WebGPU (float16), emotion 12.4 against 1.7 s
   * (RTX 4070 Ti, 2026-10-10).
   */
  emotion?: { model?: string; computeType?: string; backends?: string };
  events?: { model?: string; computeType?: string; threshold?: number; musicThreshold?: number; backends?: string };
  /** AudioSet window step, in seconds. */
  interval?: number;
  signal?: AbortSignal;
}

/** The audio-only analyses of one audio, not yet on a transcript. */
export interface AudioOnlyAnalysis {
  analyses: Analysis[];
  /** Their annotations, ids not assigned. */
  annotations: Omit<TimelineData["annotations"][number], "id">[];
  speakers: SpeakerTurn[] | null;
  speakerInfo: SpeakerInfo[] | null;
}

export interface AudioAnalyzerOptions {
  ort: OrtLike;
  Tensor: TensorFactory;
  logger?: Logger;
}

const DEFAULT_ANALYSIS_BACKENDS = "webgpu,wasm";

/** float16 when the GPU comes first; the quantized weights WebAssembly runs best otherwise. */
function defaultComputeType(backends: string | undefined, cpu: string): string {
  return (backends ?? DEFAULT_ANALYSIS_BACKENDS).startsWith("webgpu") ? "float16" : cpu;
}

export class AudioAnalyzer {
  readonly #logger: Logger;
  #vad: VAD | null = null;
  readonly #prints = new Map<string, Float32Array>();

  constructor(private readonly options: AudioAnalyzerOptions) {
    this.#logger = options.logger ?? defaultLogger;
  }

  /**
   * The audio-only ones among `analyses`, run at once where they are independent: the speaker chain
   * (diarization, then emotion on its turns), the signal, AudioSet.
   */
  async analyze(audio: Float32Array, analyses: readonly Analysis[], options: AudioAnalysisOptions = {}): Promise<AudioOnlyAnalysis> {
    const wanted = analyses.filter((a) => AUDIO_ONLY_ANALYSES.has(a));
    const duration = audio.length / 16000;
    const speakerChain = (async () => {
      const turns = wanted.includes("speakers") ? await this.#step("speakers", () => this.diarize(audio, options.diarization ?? {})) : null;
      const emotions = wanted.includes("emotion") ? await this.#step("emotion", () => this.#emotions(audio, turns?.speakers ?? null, options)) : null;
      return { turns, emotions };
    })();
    const signals = Promise.all(wanted.filter((a): a is "speech" | "profile" => a === "speech" || a === "profile").map((a) => this.#step(a, () => this.#signal(audio, a, duration, options))));
    const tagging = wanted.some((a) => a === "events" || a === "music") ? this.#tagEvents(audio, wanted, options) : Promise.resolve(null);
    const [chain, signalNotes, tagged] = await Promise.all([speakerChain, signals, tagging]);

    const timeline = new Timeline();
    for (const notes of signalNotes) for (const note of notes) timeline.add(note);
    for (const turn of chain.turns?.speakers ?? []) {
      timeline.add({ kind: "speaker", start: turn.start, end: turn.end, label: turn.speaker, speakerId: turn.speaker, source: options.diarization?.embeddingModel ?? DEFAULT_EMBEDDING_MODEL, granularity: "segment" });
    }
    if (chain.emotions) writeDetections(timeline, "emotion", chain.emotions.model, chain.emotions.found);
    if (tagged) {
      if (wanted.includes("events")) writeDetections(timeline, "event", tagged.model, tagged.events);
      if (wanted.includes("music")) writeDetections(timeline, "music", tagged.model, tagged.music);
    }
    return {
      analyses: wanted,
      annotations: timeline.toJSON().annotations.map(({ id: _id, ...note }) => note),
      speakers: chain.turns?.speakers ?? null,
      speakerInfo: chain.turns?.info ?? null,
    };
  }

  /** Speaker turns, the enrolled speakers named. */
  async diarize(audio: Float32Array, config: DiarizationConfig = {}): Promise<{ speakers: SpeakerTurn[]; info: SpeakerInfo[] }> {
    const diarizer = await SpeakerDiarizer.load({ ort: this.options.ort, Tensor: this.options.Tensor, config: this.withPrints(config) });
    try {
      const speakers = await diarizer.diarize(audio);
      return { speakers, info: speakerInfoOf(speakers, diarizer.similarities) };
    } finally {
      await diarizer.close();
    }
  }

  /** Speech spans of the whole audio, from Silero. */
  async speech(audio: Float32Array, config: Partial<VADConfig> = {}) {
    this.#vad ??= await VAD.load({ ort: this.options.ort, Tensor: this.options.Tensor, config });
    return this.#vad.detectSpeech(audio);
  }

  /** The enrolled prints, added to a diarization config that names none itself. */
  withPrints(config: DiarizationConfig): DiarizationConfig {
    if (config.knownSpeakers || this.#prints.size === 0) return config;
    return { ...config, knownSpeakers: Object.fromEntries(this.#prints) };
  }

  async enroll(name: string, audio: Float32Array, config: DiarizationConfig = {}): Promise<void> {
    const diarizer = await SpeakerDiarizer.load({ ort: this.options.ort, Tensor: this.options.Tensor, config: { ...config, segmentationModel: "" } });
    try {
      this.#prints.set(name, await diarizer.print(audio));
    } finally {
      await diarizer.close();
    }
  }

  forget(name: string): boolean {
    return this.#prints.delete(name);
  }

  get enrolled(): string[] {
    return [...this.#prints.keys()];
  }

  prints(): Record<string, number[]> {
    return Object.fromEntries([...this.#prints].map(([name, print]) => [name, Array.from(print)]));
  }

  loadPrints(prints: Readonly<Record<string, ArrayLike<number>>>): void {
    for (const [name, print] of Object.entries(prints)) this.#prints.set(name, Float32Array.from(print));
  }

  async close(): Promise<void> {
    await this.#vad?.close();
    this.#vad = null;
  }

  async #step<T>(name: string, work: () => Promise<T>): Promise<T> {
    const started = performance.now();
    this.#logger.info(`analysis '${name}' started`);
    try {
      return await work();
    } finally {
      this.#logger.info(`analysis '${name}' done in ${((performance.now() - started) / 1000).toFixed(1)} s`);
    }
  }

  async #signal(audio: Float32Array, analysis: "speech" | "profile", duration: number, options: AudioAnalysisOptions): Promise<Parameters<Timeline["add"]>[0][]> {
    if (analysis === "speech") {
      return (await this.speech(audio, options.vadConfig ?? {})).map((span) => ({ kind: "speech" as const, start: span.start, end: span.end, source: SILERO_REPO, granularity: "segment" as const }));
    }
    return duration > 0 ? [{ kind: "profile" as const, start: 0, end: duration, label: "profile", value: { ...profileAudio(audio) }, source: "profile", granularity: "clip" as const }] : [];
  }

  /** `emotion` on speech: the speakers' turns when known, speech spans otherwise. */
  async #emotions(audio: Float32Array, speakers: readonly SpeakerTurn[] | null, options: AudioAnalysisOptions): Promise<{ model: string; found: Detection[] }> {
    const regions = speakers && speakers.length > 0 ? speakers.map((t) => ({ start: t.start, end: t.end, speakerId: t.speaker })) : (await this.speech(audio, options.vadConfig ?? {})).map((s) => ({ start: s.start, end: s.end, speakerId: null }));
    const classifier = await EmotionClassifier.load({ ort: this.options.ort, Tensor: this.options.Tensor, backends: options.emotion?.backends ?? DEFAULT_ANALYSIS_BACKENDS, ...(options.emotion?.model ? { model: options.emotion.model } : {}), computeType: options.emotion?.computeType ?? defaultComputeType(options.emotion?.backends, "int4") } as never);
    try {
      return { model: classifier.model, found: await classifier.classify(audio, regions) };
    } finally {
      await classifier.close();
    }
  }

  /** `events` and `music` from AudioSet, on 10-second windows. */
  async #tagEvents(audio: Float32Array, analyses: readonly Analysis[], options: AudioAnalysisOptions): Promise<{ model: string; events: Detection[]; music: Detection[] }> {
    const want = new Set(analyses.filter((a): a is "events" | "music" => a === "events" || a === "music"));
    return this.#step([...want].join("+"), async () => {
      const tagger = await AudioSetTagger.load({ ort: this.options.ort, Tensor: this.options.Tensor, backends: options.events?.backends ?? DEFAULT_ANALYSIS_BACKENDS, ...(options.events?.model ? { model: options.events.model } : {}), computeType: options.events?.computeType ?? defaultComputeType(options.events?.backends, "int8") } as never);
      try {
        const found = await tagger.tag(audio, {
          ...(options.interval ? { interval: options.interval } : {}),
          ...(options.events?.threshold !== undefined ? { threshold: options.events.threshold } : {}),
          ...(options.events?.musicThreshold !== undefined ? { musicThreshold: options.events.musicThreshold } : {}),
          want,
        });
        return { model: tagger.model, events: found.events, music: found.music };
      } finally {
        await tagger.close();
      }
    });
  }
}

export function writeDetections(timeline: Timeline, kind: "emotion" | "event" | "music", source: string, found: readonly Detection[]): void {
  for (const d of found) {
    timeline.add({
      kind,
      start: d.start,
      end: d.end,
      label: d.label,
      confidence: d.confidence,
      source,
      granularity: "segment",
      speakerId: d.speakerId,
      ...(d.sourceLabel !== d.label ? { sourceLabel: d.sourceLabel } : {}),
    });
  }
}

/** What `mergeAnalysis` needs of a transcript. */
interface Transcript {
  duration: number;
  language?: string | null;
  languageProbability?: number;
  languagePortions?: readonly { language: string; start: number; end: number; probability: number }[] | null;
  segments: readonly { start: number; end: number; speaker?: string | null; words?: readonly { word: string; start: number; end: number; probability: number }[] | null }[];
  speakers?: SpeakerTurn[] | null;
  speakerInfo?: SpeakerInfo[] | null;
}

/**
 * A transcript with its analyses: the speakers on its segments, and one timeline — the audio-only
 * annotations, its language, its words.
 */
export function mergeAnalysis<R extends Transcript>(
  result: R,
  analysis: AudioOnlyAnalysis | null,
  analyses: readonly Analysis[],
  source: string,
  assignment: DiarizationConfig["assignmentStrategy"] = "longest",
): R & { timeline: TimelineData; analyses: Analysis[] } {
  const timeline = new Timeline();
  for (const note of analysis?.annotations ?? []) timeline.add(note as Parameters<Timeline["add"]>[0]);
  const speakers = analysis?.speakers ?? result.speakers ?? null;
  const segments = analysis?.speakers ? assignSpeakers(result.segments as never, analysis.speakers, assignment ?? "longest") : result.segments;
  if (analyses.includes("language")) {
    const found = result.languagePortions ?? [{ language: result.language ?? "", start: 0, end: result.duration, probability: result.languageProbability ?? 0 }];
    for (const p of found) timeline.add({ kind: "language", start: p.start, end: p.end, label: p.language, confidence: p.probability, source, granularity: found.length > 1 ? "segment" : "clip" });
  }

  if (analyses.includes("words")) {
    for (const segment of segments as Transcript["segments"]) {
      for (const word of segment.words ?? []) {
        timeline.add({ kind: "word", start: word.start, end: word.end, label: word.word, confidence: word.probability || null, source, granularity: "word", speakerId: segment.speaker ?? null });
      }
    }
  }
  return {
    ...result,
    segments: [...segments] as never,
    speakers,
    speakerInfo: analysis?.speakerInfo ?? result.speakerInfo ?? null,
    timeline: timeline.toJSON(),
    analyses: [...analyses],
  };
}
