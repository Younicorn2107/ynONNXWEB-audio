/** The façade: audio in, timed transcript out. */

import type { LoadOptions, ModelBundle, ModelSession } from "@youni/onnxweb-core";
import { defaultLogger, detectCapabilities, loadModel, wasmRuntime } from "@youni/onnxweb-core";
import {
  ConfigurationError,
  GenerationEngine,
  GenerationError,
  Tokenizer,
  totalSizeMb,
  IndexedDbStore,
  ResultCache,
  combineIds,
  hashBytes,
  hashParams,
  type KeyValueStore,
  type GenerationOutput,
  type TensorFactory,
  type TensorLike,
} from "@youni/onnxweb-core/core";

import { EnrollmentError, YnONNXAudioError } from "../errors.js";
import { WINDOW_SAMPLES, WINDOW_SECONDS, logMel, melConfigFor, type MelConfig } from "../audio/mel.js";
import { buildPrompt, DEFAULT_MAX_CONTEXT } from "./prompt.js";
import { type DecodingProfile, profileFor } from "./profiles.js";
import {
  DEFAULT_QUALITY,
  TEMPERATURE_SCALE,
  compressionRatio,
  isDegenerate,
  isSilent,
  type QualityConfig,
} from "./quality.js";
import { TimestampRules } from "./rules.js";
import { extractSegments, mergeSegments, type Segment } from "./segments.js";
import { SectionScheduler, WaveScheduler, type AudioInfo, type SectionState, type WindowResult } from "./schedule.js";
import { WhisperTokens } from "./tokens.js";
import { MOONSHINE_REPOS, asRoles, isPublished, moonshineSource, publishedSource, type HalfPrecision } from "./published.js";
import { maxWindows as waveWindows, sectionBudget } from "./batching.js";
import { WaveJournal } from "./journal.js";
import { ForcedAligner } from "../alignment/ctc.js";
import { buildPlan, type ExecutionPlan, type LicensePolicy } from "../plan.js";
import { DEFAULT_CTC_MODEL } from "../alignment/ctc.js";
import { DEFAULT_EMBEDDING_MODEL, DEFAULT_SEGMENTATION_MODEL, assignSpeakers, type DiarizationConfig, type SpeakerInfo, type SpeakerTurn } from "../diarization/diarizer.js";
import { Timeline, checkAnalyses, emptyTimeline, type Analysis, type TimelineData } from "../timeline.js";
import { StreamSession, type StreamOptions } from "../streaming.js";
import { AUDIO_ONLY_ANALYSES, AudioAnalyzer, mergeAnalysis, type AudioOnlyAnalysis } from "../analysis/analyzer.js";
import { mergePunctuation, proportionalWords, type Word } from "../alignment/punctuation.js";
import { WEB_MEASUREMENTS } from "./autoTables.js";
import { browserHardware, parseAuto, resolveSettings, retryRung, type Measurements, type Resolution } from "./resolver.js";
import type { AudioProfile } from "../audio/profile.js";
import { DEFAULT_VAD, VAD, type VADConfig } from "../audio/vad.js";
import {
  WHISPER_PARAMS_M,
  applyPadding,
  clipToSpeech,
  groupSpeech,
  restoreTime,
  shouldUseVad,
  type SpeechMap,
  type VADMode,
} from "../audio/vadPolicy.js";

/** Architectures this package can drive. */
export const WHISPER_FAMILIES: ReadonlySet<string> = new Set(["whisper", "distil-whisper"]);

/**
 * Moonshine: the same decoder loop, none of Whisper's prompt grammar. Its decoder has Whisper's
 * graph shape, so it runs on the same engine; it lacks language and task tokens (both published
 * sizes are English-only), `<|nospeech|>` (silence is judged on the log-probability alone) and
 * timestamp tokens (a segment spans its window). Its encoder takes the waveform at any length.
 */
export const MOONSHINE_FAMILIES: ReadonlySet<string> = new Set(["moonshine"]);

/** Moonshine's start and end tokens, from the model's config (the tokenizer does not mark them). */
export const MOONSHINE_BOS = 1;
export const MOONSHINE_EOS = 2;

/** Moonshine's `max_position_embeddings`. */
const MOONSHINE_MAX_POSITIONS = 512;

/**
 * Samples per Moonshine encoder frame: under two frames the graph fails, so an empty window is two
 * frames of silence.
 */
const MOONSHINE_SAMPLES_PER_FRAME = 384;

/**
 * Whisper's decoder has 448 positions in total, prompt included, from tiny to large-v3-turbo: the
 * prompt is already in the cache before the first step.
 */
export const MAX_TARGET_POSITIONS = 448;

const PROFILES: readonly DecodingProfile[] = ["default", "speed", "quality"];

/** Language probes spread over the audio, as on the Python side. */
export const LANGUAGE_PROBES = 3;

export interface TranscribeOptions {
  language?: string | null;
  task?: "transcribe" | "translate";
  /** `"auto"` walks the temperature ladder, a number pins it. */
  temperature?: number | "auto";
  /** Carry the previous window's tokens into the next prompt. */
  conditionOnPreviousText?: boolean;
  /**
   * Text the first window of each audio is conditioned on (Whisper's `initial_prompt`): vocabulary,
   * or what came before.
   */
  initialPrompt?: string;
  /**
   * Which measured answer to use for this checkpoint. Never overrides a setting named explicitly
   * here.
   */
  decodingProfile?: DecodingProfile;
  /** Ban any n-gram of this size already emitted in the window (0: off). */
  noRepeatNgramSize?: number;
  /** The first timestamp of a window may not exceed this (seconds). */
  maxInitialTimestamp?: number | null;
  withoutTimestamps?: boolean;
  /**
   * Remove silence before decoding: `auto` turns it on for models of 500 M parameters and more,
   * which narrate silence, and off for small ones, which lose words to the cuts (measured on the
   * Python side).
   */
  vad?: VADMode;
  vadConfig?: Partial<VADConfig> & { groupMaxDurationS?: number; groupMaxGapS?: number };
  maxContextTokens?: number;
  quality?: Partial<QualityConfig>;
  /**
   * Resume an interrupted run: the progress is journaled after every wave in this store (`true`:
   * the page's IndexedDB) and replayed by a run of the same audios and settings. The journal is
   * removed once the run ends.
   */
  checkpoint?: KeyValueStore | boolean;
  /**
   * Keep finished results (`true`: in IndexedDB), keyed by the audio's content and the settings:
   * the same audio transcribed again the same way comes back without a decode.
   */
  cache?: KeyValueStore | boolean;
  /**
   * Re-decode a section whose mean log-probability falls below -0.85 at the most accurate rung
   * measured on this device, when one fits (single audio).
   */
  retry?: boolean;
  /** The measurements a retry chooses its rung from; the browser tables by default. */
  measurements?: Measurements;
  /**
   * Analyses beyond the transcript (spec § 2.2): `speech`, `profile`, `language`, `speakers`,
   * `words`. They fill `timeline`; an unknown one is refused before anything runs.
   */
  analyze?: readonly string[];
  /** The ids of the audios, in order; by default each one's content fingerprint. */
  audioIds?: readonly (string | undefined)[];
  /**
   * Licence categories accepted (`{ non_commercial: false }`); a step outside is kept, with a
   * warning in the plan.
   */
  licenses?: LicensePolicy;
  /** How speakers are found, when `speakers` is analysed. */
  diarization?: DiarizationConfig;
  /** The `emotion` analysis: its model (a Hub repository) and precision. */
  emotion?: { model?: string; computeType?: string };
  /**
   * The `events` and `music` analyses: AudioSet model, precision, thresholds (`interval` sets the
   * cadence).
   */
  events?: { model?: string; computeType?: string; threshold?: number; musicThreshold?: number };
  /**
   * Find the language over time and transcribe each portion in its own language (a multilingual
   * model, language not given): for recordings that change language.
   */
  languageOverTime?: boolean;
  /** `offline` (the default): nothing leaves for an online service; none is configured today. */
  network?: "offline" | "allowed";
  /**
   * Seconds between two annotations of an analysis that has a cadence (spec § 2.2): for `language`,
   * one probe per `interval` seconds.
   */
  interval?: number;
  /** Time every word (see `alignment` for how). */
  wordTimestamps?: boolean;
  /**
   * How words are timed: `enabled` aligns even without `wordTimestamps`; `backend: "ctc"` insists
   * on CTC forced alignment with `ctcModel` (a Hub repository, wav2vec2-base-960h by default).
   * Without a CTC model, words are a proportional split of each segment.
   */
  alignment?: { enabled?: boolean; backend?: "auto" | "ctc"; ctcModel?: string };
  /** Windows decoded per wave; by default sized on memory (at most 8). */
  maxWindows?: number;
  /**
   * How many sections to cut the audios into, decoded side by side. Never below 90 s each; by
   * default two per audio, as many as a wave holds.
   */
  sections?: number;
  signal?: AbortSignal;
  onProgress?: (progress: TranscribeProgress) => void;
}

export interface TranscribeProgress {
  phase: "language" | "decoding" | "assembling";
  /** 0..1 over the whole audio. */
  ratio: number;
  seconds: number;
  totalSeconds: number;
}

/**
 * A transcription: the fields of Python's `TranscriptionResult` (same names, same meaning), plus
 * the spec's `timeline`, `analyses` and `plan`.
 */
export interface TranscriptionResult {
  text: string;
  language: string | null;
  languageProbability: number;
  duration: number;
  segments: Segment[];
  /** Every word of every segment, when words were timed. */
  words: Word[] | null;
  /** Who spoke when, when speakers were analysed (`speaker` is the identity, `SPEAKER_00`…). */
  speakers: SpeakerTurn[] | null;
  /** Each speaker's identity, enrolled name and similarity to its print (spec § 6.1). */
  speakerInfo: SpeakerInfo[] | null;
  /**
   * The language over time, when `languageOverTime` was asked (each portion transcribed in its
   * language).
   */
  languagePortions: LanguagePortion[] | null;
  /** The id given, else the content hash: the business identity. */
  audioId: string;
  /** A fingerprint of the decoded audio's content: the identity caches use. */
  contentHash: string;
  /** What ran (`whisper-small_int8…`), not what was asked for. */
  modelName: string;
  elapsedS: number;
  /** Audio seconds per wall-clock second; 1 is real time. */
  speedRatio: number;
  /** How it ran: the precision, the backend that actually computed, whether silence was removed. */
  metadata: { computeType: string; provider: string; decodeMode: "batched"; vadApplied: boolean };
  /**
   * Every analysis's annotations on the original timeline; empty when only the transcript was asked
   * for.
   */
  timeline: TimelineData;
  /** The analyses actually done. */
  analyses: Analysis[];
  /**
   * The plan that ran: each step, its model, device, licence and reason (with what `auto` decided).
   */
  plan: ExecutionPlan;
  /** Each decoded window, for inspection. */
  windows: WindowResult[];
}

/** What `analyze()` returns (spec § 6.2). */
export interface AudioAnalysisResult {
  audioId: string;
  contentHash: string;
  duration: number;
  timeline: TimelineData;
  analyses: Analysis[];
  plan: ExecutionPlan;
  speakers: SpeakerTurn[] | null;
  speakerInfo: SpeakerInfo[] | null;
}

/** What `detectLanguage()` returns (spec § 6.4). */
export interface LanguageDetectionResult {
  language: string;
  probability: number;
  /** The `language` annotations: one for the clip, or one per portion with `overTime`. */
  timeline: TimelineData;
  model: string;
}

/**
 * What `stream()` takes: the session's options, the other models of `double` and `post`, and how to
 * load them.
 */
export interface TranscriberStreamOptions extends StreamOptions {
  /** `double`: the fast model (a catalogue name); `auto` takes `tiny`. */
  fastModel?: string;
  /** `post`: the model of the final pass; this one when omitted. */
  finalModel?: string;
  /** Load settings for those models (backends, store…). */
  loadOptions?: Partial<Omit<TranscriberOptions, "model">>;
}

export interface TranscriberOptions extends Omit<LoadOptions, "model" | "computeType"> {
  /**
   * A published checkpoint (`"small"`, `"small@plain-int4"`), loaded from its repository unless a
   * resolver is given; or any key a resolver serves.
   */
  model: LoadOptions["model"];
  /**
   * One precision, or one per half: `{ encoder: "int8", decoder: "float32" }`. `"auto"` chooses it.
   */
  computeType?: HalfPrecision;
  /**
   * The session level bare `"auto"` takes (`quality`, `balanced`, `fast`, `max_speed`): a WER
   * budget against the best measured candidate.
   */
  profile?: string;
  /** The language expected, for the automatic choice (English-only models, the corpus). */
  language?: string | null;
  /** The audio's measurements (`profileAudio`), for the automatic choice. */
  audio?: AudioProfile | null;
  /** The measurements `auto` chooses from; the browser tables by default. */
  measurements?: Measurements;
  /** Where published checkpoints are fetched from: the Hub, or a mirror of it. */
  endpoint?: string;
  /** The ort module's Tensor, for the tensors this package builds. */
  Tensor: TensorFactory;
}

/**
 * What the loader is handed: a published checkpoint becomes its repository and a precision per
 * half; anything else passes through untouched.
 */
export function loadTarget(options: TranscriberOptions): Pick<LoadOptions, "model" | "computeType" | "resolver"> {
  const name = typeof options.model === "string" ? options.model : options.model.name;
  if (options.resolver === undefined && isPublished(name)) {
    const precision = options.computeType ?? (typeof options.model === "string" ? "int8" : options.model.computeType);
    const source = publishedSource(name, precision, options.endpoint ? { endpoint: options.endpoint } : {});
    return {
      model: { name, computeType: source.computeType.decoder },
      computeType: source.computeType,
      resolver: source.resolver,
    };
  }
  const moonshine = options.resolver === undefined ? moonshineSource(name, options.endpoint ? { endpoint: options.endpoint } : {}) : null;
  if (moonshine) {
    const precision = options.computeType ?? (typeof options.model === "string" ? "float32" : options.model.computeType);
    const scalar = typeof precision === "string" && parseAuto(precision) === null ? precision : "float32";
    return { model: { name: MOONSHINE_REPOS[name]!, computeType: scalar }, resolver: moonshine };
  }
  return {
    model: options.model,
    ...(options.computeType !== undefined ? { computeType: options.computeType } : {}),
    ...(options.resolver !== undefined ? { resolver: options.resolver } : {}),
  };
}

/** One stretch of the audio in one language. */
export interface LanguagePortion {
  language: string;
  start: number;
  end: number;
  /** The mean probability of the windows it was found in. */
  probability: number;
}

/**
 * Window answers merged into portions: an unsure window (below `minProbability`) takes its previous
 * neighbour's language (or the next one's at the start), then neighbours of one language join.
 */
export function mergePortions(probes: readonly LanguagePortion[], minProbability = 0.5): LanguagePortion[] {
  const sure = probes.map((p) => ({ ...p }));
  for (let i = 0; i < sure.length; i += 1) {
    if (sure[i]!.probability >= minProbability) continue;
    const neighbour = sure.slice(0, i).reverse().find((p) => p.probability >= minProbability) ?? sure.slice(i + 1).find((p) => p.probability >= minProbability);
    if (neighbour) sure[i]!.language = neighbour.language;
  }
  const out: (LanguagePortion & { count: number })[] = [];
  for (const p of sure) {
    const last = out[out.length - 1];
    if (last && last.language === p.language) {
      last.end = p.end;
      last.probability += p.probability;
      last.count += 1;
    } else {
      out.push({ ...p, count: 1 });
    }
  }
  return out.map(({ count, ...p }) => ({ ...p, probability: p.probability / count }));
}

/** The language covering most of the audio. */
export function dominantLanguage(portions: readonly LanguagePortion[]): { language: string; probability: number } {
  const totals = new Map<string, number>();
  for (const p of portions) totals.set(p.language, (totals.get(p.language) ?? 0) + (p.end - p.start));
  const language = [...totals].reduce((a, b) => (b[1] > a[1] ? b : a), ["en", -1] as [string, number])[0];
  const mine = portions.filter((p) => p.language === language);
  return { language, probability: mine.reduce((s, p) => s + p.probability, 0) / Math.max(mine.length, 1) };
}

/** The language of the portion holding time `t`, or null without portions. */
export function portionAt(portions: readonly LanguagePortion[] | null | undefined, t: number): string | null {
  if (!portions || portions.length === 0) return null;
  return (portions.find((p) => t >= p.start && t < p.end) ?? portions[portions.length - 1]!).language;
}

/** Below this mean log-probability a section is retried (Python's `SCORE_THRESHOLDS["retry"]`). */
export const RETRY_LOGPROB = -0.85;

/** Mean log-probability of the windows that produced text. */
export function meanLogprob(results: readonly WindowResult[]): number {
  const usable = results.filter((r) => r.tokens.length > 0 && !r.isSilent);
  return usable.length === 0 ? 0 : usable.reduce((s, r) => s + r.avgLogprob, 0) / usable.length;
}

export class AudioTranscriber {
  #vad: VAD | null = null;
  /** The audio-only analyses, and the enrolled speakers' prints. */
  #analyzerInstance: AudioAnalyzer | null = null;

  get #analyzer(): AudioAnalyzer {
    this.#analyzerInstance ??= new AudioAnalyzer({ ort: this.ort as never, Tensor: this.Tensor, logger: this.#logger });
    return this.#analyzerInstance;
  }

  private constructor(
    private readonly bundle: ModelBundle,
    private readonly tokenizer: Tokenizer,
    /** Whisper's special tokens; null for Moonshine, which has none. */
    private readonly tokens: WhisperTokens | null,
    private readonly Tensor: TensorFactory,
    private readonly encoder: ModelSession,
    private readonly engine: GenerationEngine,
    /** The log-mel front end; null for Moonshine, whose encoder takes the waveform. */
    private readonly mel: MelConfig | null,
    private readonly ort: TranscriberOptions["ort"],
  ) {}

  static async load(options: TranscriberOptions): Promise<AudioTranscriber> {
    const resolution = await AudioTranscriber.resolve(options);
    const transcriber = await AudioTranscriber.#build(options, resolution);

    const chosen = resolution?.decisions.computeType;
    if (chosen && chosen.source.startsWith("auto") && String(chosen.value).startsWith("int4")) {
      try {
        await transcriber.transcribe(new Float32Array(16000).map((_, i) => 0.05 * Math.sin(i / 7)), { temperature: 0, vad: "off", ...(transcriber.isMultilingual ? { language: "en" } : {}) });
      } catch (error) {
        options.logger?.warn(`int4 does not run here (${(error as Error).message}); falling back to int8`);
        await transcriber.close();
        const int8: Resolution = { ...resolution!, decisions: { ...resolution!.decisions, computeType: { ...chosen, value: "int8", reason: `int4 failed on this backend (${(error as Error).message}); int8 instead` } } };
        return AudioTranscriber.#build(options, int8);
      }
    }
    return transcriber;
  }

  static async #build(options: TranscriberOptions, resolution: Resolution | null): Promise<AudioTranscriber> {
    const resolved: TranscriberOptions = resolution
      ? {
          ...options,
          model: resolution.decisions.model.value,
          computeType: resolution.decisions.computeType.value,
          ...(options.backends === undefined && resolution.decisions.backend.value !== "auto"
            ? { backends: resolution.decisions.backend.value }
            : {}),
        }
      : options;
    const {
      endpoint: _endpoint,
      Tensor: _tensor,
      profile: _profile,
      language: _language,
      audio: _audio,
      measurements: _measurements,
      ...rest
    } = resolved;
    const bundle = await loadModel({ extraFiles: ["tokenizer_config.json"], ...rest, ...loadTarget(resolved) });

    const family = bundle.manifest.architecture;
    if (family !== null && !WHISPER_FAMILIES.has(family) && !MOONSHINE_FAMILIES.has(family)) {
      throw new YnONNXAudioError(
        `'${bundle.manifest.modelKey}' is a ${family} model, and this package drives Whisper and ` +
          `Moonshine graphs only (${[...WHISPER_FAMILIES, ...MOONSHINE_FAMILIES].join(", ")}).`,
      );
    }
    const moonshine = family !== null && MOONSHINE_FAMILIES.has(family);
    const encoder = bundle.get("encoder");
    if (!encoder) {
      throw new YnONNXAudioError(
        `'${bundle.manifest.modelKey}' has no encoder graph (roles: ${bundle.roles.join(", ")}). ` +
          `A Whisper entry needs an encoder and a decoder.`,
      );
    }
    const tokenizerFile = bundle.manifest.tokenizerFile ?? "tokenizer.json";

    const tokenizerJson = bundle.assetText(tokenizerFile);
    if (tokenizerJson === null) {
      throw new YnONNXAudioError(
        `'${bundle.manifest.modelKey}' came without ${tokenizerFile}: its output would be ` +
          `token ids and nothing readable.`,
      );
    }
    const tokenizer = Tokenizer.fromJson(tokenizerJson, bundle.assetText("tokenizer_config.json"));
    const tokens = moonshine ? null : new WhisperTokens(tokenizer, { vocabSize: bundle.manifest.logitsVocabSize });

    const mel = moonshine ? null : melConfigFor({ encoderDims: encoder.inputs[0]?.dims, logitsVocabSize: bundle.manifest.logitsVocabSize });
    const engine = new GenerationEngine(bundle, options.Tensor, options.logger ? { logger: options.logger } : {});
    const transcriber = new AudioTranscriber(bundle, tokenizer, tokens, options.Tensor, encoder, engine, mel, options.ort);
    transcriber.#options = resolved;
    transcriber.resolution = resolution;
    return transcriber;
  }

  /**
   * The automatic settings, decided BEFORE anything is loaded; null when neither the model nor the
   * precision is `auto`. Readable with `explain()`.
   */
  static async resolve(options: TranscriberOptions): Promise<Resolution | null> {
    const name = typeof options.model === "string" ? options.model : options.model.name;

    const precision = options.computeType ?? (typeof options.model === "string" ? "auto" : options.model.computeType);
    if (parseAuto(name) === null && parseAuto(precision) === null) return null;
    if (MOONSHINE_REPOS[name] !== undefined) return null;
    const capabilities = await detectCapabilities();
    const deviceMemory = (globalThis as { navigator?: { deviceMemory?: number } }).navigator?.deviceMemory ?? null;
    const backend = options.backends === "webgpu" || options.backends === "wasm" ? options.backends : "auto";
    return resolveSettings({
      model: name,
      ...(precision !== undefined ? { computeType: precision } : {}),
      backend,
      ...(options.profile !== undefined ? { profile: options.profile } : {}),
      language: options.language ?? null,
      audio: options.audio ?? null,
      hardware: browserHardware({
        webgpu: capabilities.webgpu,
        wasmHeapMb: wasmRuntime(capabilities.memory64).heapCeilingMb,
        deviceMemoryGb: deviceMemory,
      }),
      measurements: options.measurements ?? WEB_MEASUREMENTS,
    });
  }

  /** What `auto` decided, and why; null when nothing was automatic. */
  resolution: Resolution | null = null;
  #options: TranscriberOptions | null = null;

  /** The checkpoint name this was loaded as (`small`), without a variant. */
  get modelName(): string {
    const model = this.#options?.model;
    const name = model === undefined ? this.bundle.manifest.modelKey : typeof model === "string" ? model : model.name;
    return name.split("@")[0]!;
  }

  /** The precision of each half, as loaded. */
  get precision(): { encoder: string; decoder: string } {
    const declared = this.bundle.manifest.computeType;
    const halves = Object.fromEntries(
      declared.split(",").filter((p) => p.includes(":")).map((p) => p.split(":", 2) as [string, string]),
    );
    if (halves["encoder"] && halves["decoder"]) return { encoder: halves["encoder"], decoder: halves["decoder"] };
    const requested = this.#options?.computeType;
    if (requested !== undefined && (typeof requested !== "string" || parseAuto(requested) === null)) return asRoles(requested);
    return asRoles(declared);
  }

  get backend(): string {
    return this.encoder.backend;
  }

  get isMultilingual(): boolean {
    return this.tokens?.isMultilingual ?? false;
  }

  /**
   * How many windows one wave holds by default: Python's ceiling of 8, lowered to what the
   * WebAssembly heap or a WebGPU binding can hold.
   */
  get waveSize(): number {
    return waveWindows({
      backend: this.backend,
      kv: this.bundle.manifest.kv,
      weightsMb: totalSizeMb(this.bundle.manifest),
      heapCeilingMb: wasmRuntime().heapCeilingMb,
      maxBindingBytes: this.bundle.adapterLimits?.maxStorageBufferBindingSize ?? null,
    });
  }

  /** How many mel bands this model was resolved to take. */
  get melBands(): number {
    return this.mel?.nMels ?? 0;
  }

  /** The encoder's output for one window starting at `seek` seconds. */
  async #encode(audio: Float32Array, seek: number): Promise<TensorLike | null> {
    const offset = Math.round(seek * 16000);
    const slice = audio.subarray(offset, Math.min(offset + WINDOW_SAMPLES, audio.length));
    if (slice.length === 0) return null;
    return this.#encodeMany([[audio, seek]]);
  }

  /** The language spoken, from one decode step per probe window. */
  /**
   * Speech spans of `audio` (Silero), in seconds: what a live session decodes and detects the
   * language on.
   */
  async detectSpeech(audio: Float32Array): Promise<{ start: number; end: number }[]> {
    this.#vad ??= await VAD.load({ ort: this.ort, Tensor: this.Tensor, config: {} });
    return this.#vad.detectSpeech(audio);
  }

  async detectLanguage(audio: Float32Array, options: { overTime?: boolean; interval?: number; signal?: AbortSignal } = {}): Promise<LanguageDetectionResult> {
    const duration = audio.length / 16000;
    const model = this.bundle.manifest.modelKey;
    const timeline = new Timeline();
    const found = await this.#detect(audio, options.signal);
    if (options.overTime) {
      const portions = await this.languageOverTime(audio, { ...(options.interval ? { windowS: options.interval } : {}), ...(options.signal ? { signal: options.signal } : {}) });
      for (const p of portions) timeline.add({ kind: "language", start: p.start, end: p.end, label: p.language, confidence: p.probability, source: model, granularity: portions.length > 1 ? "segment" : "clip" });
    } else if (duration > 0) {
      timeline.add({ kind: "language", start: 0, end: duration, label: found.language, confidence: found.probability, source: model, granularity: "clip" });
    }
    return { ...found, timeline: timeline.toJSON(), model };
  }

  /** The audio's language and its probability, by three probes and a vote. */
  async #detect(audio: Float32Array, signal?: AbortSignal): Promise<{ language: string; probability: number }> {
    if (!this.tokens || !this.tokens.isMultilingual) return { language: "en", probability: 1 };
    const answers: { language: string; probability: number }[] = [];
    for (const start of probePoints(audio.length / 16000, LANGUAGE_PROBES)) {
      const answer = await this.#probeLanguage(audio, start, signal);
      if (answer) answers.push(answer);
    }
    if (answers.length === 0) return { language: "en", probability: 0 };
    if (answers.length === 1) return answers[0]!;
    const weighted = new Map<string, number>();
    for (const a of answers) weighted.set(a.language, (weighted.get(a.language) ?? 0) + a.probability);
    const winner = [...weighted].reduce((best, entry) => (entry[1] > best[1] ? entry : best))[0];
    const agreeing = answers.filter((a) => a.language === winner);
    return { language: winner, probability: agreeing.reduce((s, a) => s + a.probability, 0) / agreeing.length };
  }

  /**
   * The language of the 30-second window starting at `start`, from one decode step; null past the
   * end.
   */
  async #probeLanguage(audio: Float32Array, start: number, signal?: AbortSignal): Promise<{ language: string; probability: number } | null> {
    const tokens = this.tokens;
    if (!tokens || !tokens.isMultilingual) return { language: "en", probability: 1 };
    const encoded = await this.#encode(audio, start);
    if (!encoded) return null;
    const byId = new Map([...tokens.languages].map(([code, id]) => [id, code]));
    const ids = [...byId.keys()].sort((a, b) => a - b);
    const out = await this.engine.generate({
      prompts: [[tokens.sot]],
      encoderOutput: encoded,
      maxNewTokens: 1,
      eosTokenId: tokens.eot,
      padTokenId: tokens.eot,
      probeFirstStep: ids,
      ...(signal ? { signal } : {}),
    });
    let total = 0;
    let best = ids[0]!;
    for (const id of ids) {
      const p = out.firstStepProbs[id] ?? 0;
      total += p;
      if (p > (out.firstStepProbs[best] ?? 0)) best = id;
    }
    return { language: byId.get(best) ?? "en", probability: total > 0 ? (out.firstStepProbs[best] ?? 0) / total : 0 };
  }

  /**
   * The language over time (spec § 2.1, `over_time`): one probe per window, neighbouring windows of
   * one language merged into portions, and a window the model is unsure of (below `minProbability`)
   * absorbed by its neighbour, so one hesitant window does not cut a portion in two.
   */
  async languageOverTime(
    audio: Float32Array,
    options: { windowS?: number; minProbability?: number; signal?: AbortSignal } = {},
  ): Promise<LanguagePortion[]> {
    const duration = audio.length / 16000;
    if (!this.tokens || !this.tokens.isMultilingual) return [{ language: "en", start: 0, end: duration, probability: 1 }];
    const step = options.windowS ?? WINDOW_SECONDS;
    const probes: LanguagePortion[] = [];
    for (let start = 0; start < duration; start += step) {
      const answer = await this.#probeLanguage(audio, start, options.signal);
      if (answer) probes.push({ ...answer, start, end: Math.min(start + step, duration) });
    }
    return mergePortions(probes, options.minProbability ?? 0.5);
  }

  /**
   * Transcribe one mono 16 kHz signal: cut into sections, then walk each section window by window,
   * carrying the previous window's tokens forward.
   */
  async transcribe(audio: Float32Array, options: TranscribeOptions = {}): Promise<TranscriptionResult> {
    return (await this.transcribeMany([audio], options))[0]!;
  }

  /**
   * Transcribe several signals together: their windows share the waves, so one decoder call serves
   * windows of different audios. The results keep the order of the input.
   */
  async transcribeMany(audios: readonly Float32Array[], options: TranscribeOptions = {}): Promise<TranscriptionResult[]> {
    const started = performance.now();
    if (options.decodingProfile !== undefined && !PROFILES.includes(options.decodingProfile)) {
      throw new ConfigurationError(
        `'${String(options.decodingProfile)}' is not a decoding profile (${PROFILES.join(", ")})`,
      );
    }
    const durations = audios.map((a) => a.length / 16000);
    const total = durations.reduce((s, d) => s + d, 0);

    const analyses = checkAnalyses(options.analyze);
    if (analyses.includes("words")) options = { ...options, wordTimestamps: true };

    const early = analyses.some((a) => AUDIO_ONLY_ANALYSES.has(a)) ? audios.map((a) => settle(this.#analyzer.analyze(a, analyses, options))) : null;
    const contentHashes = await Promise.all(audios.map((audio) => hashBytes(audio)));
    const audioIds = await Promise.all(
      audios.map(async (_, i) => options.audioIds?.[i] ?? contentHashes[i]!),
    );

    if (!this.tokens && options.language && !["en", "english"].includes(options.language.toLowerCase())) {
      throw new GenerationError(`Moonshine is English-only and cannot be pinned to '${options.language}'. Leave the language unset.`);
    }

    const run = await this.#runIds(audios, options);
    if (run.cache && run.keys) {
      const cached = await Promise.all(run.keys.map((key) => run.cache!.get(key)));
      if (cached.every((c) => c !== undefined && c !== null)) return cached as TranscriptionResult[];
    }

    const languages: { language: string; probability: number }[] = [];
    const portions: (LanguagePortion[] | null)[] = audios.map(() => null);
    for (const [i, audio] of audios.entries()) {
      if (!options.language && options.languageOverTime && this.isMultilingual) {
        portions[i] = await this.languageOverTime(audio, options.signal ? { signal: options.signal } : {});
        languages.push(dominantLanguage(portions[i]!));
        continue;
      }
      if (options.language) {
        languages.push({ language: options.language, probability: 1 });
      } else {
        options.onProgress?.({ phase: "language", ratio: 0, seconds: 0, totalSeconds: total });
        languages.push(await this.#detect(audio, options.signal));
      }
    }

    const maps: (SpeechMap | null)[] = await this.#clipSilence(audios, options);
    const decodedAudios = audios.map((audio, i) => (maps[i] ? this.#clipped.get(i)! : audio));
    this.#clipped.clear();
    const decodedDurations = decodedAudios.map((a) => a.length / 16000);

    const quality: QualityConfig = { ...DEFAULT_QUALITY, ...options.quality };
    const maxWindows = Math.max(1, options.maxWindows ?? this.waveSize);
    const infos: AudioInfo[] = audios.map((_, i) => ({ id: `audio-${i}`, duration: decodedDurations[i]! }));
    const index = new Map(infos.map((info, i) => [info.id, i]));
    const scheduler = new SectionScheduler();
    const planned = scheduler.plan(infos, {
      totalBatch: options.sections ?? sectionBudget(maxWindows, audios.length),
    });

    const sections = infos.flatMap((info, i) => {
      const spans = portions[i];
      if (!spans || spans.length < 2 || maps[i]) return planned.filter((s) => s.audioId === info.id);
      return scheduler.planSpans(info, spans, 0, { budget: planned.filter((x) => x.audioId === info.id).length });
    }).map((section, index) => ({ ...section, index }));
    const waves = new WaveScheduler(WINDOW_SECONDS);
    const states = waves.states(sections);
    if (options.initialPrompt?.trim() && this.tokens) {
      const context = this.tokenizer.encode(` ${options.initialPrompt.trim()}`, { addSpecialTokens: false });
      for (const state of states) if (state.section.start === 0) state.context = context;
    }

    const journal = run.fingerprint && run.checkpoint ? new WaveJournal(run.checkpoint, `transcribe/${run.fingerprint}`) : null;
    if (journal) await journal.restore(states);

    for (const wave of waves.waves(states, maxWindows)) {
      const decoded = await this.#wave(wave, decodedAudios, index, languages, options, quality, portions, maps);
      wave.forEach((state, i) => {
        const result = decoded[i] ?? null;
        if (result) {
          state.results.push(result);
          waves.advance(state, result);
        } else {
          waves.skip(state);
        }
      });
      await journal?.save(states);
      const done = states.reduce((s, st) => s + Math.max(0, Math.min(st.seek, st.section.end) - st.section.start), 0);
      options.onProgress?.({ phase: "decoding", ratio: total > 0 ? Math.min(1, done / total) : 1, seconds: done, totalSeconds: total });
    }

    if (options.retry && audios.length === 1) {
      await this.#retryWeakSections(states, decodedAudios, index, languages, options, quality, waves);
    }

    options.onProgress?.({ phase: "assembling", ratio: 1, seconds: total, totalSeconds: total });
    const assembled = audios.map((_, i) => {
      const windows = states.filter((st) => index.get(st.audioId) === i).flatMap((st) => st.results).sort((a, b) => a.sectionIndex - b.sectionIndex || a.windowStart - b.windowStart);
      const map = maps[i] ?? null;
      const merged = mergeSegments(
        windows
          .filter((r) => !r.isSilent)
          .map((r) =>
            this.#segmentsOf(r, decodedAudios[i]!).map((segment) => ({
              ...segment,
              avgLogprob: r.avgLogprob,
              compressionRatio: r.compressionRatio,
              noSpeechProb: r.noSpeechProb,
              temperature: r.temperature,
              speaker: null,
              words: null,
            })),
          ),
      ).map((segment, id) => ({ ...segment, id }));
      return { i, merged, map, windows };
    });

    const aligned = await Promise.all(
      assembled.map(async ({ i, merged }) =>
        options.wordTimestamps || options.alignment?.enabled ? this.#align(merged, decodedAudios[i]!, options) : merged,
      ),
    );
    const elapsedMs = performance.now() - started;
    const out = assembled.map(({ i, map, windows }) => {
      const segments = aligned[i]!.map((segment) =>
        map
          ? {
              ...segment,
              start: restoreTime(segment.start, map),
              end: restoreTime(segment.end, map),
              words: segment.words?.map((w) => ({ ...w, start: restoreTime(w.start, map), end: restoreTime(w.end, map) })) ?? null,
            }
          : segment,
      );
      const allWords = segments.flatMap((s) => s.words ?? []);
      const result: TranscriptionResult = {
        text: segments.map((s) => s.text).join(" ").replace(/\s+/g, " ").trim(),
        language: languages[i]!.language,
        languageProbability: languages[i]!.probability,
        duration: durations[i]!,
        segments,
        words: allWords.length > 0 ? allWords : null,
        speakers: null,
        speakerInfo: null,
        languagePortions: portions[i] ?? null,
        audioId: audioIds[i]!,
        contentHash: contentHashes[i]!,
        modelName: this.bundle.manifest.modelKey,
        elapsedS: elapsedMs / 1000,
        speedRatio: elapsedMs > 0 ? durations[i]! / (elapsedMs / 1000) : 0,
        metadata: { computeType: this.bundle.manifest.computeType, provider: this.backend, decodeMode: "batched", vadApplied: map !== null },
        timeline: emptyTimeline(),
        analyses: [],
        plan: this.plan(options),
        windows,
      };
      return result;
    });

    if (analyses.length > 0) {
      for (const [i, result] of out.entries()) {
        const analysis = early ? unwrap(await early[i]!) : null;
        out[i] = mergeAnalysis(result, analysis, analyses, this.bundle.manifest.modelKey, options.diarization?.assignmentStrategy);
      }
    }
    await journal?.complete();
    if (run.cache && run.keys) await Promise.all(out.map((result, i) => run.cache!.set(run.keys![i]!, result)));
    return out;
  }

  /**
   * Word timestamps for an existing result (spec § 2.1): the segments are aligned on the audio they
   * came from, as `wordTimestamps` would have done.
   */
  async align(audio: Float32Array, result: TranscriptionResult, alignment: TranscribeOptions["alignment"] = {}): Promise<TranscriptionResult> {
    const segments = await this.#align(result.segments, audio, { alignment });
    const words = segments.flatMap((s) => s.words ?? []);
    return { ...result, segments, words: words.length > 0 ? words : null };
  }

  /**
   * The plan a call with these options would run, without running anything (spec § 7): steps in
   * dependency order, each with its model, device, licence and reason.
   */
  plan(options: TranscribeOptions = {}, scope: { transcribe?: boolean } = {}): ExecutionPlan {
    const analyses = checkAnalyses(options.analyze);
    const words = options.wordTimestamps === true || options.alignment?.enabled === true || analyses.includes("words");
    const config = options.alignment ?? {};
    const ctc = config.backend === "ctc" || (config.ctcModel !== undefined && config.ctcModel !== "auto") ? (config.ctcModel && config.ctcModel !== "auto" ? config.ctcModel : DEFAULT_CTC_MODEL) : null;
    const diarization = options.diarization ?? {};
    const segmentation = diarization.segmentationModel === "" ? null : diarization.segmentationModel && diarization.segmentationModel !== "auto" ? diarization.segmentationModel : DEFAULT_SEGMENTATION_MODEL;
    const precision = this.precision;
    return buildPlan({
      model: this.bundle.manifest.modelKey,
      device: this.backend,
      precision: precision.encoder === precision.decoder ? precision.encoder : `encoder ${precision.encoder}, decoder ${precision.decoder}`,
      languageKnown: Boolean(options.language) || !this.isMultilingual,
      vad: shouldUseVad(paramsOf(this.bundle.manifest.modelKey), options.vad ?? "auto"),
      analyses,
      wordTimestamps: words,
      ctcModel: ctc,
      diarization: {
        embeddingModel: diarization.embeddingModel && diarization.embeddingModel !== "auto" ? diarization.embeddingModel : DEFAULT_EMBEDDING_MODEL,
        segmentationModel: segmentation,
      },
      resolution: this.resolution,
      ...(options.licenses ? { licenses: options.licenses } : {}),
      ...(options.network ? { network: options.network } : {}),
      ...(scope.transcribe === false ? { transcribe: false, vad: false } : {}),
      ...(options.emotion?.model ? { emotionModel: options.emotion.model } : {}),
      ...(options.events?.model ? { eventsModel: options.events.model } : {}),
    });
  }

  /**
   * The analyses alone, without a transcript (spec § 2.1): `speech`, `profile`, `language`,
   * `speakers`. `words` needs a transcript and is refused.
   */
  async analyze(audio: Float32Array, analyses: readonly string[], options: TranscribeOptions = {}): Promise<AudioAnalysisResult> {
    const checked = checkAnalyses(analyses);
    if (checked.includes("words")) throw new ConfigurationError("'words' times the words of a transcript: call transcribe() with analyze: [\"words\"]");
    const duration = audio.length / 16000;
    const [audioOnly, language] = await Promise.all([
      this.#analyzer.analyze(audio, checked, options),
      checked.includes("language")
        ? this.detectLanguage(audio, {
            overTime: Boolean(options.interval || options.languageOverTime),
            ...(options.interval ? { interval: options.interval } : {}),
            ...(options.signal ? { signal: options.signal } : {}),
          })
        : Promise.resolve(null),
    ]);
    const timeline = new Timeline();
    for (const note of audioOnly.annotations) timeline.add(note as Parameters<Timeline["add"]>[0]);
    for (const { id: _id, ...annotation } of language?.timeline.annotations ?? []) timeline.add(annotation);
    const contentHash = await hashBytes(audio);
    return {
      audioId: options.audioIds?.[0] ?? contentHash,
      contentHash,
      duration,
      timeline: timeline.toJSON(),
      analyses: checked,
      plan: this.plan({ ...options, analyze: checked }, { transcribe: false }),
      speakers: audioOnly.speakers,
      speakerInfo: audioOnly.speakerInfo,
    };
  }

  /**
   * The audio-only analyses of `audio`, to merge with a transcript made elsewhere
   * (`mergeAnalysis`).
   */
  async analyzeAudio(audio: Float32Array, analyses: readonly string[], options: TranscribeOptions = {}): Promise<AudioOnlyAnalysis> {
    return this.#analyzer.analyze(audio, checkAnalyses(analyses), options);
  }

  #aligner: ForcedAligner | null = null;
  #alignerFailed = false;

  /**
   * Word spans, by the best method available: CTC forced alignment with a second model when one is
   * configured (`alignment.ctcModel`, or `backend: "ctc"`), else a proportional split of each
   * segment — which is not an alignment, and says so. (DTW over cross-attention needs a graph that
   * exports it; the published ones do not.)
   */
  async #align(segments: Segment[], signal: Float32Array, options: TranscribeOptions): Promise<Segment[]> {
    const config = options.alignment ?? {};
    const wantsCtc = config.backend === "ctc" || (config.ctcModel !== undefined && config.ctcModel !== "auto");
    if (wantsCtc && !this.#alignerFailed) {
      try {
        this.#aligner ??= await ForcedAligner.load({
          ort: this.ort,
          Tensor: this.Tensor,
          ...(config.ctcModel && config.ctcModel !== "auto" ? { model: config.ctcModel } : {}),
        });
      } catch (cause) {
        this.#alignerFailed = true;
        this.#logger.warn(`the CTC aligner could not be loaded (${String(cause)}); word timestamps fall back to a proportional split`);
      }
    }
    const aligner = this.#aligner;
    if (aligner) {
      let alignedAny = false;
      const out: Segment[] = [];
      for (const segment of segments) {
        const chunk = signal.subarray(Math.max(0, Math.trunc(segment.start * 16000)), Math.min(signal.length, Math.trunc(segment.end * 16000)));
        let words: Word[] | null = null;
        if (chunk.length > 0 && segment.text.trim()) {
          try {
            words = mergePunctuation(await aligner.align(chunk, segment.text, segment.start));
          } catch {
            // Not fatal: carry on.
          }
        }
        if (words && words.length > 0) alignedAny = true;
        out.push({ ...segment, words: words && words.length > 0 ? words : null });
      }
      if (alignedAny) return out;
    }
    this.#logger.warn(
      "no CTC model is configured: word timestamps are a proportional split of each segment, not an alignment. " +
        "Set alignment.ctcModel (or backend: \"ctc\") to get real ones.",
    );
    return segments.map((segment) => {
      const words = mergePunctuation(proportionalWords(segment));
      return { ...segment, words: words.length > 0 ? words : null };
    });
  }

  get #logger() {
    return this.#options?.logger ?? defaultLogger;
  }

  /** Re-decode the sections that scored badly, at a more accurate rung. */
  async #retryWeakSections(
    states: SectionState[],
    audios: readonly Float32Array[],
    index: ReadonlyMap<string, number>,
    languages: readonly { language: string }[],
    options: TranscribeOptions,
    quality: QualityConfig,
    waves: WaveScheduler,
  ): Promise<void> {
    const loadOptions = this.#options;
    if (!loadOptions) return;
    const device = this.backend === "webgpu" ? "webgpu" : "wasm";
    const free = device === "wasm" ? wasmRuntime().heapCeilingMb - totalSizeMb(this.bundle.manifest) * 1.25 : null;
    const upgraded = retryRung(
      this.modelName,
      this.precision,
      device,
      languages[0]?.language ?? null,
      free,
      options.measurements ?? WEB_MEASUREMENTS,
    );
    if (upgraded === null) return;
    const weak = states.filter((state) => state.results.length > 0 && meanLogprob(state.results) < RETRY_LOGPROB);
    if (weak.length === 0) return;

    const keep = new Map(weak.map((state) => [state, [...state.results]]));
    let second: AudioTranscriber | null = null;
    try {
      second = await AudioTranscriber.load({ ...loadOptions, model: this.modelName, computeType: upgraded, backends: device });
      for (const state of weak) {
        const before = meanLogprob(state.results);
        state.seek = state.section.start;
        state.results = [];
        state.context = [];
        state.done = false;
        while (!state.done) {
          const [result] = await second.#wave([state], audios, index, languages, options, quality);
          if (!result) break;
          state.results.push(result);
          waves.advance(state, result);
        }
        if (meanLogprob(state.results) < before) state.results = keep.get(state)!;
      }
    } catch {
      for (const [state, results] of keep) state.results = results;
    } finally {
      await second?.close();
    }
  }

  /**
   * The ids a run is filed under: each audio's content with the settings (result cache), and all of
   * them together (journal). Only computed when a cache or a checkpoint asks for them: hashing an
   * hour of audio is not free.
   */
  async #runIds(audios: readonly Float32Array[], options: TranscribeOptions) {
    const checkpoint = options.checkpoint === true ? new IndexedDbStore() : options.checkpoint || null;
    const cache = options.cache === true ? new ResultCache(new IndexedDbStore(), "ynonnx-transcribe") : options.cache ? new ResultCache(options.cache, "ynonnx-transcribe") : null;
    if (!checkpoint && !cache) return { checkpoint, cache, fingerprint: null, keys: null };
    const {
      signal: _signal,
      onProgress: _progress,
      checkpoint: _checkpoint,
      cache: _cache,
      ...settings
    } = options;
    const settingsId = await hashParams({ model: this.bundle.manifest.modelKey, computeType: this.bundle.manifest.computeType, ...settings });
    const contents = await Promise.all(audios.map((audio) => hashBytes(audio)));
    const keys = await Promise.all(contents.map((content) => combineIds([content, settingsId])));
    const fingerprint = await combineIds([...contents, settingsId]);
    return { checkpoint, cache, fingerprint, keys };
  }

  readonly #clipped = new Map<number, Float32Array>();

  /**
   * Detect speech and cut the silence out, per audio, as Python does: spans grouped to window size
   * and padded; the cut kept only when it removes at least half a second. Returns each audio's map
   * back to its timeline, or null where nothing was cut. A VAD that fails leaves the audio whole.
   */
  async #clipSilence(audios: readonly Float32Array[], options: TranscribeOptions): Promise<(SpeechMap | null)[]> {
    if (!shouldUseVad(paramsOf(this.bundle.manifest.modelKey), options.vad ?? "auto")) return audios.map(() => null);
    const config = options.vadConfig ?? {};
    try {
      this.#vad ??= await VAD.load({ ort: this.ort, Tensor: this.Tensor, config });
    } catch {
      return audios.map(() => null);
    }
    const maps: (SpeechMap | null)[] = [];
    for (const [i, audio] of audios.entries()) {
      const duration = audio.length / 16000;
      const segments = await this.#vad.detectSpeech(audio);
      const grouped = applyPadding(
        groupSpeech(segments, config.groupMaxDurationS ?? 28, config.groupMaxGapS ?? 2),
        config.speechPadMs ?? DEFAULT_VAD.speechPadMs,
        duration,
      );
      const { clipped, map } = clipToSpeech(audio, grouped);
      if (map.length > 0 && clipped.length < audio.length - 8000) {
        this.#clipped.set(i, clipped);
        maps.push(map);
      } else {
        maps.push(null);
      }
    }
    return maps;
  }

  /**
   * One wave: every window whose prompt has the same length and the same rules is decoded in ONE
   * call; a window the quality thresholds reject is then retried alone, up the temperature ladder.
   */
  async #wave(
    wave: readonly SectionState[],
    audios: readonly Float32Array[],
    index: ReadonlyMap<string, number>,
    languages: readonly { language: string }[],
    options: TranscribeOptions,
    quality: QualityConfig,
    portions: readonly (LanguagePortion[] | null)[] = [],
    maps: readonly (SpeechMap | null)[] = [],
  ): Promise<(WindowResult | null)[]> {
    if (options.signal?.aborted) throw new YnONNXAudioError("transcription was cancelled");
    const ladder =
      options.temperature === undefined || options.temperature === "auto" ? TEMPERATURE_SCALE : [options.temperature];
    const out: (WindowResult | null)[] = wave.map(() => null);

    const plans = wave.map((state) => {
      const at = index.get(state.audioId)!;

      const language = (maps[at] ? null : portionAt(portions[at], state.seek)) ?? languages[at]!.language;
      const role = options.decodingProfile ?? this.resolution?.decisions.decodingProfile.value ?? "default";
      const measured = profileFor(this.bundle.manifest.modelKey, role, language);
      const carry = options.conditionOnPreviousText ?? measured.conditionOnPreviousText ?? true;

      const prompt = this.tokens
        ? buildPrompt(this.tokens, {
            language,
            ...(options.task ? { task: options.task } : {}),
            context: carry ? state.context : [],
            maxContext: options.maxContextTokens ?? DEFAULT_MAX_CONTEXT,
          })
        : [MOONSHINE_BOS];
      const ngram = options.noRepeatNgramSize ?? measured.noRepeatNgramSize ?? 0;
      return { state, audio: audios[at]!, prompt, ngram };
    });

    const buckets = new Map<string, number[]>();
    plans.forEach((plan, i) => {
      const key = `${plan.prompt.length}|${plan.ngram}`;
      buckets.set(key, [...(buckets.get(key) ?? []), i]);
    });

    for (const members of buckets.values()) {
      const live = members.filter((i) => this.#hasAudio(plans[i]!.audio, plans[i]!.state.seek));
      if (live.length === 0) continue;
      const encoded = await this.#encodeMany(live.map((i) => [plans[i]!.audio, plans[i]!.state.seek, windowEnd(plans[i]!.state)] as const));
      const first = await this.#generate(
        encoded,
        live.map((i) => plans[i]!.prompt),
        plans[live[0]!]!.ngram,
        ladder[0] ?? 0,
        options,
      );
      for (const [row, i] of live.entries()) {
        const plan = plans[i]!;
        let score = await this.#score(first, row);
        let temperature = ladder[0] ?? 0;

        for (const next of ladder.slice(1)) {
          if (!isDegenerate(score, quality)) break;
          const single = await this.#encodeMany([[plan.audio, plan.state.seek, windowEnd(plan.state)]]);
          score = await this.#score(await this.#generate(single, [plan.prompt], plan.ngram, next, options), 0);
          temperature = next;
        }
        out[i] = {
          audioId: plan.state.audioId,
          sectionIndex: plan.state.section.index,
          windowStart: plan.state.seek,
          tokens: score.ids,
          text: score.text,
          avgLogprob: score.avgLogprob,
          compressionRatio: score.compressionRatio,
          noSpeechProb: score.noSpeechProb,
          isSilent: isSilent(score.noSpeechProb, score.avgLogprob, quality),
          lastSegmentEnd: score.lastEnd === null ? null : plan.state.seek + score.lastEnd,
          temperature,
        };
      }
    }
    return out;
  }

  #hasAudio(audio: Float32Array, seek: number): boolean {
    return Math.round(seek * 16000) < audio.length;
  }

  /** The encoder's output for windows given as (audio, start in seconds), batched. */
  async #encodeMany(windows: readonly (readonly [Float32Array, number, (number | undefined)?])[]): Promise<TensorLike> {
    const melConfig = this.mel;
    if (melConfig === null) {
      const slices = windows.map(([audio, seek, end]) => {
        const offset = Math.round(seek * 16000);
        const slice = audio.subarray(offset, Math.min(offset + WINDOW_SAMPLES, audio.length, end === undefined ? Infinity : Math.round(end * 16000)));
        return slice.length > 0 ? slice : new Float32Array(MOONSHINE_SAMPLES_PER_FRAME * 2);
      });
      const longest = Math.max(...slices.map((x) => x.length));
      const data = new Float32Array(slices.length * longest);
      slices.forEach((x, i) => data.set(x, i * longest));
      const input = this.encoder.inputs[0]?.name ?? "input_values";
      return this.engine.encode({ [input]: new this.Tensor("float32", data, [slices.length, longest]) });
    }
    const mels = windows.map(([audio, seek, end]) => {
      const offset = Math.round(seek * 16000);
      return logMel(audio.subarray(offset, Math.min(offset + WINDOW_SAMPLES, audio.length, end === undefined ? Infinity : Math.round(end * 16000))), melConfig);
    });
    const { nMels, frames } = mels[0]!;
    const data = new Float32Array(mels.length * nMels * frames);
    mels.forEach((mel, i) => data.set(mel.data, i * nMels * frames));
    const name = this.encoder.inputs[0]?.name ?? "input_features";
    return this.engine.encode({ [name]: new this.Tensor("float32", data, [mels.length, nMels, frames]) });
  }

  #generate(encoded: TensorLike, prompts: number[][], ngram: number, temperature: number, options: TranscribeOptions) {
    if (!this.tokens) {
      const ceiling = Math.min(MAX_TARGET_POSITIONS, MOONSHINE_MAX_POSITIONS);
      return this.engine.generate({
        prompts,
        encoderOutput: encoded,
        noRepeatNgramSize: ngram,
        temperature,
        maxPositions: MOONSHINE_MAX_POSITIONS,
        maxNewTokens: Math.max(1, ceiling - (prompts[0]?.length ?? 0)),
        eosTokenId: MOONSHINE_EOS,
        padTokenId: MOONSHINE_EOS,
        ...(options.signal ? { signal: options.signal } : {}),
      });
    }
    const noSpeech = this.tokens.noSpeech;
    return this.engine.generate({
      prompts,
      encoderOutput: encoded,
      processors: [
        new TimestampRules(this.tokens, {
          maxInitialTimestamp: options.maxInitialTimestamp === undefined ? 1.0 : options.maxInitialTimestamp,
          withoutTimestamps: options.withoutTimestamps ?? false,
        }),
      ],
      noRepeatNgramSize: ngram,
      temperature,
      maxPositions: MAX_TARGET_POSITIONS,
      maxNewTokens: MAX_TARGET_POSITIONS,
      eosTokenId: this.tokens.eot,
      padTokenId: this.tokens.eot,
      ...(noSpeech !== null ? { probeFirstStep: [noSpeech] } : {}),
      ...(options.signal ? { signal: options.signal } : {}),
    });
  }

  async #score(out: GenerationOutput, row: number) {
    const ids = out.sequences[row] ?? [];
    if (!this.tokens) {
      const text = this.tokenizer
        .decode(ids.filter((t) => t !== MOONSHINE_BOS && t !== MOONSHINE_EOS), { skipSpecialTokens: true })
        .trim();
      return { ids, text, lastEnd: null, avgLogprob: out.avgLogprob[row] ?? 0, compressionRatio: await compressionRatio(text), noSpeechProb: 0 };
    }
    const parsed = extractSegments(ids, this.tokens, this.tokenizer, 0);
    const noSpeech = this.tokens.noSpeech;
    return {
      ids,
      text: parsed.text,
      lastEnd: parsed.lastEnd,
      avgLogprob: out.avgLogprob[row] ?? 0,
      compressionRatio: await compressionRatio(parsed.text),
      noSpeechProb: noSpeech !== null ? (out.firstStepProbsPerRow[row]?.[noSpeech] ?? 0) : 0,
    };
  }

  /**
   * A window's segments: Whisper's from its timestamp tokens; Moonshine's is one segment spanning
   * the audio the window held.
   */
  #segmentsOf(result: WindowResult, audio: Float32Array): Segment[] {
    if (this.tokens) return extractSegments(result.tokens, this.tokens, this.tokenizer, result.windowStart).segments;
    if (!result.text || result.tokens.length === 0) return [];
    const end = Math.min(result.windowStart + WINDOW_SECONDS, audio.length / 16000);
    return [{ start: result.windowStart, end, text: result.text, tokens: [...result.tokens] }];
  }

  /**
   * Label an existing transcription with speakers. Separate from `transcribe`, as in Python:
   * diarization loads two more graphs, closed again here.
   */
  async diarize(audio: Float32Array, result: TranscriptionResult, config: DiarizationConfig = {}): Promise<TranscriptionResult> {
    const { speakers, info } = await this.#analyzer.diarize(audio, config);
    return { ...result, segments: assignSpeakers(result.segments, speakers, config.assignmentStrategy ?? "longest"), speakers, speakerInfo: info };
  }

  /**
   * Open a streaming session (spec § 4): feed it audio as it arrives, read its updates, close it
   * for the final transcript. `simple` uses this model; `double` loads a fast one beside it (this
   * one corrects); `post` re-runs this one, or `finalModel`, on the whole recording at the end.
   */
  async stream(options: TranscriberStreamOptions = {}): Promise<StreamSession> {
    const mode = options.mode ?? "simple";
    const load = (model: string) => AudioTranscriber.load({ ort: this.ort, Tensor: this.Tensor, ...(options.loadOptions ?? {}), model } as TranscriberOptions);
    if (mode === "double") {
      const fast = await load(!options.fastModel || options.fastModel === "auto" ? "tiny" : options.fastModel);
      return new StreamSession(this, options, fast, () => fast.close());
    }
    if (mode === "post" && options.finalModel) {
      const final = await load(options.finalModel);
      return new StreamSession(this, options, final, () => final.close());
    }
    return new StreamSession(this, options);
  }

  /**
   * Enrol a known speaker from a sample of their voice (a few seconds of speech): diarization then
   * names their turns. Replaces an earlier print of the same name.
   */
  async enrollSpeaker(name: string, audio: Float32Array, config: DiarizationConfig = {}): Promise<void> {
    if (!name.trim()) throw new EnrollmentError("a speaker needs a name");
    await this.#analyzer.enroll(name, audio, config);
  }

  forgetSpeaker(name: string): void {
    if (!this.#analyzer.forget(name)) throw new EnrollmentError(`'${name}' is not enrolled (enrolled: ${this.#analyzer.enrolled.join(", ") || "nobody"})`);
  }

  /** The enrolled names. */
  speakers(): string[] {
    return this.#analyzer.enrolled;
  }

  /** The prints, serializable: a server keeps them and hands them back to the next page. */
  speakerPrints(): Record<string, number[]> {
    return this.#analyzer.prints();
  }

  /** Restore prints kept elsewhere (the inverse of `speakerPrints`). */
  loadSpeakerPrints(prints: Readonly<Record<string, ArrayLike<number>>>): void {
    this.#analyzer.loadPrints(prints);
  }

  async close(): Promise<void> {
    await this.#aligner?.close();
    this.#aligner = null;
    await this.#vad?.close();
    this.#vad = null;
    await this.#analyzerInstance?.close();
    await this.bundle.close();
  }
}

/**
 * The parameter count of a checkpoint, from its name (`whisper-large-v3`,
 * `org/whisper-tiny_int8`…).
 */
export function paramsOf(modelKey: string): number | null {
  const last = modelKey.split("/").pop() ?? modelKey;
  const name = (last.split("_", 1)[0] ?? last).replace(/^whisper-/, "");
  return WHISPER_PARAMS_M[name] ?? null;
}

/** Window starts spread over the audio, never past its end (Python's `_probe_points`). */
export function probePoints(duration: number, probes: number): number[] {
  const count = Math.max(1, Math.trunc(probes));
  if (duration <= WINDOW_SECONDS || count === 1) return [0];
  const usable = Math.max(0, duration - WINDOW_SECONDS);
  return Array.from({ length: count }, (_, i) => Math.round((usable * i * 1000) / (count - 1)) / 1000);
}

type Settled<T> = { ok: true; value: T } | { ok: false; error: unknown };

/** A promise that never rejects unobserved: started early, awaited later. */
function settle<T>(promise: Promise<T>): Promise<Settled<T>> {
  return promise.then(
    (value) => ({ ok: true as const, value }),
    (error: unknown) => ({ ok: false as const, error }),
  );
}

function unwrap<T>(settled: Settled<T>): T {
  if (!settled.ok) throw settled.error;
  return settled.value;
}

/** Where a window must stop: the section's limit, when the audio changes language after it. */
function windowEnd(state: { section: { limit?: number } }): number | undefined {
  return state.section.limit;
}
