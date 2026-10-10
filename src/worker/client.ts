/** Driving the transcription worker from the page. */

import { WorkerClient, type Logger } from "@youni/onnxweb-core";

import { AUDIO_ONLY_ANALYSES, mergeAnalysis, type AudioOnlyAnalysis } from "../analysis/analyzer.js";
import { checkAnalyses } from "../timeline.js";

import type { AudioAnalysisResult, LanguageDetectionResult, LanguagePortion, TranscribeOptions, TranscriberStreamOptions, TranscriptionResult } from "../whisper/transcriber.js";
import type { StreamResult, StreamUpdate } from "../streaming.js";
import type { DiarizationConfig } from "../diarization/diarizer.js";
import type { ExecutionPlan } from "../plan.js";
import type { Resolution } from "../whisper/resolver.js";
import type { WorkerLoadRequest } from "./entry.js";

type Options = Omit<TranscribeOptions, "signal" | "onProgress" | "checkpoint" | "cache"> & { checkpoint?: boolean; cache?: boolean };

/** A copy of the samples, as a buffer to transfer: the caller keeps its own array. */
function bufferOf(audio: Float32Array): ArrayBuffer {
  return audio.buffer.slice(audio.byteOffset, audio.byteOffset + audio.byteLength) as ArrayBuffer;
}

/**
 * The audio-only analyses that depend on one another, kept together: a group per worker, the groups
 * at once (emotion reads the speakers' turns).
 */
const ANALYSIS_GROUPS: readonly (readonly string[])[] = [["speakers", "emotion"], ["events", "music"], ["speech", "profile"]];

/** The groups' analyses as one. */
function joinAnalyses(parts: AudioOnlyAnalysis[]): AudioOnlyAnalysis {
  return {
    analyses: parts.flatMap((p) => p.analyses),
    annotations: parts.flatMap((p) => p.annotations),
    speakers: parts.find((p) => p.speakers)?.speakers ?? null,
    speakerInfo: parts.find((p) => p.speakerInfo)?.speakerInfo ?? null,
  };
}

export interface WorkerTranscriberOptions extends WorkerLoadRequest {
  /** Where the built worker file lives. */
  url: string;
  logger?: Logger;
  /**
   * Run the analyses that need the audio only (speakers, emotion, events, music, speech, profile)
   * in a second worker, at the same time as the transcription: two runtimes, truly in parallel. On
   * by default; off runs everything in one worker (less memory).
   */
  parallel?: boolean;
}

export class WorkerTranscriber {
  readonly #client: WorkerClient;
  /** A worker per group of audio-only analyses (`ANALYSIS_GROUPS`); made on first need. */
  readonly #analysis = new Map<string, WorkerClient>();

  constructor(private readonly options: WorkerTranscriberOptions) {
    this.#client = new WorkerClient({
      url: options.url,
      name: "ynonnxweb-audio",
      ...(options.logger ? { logger: options.logger } : {}),
    });
  }

  /**
   * Load the model without transcribing — useful for a progress bar (`{ loading: FetchProgress }`).
   */
  async load(onProgress?: (progress: unknown) => void): Promise<{ backend: string; melBands: number; resolution: Resolution | null }> {
    return this.#client.call("load", this.#request(), onProgress ? { onProgress } : {});
  }

  /** What `auto` would decide, without loading. */
  async resolve(): Promise<Resolution | null> {
    return this.#client.call("resolve", this.#request());
  }

  async plan(options: Options = {}): Promise<ExecutionPlan> {
    return this.#client.call("plan", { ...this.#request(), options });
  }

  async analyze(audio: Float32Array, analyses: string[], options: Options = {}): Promise<AudioAnalysisResult> {
    const buffer = bufferOf(audio);
    return this.#client.call("analyze", { ...this.#request(), audio: buffer, analyses, options }, { transfer: [buffer] });
  }

  async diarize(audio: Float32Array, result: TranscriptionResult, config: DiarizationConfig = {}): Promise<TranscriptionResult> {
    const buffer = bufferOf(audio);
    return this.#client.call("diarize", { ...this.#request(), audio: buffer, result, config }, { transfer: [buffer] });
  }

  async align(audio: Float32Array, result: TranscriptionResult, options: Options = {}): Promise<TranscriptionResult> {
    const buffer = bufferOf(audio);
    return this.#client.call("align", { ...this.#request(), audio: buffer, result, options }, { transfer: [buffer] });
  }

  async detectLanguage(audio: Float32Array, options: { overTime?: boolean; interval?: number } = {}): Promise<LanguageDetectionResult> {
    const buffer = bufferOf(audio);
    return this.#client.call("detect_language", { ...this.#request(), audio: buffer, options }, { transfer: [buffer] });
  }

  async languageOverTime(audio: Float32Array): Promise<LanguagePortion[]> {
    const buffer = bufferOf(audio);
    return this.#client.call("language_over_time", { ...this.#request(), audio: buffer }, { transfer: [buffer] });
  }

  async enrollSpeaker(name: string, audio: Float32Array): Promise<string[]> {
    const buffer = bufferOf(audio);
    return this.#client.call("enroll_speaker", { ...this.#request(), audio: buffer, name }, { transfer: [buffer] });
  }

  async forgetSpeaker(name: string): Promise<string[]> {
    return this.#client.call("forget_speaker", { ...this.#request(), name });
  }

  /** The enrolled prints; with `prints`, restores them first. */
  async speakerPrints(prints?: Record<string, number[]>): Promise<Record<string, number[]>> {
    return this.#client.call("speaker_prints", { ...this.#request(), ...(prints ? { prints } : {}) });
  }

  /** Transcribe off the main thread. */
  async transcribe(
    audio: Float32Array,
    options: Options = {},
    onProgress?: (progress: unknown) => void,
    signal?: AbortSignal,
  ): Promise<TranscriptionResult> {
    const analyses = checkAnalyses(options.analyze as string[] | undefined);
    const apart = this.options.parallel !== false ? analyses.filter((a) => AUDIO_ONLY_ANALYSES.has(a)) : [];
    if (apart.length === 0) {
      const buffer = bufferOf(audio);
      return this.#client.call(
        "transcribe",
        { ...this.#request(), audio: buffer, options },
        { transfer: [buffer], ...(onProgress ? { onProgress } : {}), ...(signal ? { signal } : {}) },
      );
    }

    const prints = await this.speakerPrints().catch(() => ({}));
    const forTranscript = { ...options, analyze: analyses.filter((a) => !AUDIO_ONLY_ANALYSES.has(a)) };
    const transcriptBuffer = bufferOf(audio);
    const [result, analysis] = await Promise.all([
      this.#client.call<TranscriptionResult>(
        "transcribe",
        { ...this.#request(), audio: transcriptBuffer, options: forTranscript },
        { transfer: [transcriptBuffer], ...(onProgress ? { onProgress } : {}), ...(signal ? { signal } : {}) },
      ),
      Promise.all(
        ANALYSIS_GROUPS.map((group) => apart.filter((a) => group.includes(a)))
          .filter((wanted) => wanted.length > 0)
          .map((wanted) => {
            const key = wanted.join("+");
            let worker = this.#analysis.get(key);
            if (!worker) {
              worker = new WorkerClient({ url: this.options.url, name: `ynonnxweb-audio-${key}`, ...(this.options.logger ? { logger: this.options.logger } : {}) });
              this.#analysis.set(key, worker);
            }
            const buffer = bufferOf(audio);
            return worker.call<AudioOnlyAnalysis>(
              "analyze_audio",
              { audio: buffer, analyses: wanted, options, prints, ...(this.options.wasmPaths ? { wasmPaths: this.options.wasmPaths } : {}) },
              { transfer: [buffer], ...(onProgress ? { onProgress } : {}), ...(signal ? { signal } : {}) },
            );
          }),
      ).then(joinAnalyses),
    ]);
    return mergeAnalysis(result, analysis, analyses, result.modelName, options.diarization?.assignmentStrategy) as TranscriptionResult;
  }

  /**
   * A live session in the worker: `feed` it 16 kHz audio as it arrives, `onUpdate` sees every
   * revision, `close` ends it with the final transcript. Nothing of the decoding runs on the page.
   */
  async stream(
    options: Omit<TranscriberStreamOptions, "onUpdate" | "signal" | "loadOptions">,
    onUpdate: (update: StreamUpdate) => void,
    onProgress?: (progress: unknown) => void,
  ): Promise<{ streamId: string; backend: string; feed(chunk: Float32Array): void; close(): Promise<StreamResult | null> }> {
    const key = `${Date.now().toString(36)}-${Math.random().toString(36).slice(2)}`;
    let opened: (value: { streamId: string; backend: string }) => void = () => undefined;
    const ready = new Promise<{ streamId: string; backend: string }>((resolve) => (opened = resolve));
    const finished = this.#client.call<StreamResult>("stream", { ...this.#request(), key, stream: options }, {
      onProgress: (value: unknown) => {
        const v = value as { update?: StreamUpdate; opened?: { streamId: string; backend: string } };
        if (v.update) onUpdate(v.update);
        else if (v.opened) opened(v.opened);
        else onProgress?.(value);
      },
    });

    const info = await Promise.race([ready, finished.then(() => ({ streamId: "", backend: "" }))]);
    return {
      ...info,
      feed: (chunk) => {
        const buffer = bufferOf(chunk);
        void this.#client.call("stream_feed", { key, audio: buffer }, { transfer: [buffer] });
      },
      close: async () => {
        await this.#client.call("stream_close", { key });
        return finished;
      },
    };
  }

  async close(): Promise<void> {
    try {
      await this.#client.call("close", null);
      await Promise.all([...this.#analysis.values()].map((w) => w.call("close", null).catch(() => undefined)));
    } finally {
      this.#client.terminate();
      for (const w of this.#analysis.values()) w.terminate();
      this.#analysis.clear();
    }
  }

  #request(): WorkerLoadRequest {
    const { model, backends, resolverBase, endpoint, computeType, profile, language, hub, wasmPaths } = this.options;
    return {
      model,
      ...(hub !== undefined ? { hub } : {}),
      ...(wasmPaths !== undefined ? { wasmPaths } : {}),
      ...(backends !== undefined ? { backends } : {}),
      ...(resolverBase !== undefined ? { resolverBase } : {}),
      ...(endpoint !== undefined ? { endpoint } : {}),
      ...(computeType !== undefined ? { computeType } : {}),
      ...(profile !== undefined ? { profile } : {}),
      ...(language !== undefined ? { language } : {}),
    };
  }
}
