/** The transcription worker: everything heavy, off the main thread. */

import { registerTask, serve } from "@youni/onnxweb-core";

import { AudioAnalyzer, type AudioOnlyAnalysis } from "../analysis/analyzer.js";
import { AudioTranscriber, type TranscriberOptions, type TranscriberStreamOptions } from "../whisper/transcriber.js";
import type { StreamResult, StreamSession } from "../streaming.js";
import { checkAnalyses } from "../timeline.js";
import type { LanguageDetectionResult, LanguagePortion, TranscribeOptions, TranscriptionResult } from "../whisper/transcriber.js";
import type { DiarizationConfig } from "../diarization/diarizer.js";
import type { ExecutionPlan } from "../plan.js";
import type { Resolution } from "../whisper/resolver.js";
import type { HalfPrecision } from "../whisper/published.js";

export interface WorkerLoadRequest {
  /** Everything `AudioTranscriber.load` takes, minus what cannot cross a message. */
  model: TranscriberOptions["model"];
  backends?: unknown;
  /**
   * A server's model mount (`/ynonnx/models/`); published checkpoints come from the Hub otherwise.
   */
  resolverBase?: string;
  /** The Hub, or a mirror of it, for published checkpoints. */
  endpoint?: string;
  /**
   * A Hub repository described in plain data (a resolver object cannot cross a message): its repo,
   * and the folder of each precision when it publishes one folder per variant.
   */
  hub?: { repo: string; subfolder?: string; variants?: Record<string, string>; manifest?: boolean };
  /**
   * Where onnxruntime-web's .wasm files are served. Bundled into this worker, its own guess (beside
   * the worker script) is wrong whenever the host serves them elsewhere, which a bundler usually
   * does.
   */
  wasmPaths?: string;
  computeType?: HalfPrecision;
  profile?: string;
  language?: string | null;
}

type Options = Omit<TranscribeOptions, "signal" | "onProgress" | "checkpoint" | "cache"> & { checkpoint?: boolean; cache?: boolean };

export interface WorkerAudioRequest extends WorkerLoadRequest {
  audio: ArrayBuffer;
  options?: Options;
}

/** One transcriber, kept between calls. */
let current: { key: string; transcriber: AudioTranscriber } | null = null;

/**
 * The library's log, sent to the page as progress (`{ log: { level, message } }`) of whichever task
 * is running: what ran where, and for how long, is visible there rather than in a worker console
 * nobody opens.
 */
let report: ((value: unknown) => void) | null = null;
const forward = (level: "info" | "warn" | "error") => (message: string) => report?.({ log: { level, message } });
const workerLogger = { debug: () => undefined, info: forward("info"), warn: forward("warn"), error: forward("error") };

async function transcriberFor(request: WorkerLoadRequest, onProgress?: (progress: unknown) => void): Promise<AudioTranscriber> {
  const key = JSON.stringify([
    request.model,
    request.backends,
    request.resolverBase,
    request.endpoint,
    request.hub,
    request.wasmPaths,
    request.computeType,
    request.profile,
    request.language,
  ]);
  if (current?.key === key) return current.transcriber;

  await current?.transcriber.close();
  current = null;

  const ort = await import("onnxruntime-web");
  const { huggingFaceResolver, serverResolver } = await import("@youni/onnxweb-core");
  const hub = request.hub;

  const transcriber = await AudioTranscriber.load({
    ort: ort as never,
    Tensor: ort.Tensor as never,
    model: request.model,
    logger: workerLogger as never,
    ...(request.wasmPaths ? { wasmPaths: request.wasmPaths } : {}),
    ...(request.backends ? { backends: request.backends as never } : {}),
    ...(request.computeType ? { computeType: request.computeType } : {}),
    ...(request.profile ? { profile: request.profile } : {}),
    ...(request.language !== undefined ? { language: request.language } : {}),
    ...(request.endpoint ? { endpoint: request.endpoint } : {}),
    ...(request.resolverBase ? { resolver: serverResolver({ base: request.resolverBase }) } : {}),
    ...(hub
      ? {
          resolver: huggingFaceResolver({
            repo: hub.repo,
            ...(hub.subfolder !== undefined ? { subfolder: hub.subfolder } : {}),
            ...(hub.variants ? { variants: hub.variants } : {}),
            manifest: hub.manifest ?? true,
            ...(request.endpoint ? { endpoint: request.endpoint } : {}),
          }),
        }
      : {}),
    ...(onProgress ? { onProgress: (p: unknown) => onProgress({ loading: p }) } : {}),
  });
  current = { key, transcriber };
  return transcriber;
}

registerTask<WorkerLoadRequest, { backend: string; melBands: number; resolution: Resolution | null }>(
  "load",
  async (request, helpers) => {
    report = (p) => helpers.progress(p);
    const transcriber = await transcriberFor(request, (p) => helpers.progress(p));
    return { backend: transcriber.backend, melBands: transcriber.melBands, resolution: transcriber.resolution };
  },
);

/** What `auto` would decide, without loading anything. */
registerTask<WorkerLoadRequest, Resolution | null>("resolve", async (request) => {
  const ort = await import("onnxruntime-web");
  return AudioTranscriber.resolve({ ort: ort as never, Tensor: ort.Tensor as never, ...request } as TranscriberOptions);
});

registerTask<WorkerLoadRequest & { options?: Options }, ExecutionPlan>("plan", async (request) =>
  (await transcriberFor(request)).plan(request.options ?? {}),
);

registerTask<WorkerAudioRequest & { analyses: string[] }, Awaited<ReturnType<AudioTranscriber["analyze"]>>>("analyze", async (request, helpers) =>
  (await transcriberFor(request)).analyze(new Float32Array(request.audio), request.analyses, { ...request.options, signal: helpers.signal } as TranscribeOptions),
);

registerTask<WorkerAudioRequest & { result: TranscriptionResult; config?: DiarizationConfig }, TranscriptionResult>("diarize", async (request) =>
  (await transcriberFor(request)).diarize(new Float32Array(request.audio), request.result, request.config ?? {}),
);

registerTask<WorkerAudioRequest & { result: TranscriptionResult }, TranscriptionResult>("align", async (request) =>
  (await transcriberFor(request)).align(new Float32Array(request.audio), request.result, request.options?.alignment ?? {}),
);

registerTask<WorkerAudioRequest & { options?: { overTime?: boolean; interval?: number } }, LanguageDetectionResult>("detect_language", async (request, helpers) =>
  (await transcriberFor(request)).detectLanguage(new Float32Array(request.audio), { ...(request.options ?? {}), signal: helpers.signal }),
);

registerTask<WorkerAudioRequest, LanguagePortion[]>("language_over_time", async (request, helpers) =>
  (await transcriberFor(request)).languageOverTime(new Float32Array(request.audio), { signal: helpers.signal }),
);

registerTask<WorkerAudioRequest & { name: string }, string[]>("enroll_speaker", async (request) => {
  const transcriber = await transcriberFor(request);
  await transcriber.enrollSpeaker(request.name, new Float32Array(request.audio));
  return transcriber.speakers();
});

registerTask<WorkerLoadRequest & { name: string }, string[]>("forget_speaker", async (request) => {
  const transcriber = await transcriberFor(request);
  transcriber.forgetSpeaker(request.name);
  return transcriber.speakers();
});

/** The enrolled prints, for a server to keep; with `prints`, restores them first. */
registerTask<WorkerLoadRequest & { prints?: Record<string, number[]> }, Record<string, number[]>>("speaker_prints", async (request) => {
  const transcriber = await transcriberFor(request);
  if (request.prints) transcriber.loadSpeakerPrints(request.prints);
  return transcriber.speakerPrints();
});

registerTask<WorkerAudioRequest, TranscriptionResult>(
  "transcribe",
  async (request, helpers) => {
    report = (p) => helpers.progress(p);
    const transcriber = await transcriberFor(request, (p) => helpers.progress(p));
    const audio = new Float32Array(request.audio);
    return transcriber.transcribe(audio, {
      ...(request.options as TranscribeOptions),

      signal: helpers.signal,
      onProgress: (progress) => helpers.progress(progress),
    });
  },
);

/**
 * The audio-only analyses, with no transcriber loaded: a second worker runs them while the first
 * one transcribes (one onnxruntime runs one graph at a time; two workers are two runtimes).
 */
let analyzer: AudioAnalyzer | null = null;

registerTask<{ audio: ArrayBuffer; analyses: string[]; options?: Options; prints?: Record<string, number[]>; wasmPaths?: string }, AudioOnlyAnalysis>(
  "analyze_audio",
  async (request, helpers) => {
    report = (p) => helpers.progress(p);
    if (!analyzer) {
      const ort = await import("onnxruntime-web");
      if (request.wasmPaths) (ort.env.wasm as { wasmPaths?: string }).wasmPaths ??= request.wasmPaths;
      analyzer = new AudioAnalyzer({ ort: ort as never, Tensor: ort.Tensor as never, logger: workerLogger as never });
    }
    if (request.prints) analyzer.loadPrints(request.prints);
    return analyzer.analyze(new Float32Array(request.audio), checkAnalyses(request.analyses), { ...(request.options as TranscribeOptions), signal: helpers.signal });
  },
);

const sessions = new Map<string, { session: StreamSession; done: (result: StreamResult) => void }>();

registerTask<WorkerLoadRequest & { key: string; stream: Omit<TranscriberStreamOptions, "onUpdate" | "signal" | "loadOptions"> }, StreamResult>("stream", async (request, helpers) => {
  report = (p) => helpers.progress(p);
  const transcriber = await transcriberFor(request, (p) => helpers.progress(p));
  const session = await transcriber.stream({
    ...request.stream,

    loadOptions: {
      ...(request.endpoint ? { endpoint: request.endpoint } : {}),
      ...(request.backends ? { backends: request.backends as never } : {}),
      ...(request.wasmPaths ? { wasmPaths: request.wasmPaths } : {}),
      logger: workerLogger as never,
    },
  });
  session.on((update) => helpers.progress({ update }));
  helpers.progress({ opened: { streamId: session.streamId, backend: transcriber.backend } });
  return new Promise<StreamResult>((done) => sessions.set(request.key, { session, done }));
});

registerTask<{ key: string; audio: ArrayBuffer }, null>("stream_feed", async (request) => {
  sessions.get(request.key)?.session.feed(new Float32Array(request.audio));
  return null;
});

registerTask<{ key: string }, StreamResult | null>("stream_close", async (request) => {
  const open = sessions.get(request.key);
  if (!open) return null;
  sessions.delete(request.key);
  const result = await open.session.close();
  open.done(result);
  return result;
});

registerTask<null, null>("close", async () => {
  await current?.transcriber.close();
  current = null;
  await analyzer?.close();
  analyzer = null;
  return null;
});

serve();
