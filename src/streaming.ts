/**
 * Streaming (spec § 4): an incremental run of the same library, fed audio as it arrives, publishing
 * provisional segments and revising them.
 */

import { ConfigurationError } from "@youni/onnxweb-core/core";

import { StreamingError } from "./errors.js";
import type { Segment } from "./whisper/segments.js";
import type { TranscribeOptions, TranscriptionResult } from "./whisper/transcriber.js";

export type StreamMode = "simple" | "double" | "post";
export type Producer = "single" | "fast" | "large" | "post_final";

export interface StreamSegment {
  start: number;
  end: number;
  text: string;
  avgLogprob: number;
}

export interface StreamUpdate {
  streamId: string;
  mode: StreamMode;
  segmentId: string;
  revision: number;
  final: boolean;
  producer: Producer;
  /**
   * The segment as it now stands; null when it was removed (replaced, merged, or decoded to
   * nothing).
   */
  segment: StreamSegment | null;
}

/** What a streaming session needs from a transcriber. */
export interface StreamingTranscriber {
  readonly isMultilingual: boolean;
  readonly modelName: string;
  transcribe(audio: Float32Array, options?: TranscribeOptions): Promise<TranscriptionResult>;
  detectLanguage(audio: Float32Array, options?: { signal?: AbortSignal }): Promise<{ language: string; probability?: number }>;
  /**
   * Speech spans, in seconds; without it every window is decoded and the language read on the first
   * seconds.
   */
  detectSpeech?(audio: Float32Array): Promise<{ start: number; end: number }[]>;
}

export interface StreamOptions {
  /** The session's identity; generated when omitted. */
  audioId?: string;
  mode?: StreamMode;
  /** The language, when known: otherwise detected once, on the first seconds. */
  language?: string | null;
  /** Re-decode weak segments at a warmer temperature (`simple`, and `large` in `double`). */
  enableRetry?: boolean;
  /** Seconds of new audio between two decoding steps (default 1). */
  stepS?: number;
  /**
   * With no `language`: check it again every this many seconds of new speech (20), on the last 15 s
   * of speech, and switch when another one is sure (probability ≥ 0.8). 0 detects once.
   */
  languageRecheckS?: number;
  /** Called with every update, in order. */
  onUpdate?: (update: StreamUpdate) => void;
  signal?: AbortSignal;
}

/** What `close()` returns: the session's final transcript. */
export interface StreamResult {
  streamId: string;
  mode: StreamMode;
  text: string;
  segments: StreamSegment[];
  language: string | null;
  duration: number;
  plan: StreamPlan;
  /** `post`: the full transcription of the recording, which the segments come from. */
  final: TranscriptionResult | null;
}

/** The plan of a session (spec § 7): models and their roles, chunks, revision and finalization. */
export interface StreamPlan {
  mode: StreamMode;
  chunkTargetS: number;
  maxChunkDurationS: number;
  modelRoles: Partial<Record<"single" | "fast" | "large" | "final", { model: string; maxChunkDurationS: number; enableRetry: boolean }>>;
  enableRetry: boolean;
  promptInjection: boolean;
  revisionPolicy: string;
  finalizationPolicy: string;
  audioPersistence: boolean;
}

export const SAMPLE_RATE = 16000;
/** The longest window a step decodes (Whisper's own). */
export const MAX_WINDOW_S = 30;
/** The fast model's window in `double`. */
export const FAST_WINDOW_S = 3;
/** A segment ending this long before the newest audio has its right context: it is settled. */
export const SETTLE_MARGIN_S = 1.5;
/** In `double`, settled audio the large model waits for before decoding it. */
export const LARGE_STEP_S = 8;
/** Audio needed before the language is detected (speech, when it can be told). */
const LANGUAGE_PROBE_S = 2;
/** A pause this long after speech ends an utterance: what came before it settles. */
export const PAUSE_S = 0.8;
/** Speech read to check the language again. */
const LANGUAGE_RECHECK_WINDOW_S = 15;
/** A language replaces the current one only this sure. */
const LANGUAGE_SWITCH_PROBABILITY = 0.8;

interface Tracked {
  id: string;
  revision: number;
  producer: Producer;
  segment: StreamSegment;
}

const same = (a: StreamSegment, b: StreamSegment) => a.text === b.text && Math.abs(a.start - b.start) < 1e-3 && Math.abs(a.end - b.end) < 1e-3;

/** A random session id (opaque, never derived from a file name). */
function sessionId(): string {
  const bytes = new Uint8Array(8);
  globalThis.crypto.getRandomValues(bytes);
  return `stream-${Array.from(bytes, (b) => b.toString(16).padStart(2, "0")).join("")}`;
}

/** One decoder walking the stream: settled segments, open ones, and the seek between them. */
class Lane {
  seek = 0;
  settled: Tracked[] = [];
  open: Tracked[] = [];
  /** The previous decoding of the audio from `seek`: a segment both agree on is settled. */
  previous: StreamSegment[] = [];

  constructor(
    readonly producer: Producer,
    readonly windowS: number,
  ) {}

  get context(): string {
    return this.settled.slice(-8).map((t) => t.segment.text).join(" ");
  }
}

export class StreamSession {
  readonly streamId: string;
  readonly mode: StreamMode;
  readonly #listeners: ((update: StreamUpdate) => void)[] = [];
  readonly #queue: StreamUpdate[] = [];
  readonly #waiters: ((value: IteratorResult<StreamUpdate>) => void)[] = [];
  #chunks: Float32Array[] = [];
  #samples = 0;
  #audio: Float32Array | null = null;
  #decodedAt = 0;
  #language: string | null;
  #closed = false;
  #ended = false;
  #work: Promise<void> = Promise.resolve();
  #ids = 0;
  #final: TranscriptionResult | null = null;
  /** Speech spans of the whole stream, as far as `#speechUntil`. */
  #speech: { start: number; end: number }[] = [];
  #speechUntil = 0;
  /** The language was detected (not given): it may be checked again. */
  readonly #detectedLanguage: boolean;
  #languageCheckedAt = 0;
  readonly #single: Lane;
  readonly #large: Lane | null;

  constructor(
    readonly transcriber: StreamingTranscriber,
    readonly options: StreamOptions = {},
    /** `double`: the fast model; `post`: the model of the final pass (the same one when omitted). */
    readonly second: StreamingTranscriber | null = null,
    /** Run once the session is closed (to release a model loaded for it). */
    readonly onClosed: (() => Promise<void>) | null = null,
  ) {
    this.mode = options.mode ?? "simple";
    if (!["simple", "double", "post"].includes(this.mode)) throw new ConfigurationError(`'${String(this.mode)}' is not a streaming mode (simple, double, post)`);
    if (this.mode === "double" && !second) throw new StreamingError("'double' needs a fast model beside the large one");
    this.streamId = options.audioId ?? sessionId();
    this.#language = options.language ?? (transcriber.isMultilingual ? null : "en");
    this.#detectedLanguage = !this.#language;
    this.#single = new Lane(this.mode === "double" ? "fast" : "single", this.mode === "double" ? FAST_WINDOW_S : MAX_WINDOW_S);
    this.#large = this.mode === "double" ? new Lane("large", MAX_WINDOW_S) : null;
    if (options.onUpdate) this.#listeners.push(options.onUpdate);
  }

  /** The plan this session runs. */
  get plan(): StreamPlan {
    const retry = this.options.enableRetry ?? false;
    const roles: StreamPlan["modelRoles"] =
      this.mode === "double"
        ? {
            fast: { model: this.second!.modelName, maxChunkDurationS: FAST_WINDOW_S, enableRetry: false },
            large: { model: this.transcriber.modelName, maxChunkDurationS: MAX_WINDOW_S, enableRetry: retry },
          }
        : { single: { model: this.transcriber.modelName, maxChunkDurationS: MAX_WINDOW_S, enableRetry: retry } };
    if (this.mode === "post") roles.final = { model: (this.second ?? this.transcriber).modelName, maxChunkDurationS: MAX_WINDOW_S, enableRetry: true };
    return {
      mode: this.mode,
      chunkTargetS: this.options.stepS ?? 1,
      maxChunkDurationS: this.mode === "double" ? FAST_WINDOW_S : MAX_WINDOW_S,
      modelRoles: roles,
      enableRetry: retry,
      promptInjection: true,
      revisionPolicy:
        this.mode === "double"
          ? `fast segments are revised while unsettled (${SETTLE_MARGIN_S} s); large segments replace them over the time they cover`
          : `a segment is decoded again at every step until it ends ${SETTLE_MARGIN_S} s before the newest audio`,
      finalizationPolicy: this.mode === "post" ? "close() runs a full transcription that replaces every provisional segment" : "close() finalizes every segment; nothing is final before",
      audioPersistence: true,
    };
  }

  /** Seconds of audio received. */
  get duration(): number {
    return this.#samples / SAMPLE_RATE;
  }

  /** Every update, as an async iterator (ends after `close()`). */
  async *updates(): AsyncGenerator<StreamUpdate> {
    for (;;) {
      const next = this.#queue.shift();
      if (next) {
        yield next;
        continue;
      }
      if (this.#ended) return;
      const value = await new Promise<IteratorResult<StreamUpdate>>((resolve) => this.#waiters.push(resolve));
      if (value.done) return;
      yield value.value;
    }
  }

  on(listener: (update: StreamUpdate) => void): () => void {
    this.#listeners.push(listener);
    return () => this.#listeners.splice(this.#listeners.indexOf(listener), 1);
  }

  /** Append mono 16 kHz audio; a decoding step runs when enough has arrived. */
  feed(chunk: Float32Array): void {
    if (this.#closed) throw new StreamingError("the session is closed");
    if (chunk.length === 0) return;
    this.#chunks.push(chunk.slice());
    this.#samples += chunk.length;
    this.#audio = null;
    if (this.duration - this.#decodedAt >= (this.options.stepS ?? 1)) this.#schedule(false);
  }

  /** Decode what has arrived now. Finalizes nothing. */
  async flush(): Promise<void> {
    this.#schedule(false);
    await this.#work;
  }

  /** End the stream: last steps, then every segment final (after the full pass in `post`). */
  async close(): Promise<StreamResult> {
    if (!this.#closed) {
      this.#closed = true;
      this.#schedule(true);
      try {
        await this.#work;
      } finally {
        this.#ended = true;
        for (const resolve of this.#waiters.splice(0)) resolve({ value: undefined, done: true });
        await this.onClosed?.();
      }
    } else {
      await this.#work;
    }
    const segments = this.segments;
    return {
      streamId: this.streamId,
      mode: this.mode,
      text: segments.map((s) => s.text).join(" ").replace(/\s+/g, " ").trim(),
      segments,
      language: this.#language,
      duration: this.duration,
      plan: this.plan,
      final: this.#final,
    };
  }

  /** The segments as they stand: settled then open, in time order. */
  get segments(): StreamSegment[] {
    const lanes = this.#large ? [this.#large, this.#single] : [this.#single];
    return lanes
      .flatMap((lane) => [...lane.settled, ...lane.open])
      .map((t) => t.segment)
      .sort((a, b) => a.start - b.start);
  }

  #schedule(closing: boolean): void {
    this.#work = this.#work.then(() => this.#step(closing));
  }

  #all(): Float32Array {
    if (this.#audio && this.#audio.length === this.#samples) return this.#audio;
    const out = new Float32Array(this.#samples);
    let at = 0;
    for (const chunk of this.#chunks) {
      out.set(chunk, at);
      at += chunk.length;
    }
    this.#chunks = [out];
    this.#audio = out;
    return out;
  }

  #emit(tracked: Tracked, final: boolean, removed = false): void {
    const update: StreamUpdate = {
      streamId: this.streamId,
      mode: this.mode,
      segmentId: tracked.id,
      revision: tracked.revision,
      final,
      producer: tracked.producer,
      segment: removed ? null : { ...tracked.segment },
    };
    for (const listener of this.#listeners) listener(update);
    const waiter = this.#waiters.shift();
    if (waiter) waiter({ value: update, done: false });
    else this.#queue.push(update);
  }

  /** The stream's speech spans, the new audio read (with a second of context before it). */
  async #updateSpeech(audio: Float32Array): Promise<void> {
    if (!this.transcriber.detectSpeech) return;
    const end = audio.length / SAMPLE_RATE;
    if (end - this.#speechUntil < 0.25) return;
    const from = Math.max(0, this.#speechUntil - 1);
    const found = await this.transcriber.detectSpeech(audio.subarray(Math.round(from * SAMPLE_RATE)));

    this.#speech = this.#speech.filter((s) => s.end <= from);
    for (const s of found) {
      const span = { start: from + s.start, end: from + s.end };
      const last = this.#speech[this.#speech.length - 1];
      if (last && span.start - last.end < 0.3) last.end = Math.max(last.end, span.end);
      else this.#speech.push(span);
    }
    this.#speechUntil = end;
  }

  /** Seconds of speech in `[from, to]`, all of it when the transcriber cannot tell. */
  #speechIn(from: number, to: number): number {
    if (!this.transcriber.detectSpeech) return to - from;
    return this.#speech.reduce((sum, s) => sum + Math.max(0, Math.min(s.end, to) - Math.max(s.start, from)), 0);
  }

  /** The speech of the stream, joined, its last `seconds` at most. */
  #speechAudio(audio: Float32Array, seconds: number): Float32Array {
    if (!this.transcriber.detectSpeech || this.#speech.length === 0) return audio.subarray(Math.max(0, audio.length - seconds * SAMPLE_RATE));
    const parts: Float32Array[] = [];
    let total = 0;
    for (const s of [...this.#speech].reverse()) {
      const part = audio.subarray(Math.round(s.start * SAMPLE_RATE), Math.round(s.end * SAMPLE_RATE));
      parts.unshift(part);
      total += part.length;
      if (total >= seconds * SAMPLE_RATE) break;
    }
    const out = new Float32Array(total);
    let at = 0;
    for (const p of parts) {
      out.set(p, at);
      at += p.length;
    }
    return out.subarray(Math.max(0, out.length - seconds * SAMPLE_RATE));
  }

  async #ensureLanguage(audio: Float32Array): Promise<boolean> {
    const end = audio.length / SAMPLE_RATE;
    const speech = this.#speechIn(0, end);
    const options = this.options.signal ? { signal: this.options.signal } : {};
    if (!this.#language) {
      if (speech < LANGUAGE_PROBE_S && !this.#closed) return false;
      this.#language = (await this.transcriber.detectLanguage(this.#speechAudio(audio, MAX_WINDOW_S), options)).language;
      this.#languageCheckedAt = speech;
      return true;
    }
    const every = this.options.languageRecheckS ?? 20;
    if (this.#detectedLanguage && every > 0 && speech - this.#languageCheckedAt >= every) {
      this.#languageCheckedAt = speech;
      const found = await this.transcriber.detectLanguage(this.#speechAudio(audio, LANGUAGE_RECHECK_WINDOW_S), options);
      if (found.language !== this.#language && (found.probability ?? 0) >= LANGUAGE_SWITCH_PROBABILITY) this.#language = found.language;
    }
    return true;
  }

  async #step(closing: boolean): Promise<void> {
    if (this.options.signal?.aborted) throw new StreamingError("the stream was cancelled");
    const audio = this.#all();
    const end = audio.length / SAMPLE_RATE;
    await this.#updateSpeech(audio);
    if (end > 0 && (await this.#ensureLanguage(audio))) {
      const fast = this.mode === "double" ? this.second! : this.transcriber;

      if (this.#large) await this.#largeStep(audio, end, closing);
      await this.#walk(this.#single, fast, audio, end, closing, this.mode !== "double" && (this.options.enableRetry ?? false));
      this.#decodedAt = end;
    }
    if (!closing) return;
    if (this.mode === "post") await this.#finalPass(audio);
    else this.#finalizeAll();
  }

  /**
   * Decode the lane's unsettled audio, revise its open segments, settle those well behind the
   * newest audio.
   */
  async #walk(lane: Lane, transcriber: StreamingTranscriber, audio: Float32Array, end: number, closing: boolean, retry: boolean): Promise<void> {
    if (this.#large && lane.seek < this.#large.seek) {
      lane.seek = this.#large.seek;
      for (const t of lane.open.splice(0)) this.#retire(t);
    }
    while (lane.seek < end - 0.1) {
      if (!closing && this.#speechIn(lane.seek, end) < 0.2) {
        lane.seek = Math.max(lane.seek, end - 0.5);
        for (const t of lane.open.splice(0)) this.#retire(t);
        lane.previous = [];
        break;
      }
      const stop = Math.min(end, lane.seek + lane.windowS);
      const window = audio.subarray(Math.round(lane.seek * SAMPLE_RATE), Math.round(stop * SAMPLE_RATE));
      const result = await transcriber.transcribe(window, {
        language: this.#language,
        temperature: retry ? "auto" : 0,
        vad: "off",
        initialPrompt: lane.context,
        ...(this.options.signal ? { signal: this.options.signal } : {}),
      });
      const segments = cleanSegments(result.segments, lane.seek, stop);

      const full = stop - lane.seek >= lane.windowS - 0.05 && stop < end;
      let settle = 0;
      if (closing) settle = segments.length;
      else {
        while (
          settle < segments.length &&
          segments[settle]!.end <= end - SETTLE_MARGIN_S &&
          lane.previous[settle] !== undefined &&
          lane.previous[settle]!.text === segments[settle]!.text &&
          Math.abs(lane.previous[settle]!.start - segments[settle]!.start) < 0.5
        ) {
          settle += 1;
        }
        if (full && settle === 0) settle = Math.max(segments.length - 1, segments.length > 0 ? 1 : 0);

        const lastSpeech = this.#speech.length > 0 ? this.#speech[this.#speech.length - 1]! : null;
        if (this.transcriber.detectSpeech && lastSpeech && end - lastSpeech.end >= PAUSE_S) {
          while (settle < segments.length && segments[settle]!.end <= lastSpeech.end + 0.3) settle += 1;
        }
      }
      this.#revise(lane, segments);
      const settledNow = lane.open.splice(0, settle);
      lane.settled.push(...settledNow);
      lane.previous = segments.slice(settle);
      if (settledNow.length > 0) lane.seek = settledNow[settledNow.length - 1]!.segment.end;
      else if (full) {
        lane.seek = stop;
        lane.previous = [];
      }

      if (!(full || (closing && stop < end))) break;
      if (settledNow.length === 0 && !full) break;
    }
  }

  /**
   * The lane's open segments become `segments`: same ids by position, a new revision where
   * something changed.
   */
  #revise(lane: Lane, segments: StreamSegment[]): void {
    const next: Tracked[] = [];
    segments.forEach((segment, i) => {
      const previous = lane.open[i];
      if (previous) {
        if (!same(previous.segment, segment)) {
          previous.segment = segment;
          previous.revision += 1;
          this.#emit(previous, false);
        }
        next.push(previous);
      } else {
        const tracked: Tracked = { id: `${this.streamId}-${this.#ids++}`, revision: 0, producer: lane.producer, segment };
        this.#emit(tracked, false);
        next.push(tracked);
      }
    });
    for (const gone of lane.open.slice(segments.length)) this.#retire(gone);
    lane.open = next;
  }

  /** A segment that no longer exists: re-emitted empty, final. */
  #retire(tracked: Tracked): void {
    tracked.revision += 1;
    this.#emit(tracked, true, true);
  }

  /**
   * `double`: the large model redoes the fast lane's settled audio and replaces its segments there.
   */
  async #largeStep(audio: Float32Array, end: number, closing: boolean): Promise<void> {
    const large = this.#large!;
    const ready = closing ? end : Math.max(...this.#single.settled.map((t) => t.segment.end), large.seek);
    if (!closing && ready - large.seek < LARGE_STEP_S) return;
    while (large.seek < ready - 0.1) {
      const stop = Math.min(ready, large.seek + MAX_WINDOW_S);
      const window = audio.subarray(Math.round(large.seek * SAMPLE_RATE), Math.round(stop * SAMPLE_RATE));
      const result = await this.transcriber.transcribe(window, {
        language: this.#language,
        temperature: this.options.enableRetry ? "auto" : 0,
        vad: "off",
        initialPrompt: large.context,
        ...(this.options.signal ? { signal: this.options.signal } : {}),
      });
      const segments = cleanSegments(result.segments, large.seek, stop);
      for (const segment of segments) {
        const tracked: Tracked = { id: `${this.streamId}-${this.#ids++}`, revision: 0, producer: "large", segment };
        large.settled.push(tracked);
        this.#emit(tracked, false);
      }
      const covered = segments.length > 0 ? segments[segments.length - 1]!.end : stop;

      const fast = this.#single;
      for (const t of [...fast.settled, ...fast.open]) if ((t.segment.start + t.segment.end) / 2 < covered) this.#retire(t);
      fast.settled = fast.settled.filter((t) => (t.segment.start + t.segment.end) / 2 >= covered);
      fast.open = fast.open.filter((t) => (t.segment.start + t.segment.end) / 2 >= covered);
      large.seek = Math.max(covered, large.seek + 0.1);
      if (stop >= ready) break;
    }
  }

  /** Every standing segment, re-emitted final. */
  #finalizeAll(): void {
    const lanes = this.#large ? [this.#large, this.#single] : [this.#single];
    for (const lane of lanes) {
      lane.settled.push(...lane.open.splice(0));
      for (const t of lane.settled) {
        t.revision += 1;
        this.#emit(t, true);
      }
    }
  }

  /** `post`: the whole recording transcribed again; it replaces every provisional segment. */
  async #finalPass(audio: Float32Array): Promise<void> {
    const lane = this.#single;
    for (const t of [...lane.settled, ...lane.open]) this.#retire(t);
    lane.settled = [];
    lane.open = [];
    const transcriber = this.second ?? this.transcriber;
    const result = await transcriber.transcribe(audio, {
      ...(this.#language ? { language: this.#language } : {}),
      ...(this.options.signal ? { signal: this.options.signal } : {}),
      audioIds: [this.streamId],
    });
    for (const s of result.segments.filter((x) => x.text.trim())) {
      const tracked: Tracked = { id: `${this.streamId}-${this.#ids++}`, revision: 0, producer: "post_final", segment: { start: s.start, end: s.end, text: s.text.trim(), avgLogprob: s.avgLogprob ?? 0 } };
      lane.settled.push(tracked);
      this.#emit(tracked, true);
    }
    this.#final = result;
  }
}

/** Whisper's own bar for a degenerate output: text that compresses this well is repeating itself. */
export const MAX_COMPRESSION_RATIO = 2.4;

/**
 * A window's segments on the stream's timeline, kept only while believable: a window shorter than
 * 30 s is padded with silence, and Whisper may write timestamps past the audio's end, or loop on
 * its incomplete tail. A segment starting past the audio, or whose text repeats itself, ends the
 * list; a segment identical to the one before it is dropped.
 */
export function cleanSegments(segments: readonly Segment[], offset: number, stop: number): StreamSegment[] {
  const out: StreamSegment[] = [];
  for (const s of segments) {
    const text = s.text.trim();
    if (!text) continue;
    const start = round3(Math.min(stop, offset + s.start));
    if (start >= stop - 0.05) break;
    if ((s.compressionRatio ?? 0) > MAX_COMPRESSION_RATIO || selfRepeating(text)) break;
    let end = round3(Math.min(stop, offset + s.end));
    if (end <= start) end = round3(stop);
    if (out.length > 0 && out[out.length - 1]!.text === text) continue;
    out.push({ start, end, text, avgLogprob: s.avgLogprob ?? 0 });
  }
  return out;
}

/** A phrase of three words or more said three times in a row. */
function selfRepeating(text: string): boolean {
  return /(\b.{12,}?)(?:\s*\1){2,}/u.test(text);
}

function round3(x: number): number {
  return Math.round(x * 1000) / 1000;
}
