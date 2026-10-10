/** Whisper's control tokens, resolved from the tokenizer rather than hardcoded. */

import { YnONNXAudioError } from "../errors.js";
import type { Tokenizer } from "@youni/onnxweb-core/core";

/**
 * The first vocabulary size of a multilingual checkpoint. An English-only one has 51864 — and ships
 * the FULL multilingual tokenizer, so counting its language tokens says "multilingual" for a model
 * that is not. The size is the marker, as in the reference implementation.
 */
export const MULTILINGUAL_VOCAB = 51865;

export const SOT = "<|startoftranscript|>";
export const SOT_PREV = "<|startofprev|>";
export const EOT = "<|endoftext|>";
export const NO_SPEECH = "<|nospeech|>";
export const NO_SPEECH_LEGACY = "<|nocaptions|>";
export const NO_TIMESTAMPS = "<|notimestamps|>";
export const TRANSCRIBE = "<|transcribe|>";
export const TRANSLATE = "<|translate|>";
export const TIMESTAMP_ZERO = "<|0.00|>";

/** Whisper's timestamp grid: 20 ms per step, 1 501 stamps over 30 s. */
export const TIME_PRECISION = 0.02;
export const TIMESTAMP_COUNT = 1501;

const TIMESTAMP_RE = /^<\|(\d+\.\d\d)\|>$/;

export class WhisperTokens {
  readonly sot: number;
  readonly eot: number;
  readonly sotPrev: number | null;
  readonly noSpeech: number | null;
  readonly noTimestamps: number | null;
  readonly transcribe: number | null;
  readonly translate: number | null;
  readonly timestampBegin: number | null;
  readonly languages: ReadonlyMap<string, number>;

  readonly isMultilingual: boolean;

  /**
   * @param vocabSize The logits width, when the graph declares it; the tokenizer's own size
   * otherwise. `multilingual` overrides both (tests).
   */
  constructor(
    private readonly tokenizer: Tokenizer,
    options: { vocabSize?: number | null; multilingual?: boolean } = {},
  ) {
    const sot = tokenizer.tokenToId(SOT);
    const eot = tokenizer.tokenToId(EOT);
    if (sot === null || eot === null) {
      throw new YnONNXAudioError(
        `this tokenizer has no ${SOT} or ${EOT}, so it is not a Whisper tokenizer. ` +
          `Check that tokenizer.json came from the same entry as the graphs.`,
      );
    }
    this.sot = sot;
    this.eot = eot;
    this.sotPrev = tokenizer.tokenToId(SOT_PREV);
    this.noSpeech = tokenizer.tokenToId(NO_SPEECH) ?? tokenizer.tokenToId(NO_SPEECH_LEGACY);
    this.noTimestamps = tokenizer.tokenToId(NO_TIMESTAMPS);
    this.transcribe = tokenizer.tokenToId(TRANSCRIBE);
    this.translate = tokenizer.tokenToId(TRANSLATE);
    this.timestampBegin = tokenizer.tokenToId(TIMESTAMP_ZERO);

    const languages = new Map<string, number>();
    for (const [token, id] of Object.entries(tokenizer.specialIds())) {
      const match = /^<\|([a-z]{2,3}(?:_[a-z]+)?)\|>$/.exec(token);
      if (match && token !== EOT) languages.set(match[1]!, id);
    }
    const size = options.vocabSize ?? tokenizer.vocabSize;
    this.isMultilingual = options.multilingual ?? size >= MULTILINGUAL_VOCAB;

    this.languages = this.isMultilingual ? languages : new Map();
  }

  language(code: string): number | null {
    return this.languages.get(code.toLowerCase()) ?? null;
  }

  /** Is this id one of the 1 501 timestamp tokens? */
  isTimestamp(id: number): boolean {
    return this.timestampBegin !== null && id >= this.timestampBegin;
  }

  /** Seconds for a timestamp id, or null if it is not one. */
  timeOf(id: number): number | null {
    if (!this.isTimestamp(id)) return null;
    return (id - this.timestampBegin!) * TIME_PRECISION;
  }

  /** The id for a time in seconds, clamped to the grid. */
  timestampFor(seconds: number): number | null {
    if (this.timestampBegin === null) return null;
    const step = Math.round(seconds / TIME_PRECISION);
    return this.timestampBegin + Math.max(0, Math.min(step, TIMESTAMP_COUNT - 1));
  }

  /**
   * Ids that are prompt vocabulary, never transcript — Python's `WhisperPrompt.structural_ids`: the
   * markers, the task and every language.
   */
  suppressed(options?: { allowTimestamps?: boolean }): number[] {
    const ids: number[] = [];
    for (const id of [this.sot, this.sotPrev, this.noSpeech, this.transcribe, this.translate]) {
      if (id !== null) ids.push(id);
    }
    ids.push(...this.languages.values());

    if (options?.allowTimestamps !== false && this.noTimestamps !== null) {
      ids.push(this.noTimestamps);
    }
    return [...new Set(ids)].sort((a, b) => a - b);
  }

  parseTimestampToken(token: string): number | null {
    const match = TIMESTAMP_RE.exec(token);
    return match ? Number(match[1]) : null;
  }
}
