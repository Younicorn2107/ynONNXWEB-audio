/** Turning a token stream into timed segments. */

import type { Word } from "../alignment/punctuation.js";
import { TIME_PRECISION, type WhisperTokens } from "./tokens.js";
import type { Tokenizer } from "@youni/onnxweb-core/core";

export interface Segment {
  start: number;
  end: number;
  text: string;
  tokens: number[];
  /** Position in the transcript. */
  id?: number;
  /**
   * The quality of the window it came from: a caller filtering on it needs to know which segment.
   */
  avgLogprob?: number;
  compressionRatio?: number;
  noSpeechProb?: number;
  temperature?: number;
  /** Set by diarization. */
  speaker?: string | null;
  /** Present only when words were aligned. */
  words?: Word[] | null;
}

export interface SegmentsResult {
  segments: Segment[];
  text: string;
  /** End of the last CLOSED segment; null when none closed. */
  lastEnd: number | null;
}

/** Split `ids` into segments, offsetting every time by `offset` seconds. */
export function extractSegments(
  ids: readonly number[],
  tokens: WhisperTokens,
  tokenizer: Tokenizer,
  offset = 0,
): SegmentsResult {
  const segments: Segment[] = [];
  let current: number[] = [];
  let start: number | null = null;
  let lastEnd: number | null = null;

  const flush = (end: number | null): void => {
    if (current.length === 0) {
      return;
    }
    const text = tokenizer.decode(current).trim();
    if (text.length > 0 && start !== null) {
      segments.push({
        start: offset + start,
        end: offset + (end ?? start),
        text,
        tokens: [...current],
      });
    }
    current = [];
  };

  for (const id of ids) {
    const time = tokens.timeOf(id);
    if (time === null) {
      if (id < tokens.eot) current.push(id);
      continue;
    }
    if (start === null) {
      start = time;
      continue;
    }

    flush(time);
    lastEnd = offset + time;
    start = time;
  }

  if (current.length > 0) {
    if (start === null) start = 0;
    flush(null);
  }

  const text = segments
    .map((segment) => segment.text)
    .join(" ")
    .replace(/\s+/g, " ")
    .trim();

  return { segments, text, lastEnd };
}

/** Merge overlapping segments from consecutive windows. */
export function mergeSegments(groups: readonly Segment[][]): Segment[] {
  const all = groups.flat().sort((a, b) => a.start - b.start);
  const out: Segment[] = [];
  for (const segment of all) {
    const previous = out.at(-1);

    if (previous && segment.start < previous.end - TIME_PRECISION) {
      if (normalise(previous.text) === normalise(segment.text)) continue;
      if (normalise(segment.text).startsWith(normalise(previous.text))) {
        out[out.length - 1] = { ...segment, start: previous.start };
        continue;
      }
    }
    out.push(segment);
  }
  return out;
}

function normalise(text: string): string {
  return text.toLowerCase().replace(/[^\p{L}\p{N}]+/gu, " ").trim();
}
