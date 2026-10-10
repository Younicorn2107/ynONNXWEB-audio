/** Word timestamps by dynamic time warping over cross-attention. */

import { AlignmentError } from "../errors.js";
import type { Word } from "./punctuation.js";

/** Audio seconds per encoder frame: 30 s over 1 500 positions. */
export const SECONDS_PER_FRAME = 0.02;

/** Median filter width over the attention, in frames (Whisper's own). */
export const MEDIAN_FILTER_WIDTH = 7;

/**
 * A monotonic path through a [tokens × frames] cost matrix (row-major), with the three classic
 * moves.
 */
export function dtwPath(cost: ArrayLike<number>, tokens: number, frames: number): { tokenPath: number[]; framePath: number[] } {
  if (tokens === 0 || frames === 0) return { tokenPath: [], framePath: [] };
  const width = frames + 1;
  const accumulated = new Float64Array((tokens + 1) * width).fill(Number.POSITIVE_INFINITY);
  accumulated[0] = 0;

  const move = new Uint8Array((tokens + 1) * width);
  for (let i = 1; i <= tokens; i += 1) {
    for (let j = 1; j <= frames; j += 1) {
      const options = [accumulated[(i - 1) * width + j - 1]!, accumulated[(i - 1) * width + j]!, accumulated[i * width + j - 1]!];

      let best = 0;
      if (options[1]! < options[best]!) best = 1;
      if (options[2]! < options[best]!) best = 2;
      accumulated[i * width + j] = cost[(i - 1) * frames + j - 1]! + options[best]!;
      move[i * width + j] = best;
    }
  }
  const tokenPath: number[] = [];
  const framePath: number[] = [];
  let i = tokens;
  let j = frames;
  while (i > 0 && j > 0) {
    tokenPath.push(i - 1);
    framePath.push(j - 1);
    const step = move[i * width + j];
    if (step === 0) {
      i -= 1;
      j -= 1;
    } else if (step === 1) {
      i -= 1;
    } else {
      j -= 1;
    }
  }
  return { tokenPath: tokenPath.reverse(), framePath: framePath.reverse() };
}

/** Median along the frame axis, edges repeated. Attention is spiky; the path should not be. */
export function medianFilter(matrix: ArrayLike<number>, rows: number, columns: number, width = MEDIAN_FILTER_WIDTH): Float64Array {
  const out = Float64Array.from(matrix);
  if (width <= 1 || columns <= width) return out;
  const pad = Math.floor(width / 2);
  const window = new Float64Array(width);
  for (let r = 0; r < rows; r += 1) {
    for (let c = 0; c < columns; c += 1) {
      for (let k = 0; k < width; k += 1) {
        const index = Math.min(columns - 1, Math.max(0, c + k - pad));
        window[k] = matrix[r * columns + index]!;
      }
      window.sort();
      out[r * columns + c] = width % 2 === 1 ? window[pad]! : (window[width / 2 - 1]! + window[width / 2]!) / 2;
    }
  }
  return out;
}

/**
 * Token spans from cross-attention: [tokens × frames], heads already averaged. Per token, the
 * weights are filtered, centred and scaled, and the path follows where the model was LOOKING (cost
 * = −attention).
 */
export function tokenSpans(weights: Float32Array, tokens: number, frames: number, windowStart = 0, medianWidth = MEDIAN_FILTER_WIDTH): [number, number][] {
  if (weights.length < tokens * frames) throw new AlignmentError("the cross-attention is smaller than tokens × frames");
  const filtered = medianFilter(weights.subarray(0, tokens * frames), tokens, frames, medianWidth);
  const cost = new Float64Array(tokens * frames);
  for (let t = 0; t < tokens; t += 1) {
    let mean = 0;
    for (let f = 0; f < frames; f += 1) mean += filtered[t * frames + f]!;
    mean /= frames;
    let variance = 0;
    for (let f = 0; f < frames; f += 1) variance += (filtered[t * frames + f]! - mean) ** 2;
    const deviation = Math.max(Math.sqrt(variance / frames), 1e-6);
    for (let f = 0; f < frames; f += 1) cost[t * frames + f] = -((filtered[t * frames + f]! - mean) / deviation);
  }
  const { tokenPath, framePath } = dtwPath(cost, tokens, frames);
  const starts = new Map<number, number>();
  const ends = new Map<number, number>();
  tokenPath.forEach((token, k) => {
    if (!starts.has(token)) starts.set(token, framePath[k]!);
    ends.set(token, framePath[k]!);
  });
  return Array.from({ length: tokens }, (_, t) => [
    windowStart + (starts.get(t) ?? 0) * SECONDS_PER_FRAME,
    windowStart + ((ends.get(t) ?? 0) + 1) * SECONDS_PER_FRAME,
  ]);
}

/**
 * Sub-word pieces merged into words (a word starts at a piece beginning with a space), outer bounds
 * kept, and no word starting before the previous ends.
 */
export function groupIntoWords(pieces: readonly string[], spans: readonly [number, number][]): Word[] {
  const words: Word[] = [];
  let current = "";
  let start = spans[0]?.[0] ?? 0;
  let end = start;
  pieces.forEach((piece, k) => {
    const [pieceStart, pieceEnd] = spans[k]!;
    const startsWord = piece.startsWith(" ") || piece.startsWith("\n") || !current;
    if (startsWord && current.trim()) {
      words.push({ word: current.trim(), start, end, probability: 0 });
      current = "";
      start = pieceStart;
    }
    if (!current) start = pieceStart;
    current += piece;
    end = Math.max(end, pieceEnd);
  });
  if (current.trim()) words.push({ word: current.trim(), start, end, probability: 0 });
  const out: Word[] = [];
  for (const word of words) {
    const previous = out[out.length - 1];
    const begin = previous && word.start < previous.end ? previous.end : word.start;
    out.push({ word: word.word, start: Math.round(begin * 1000) / 1000, end: Math.round(Math.max(word.end, begin) * 1000) / 1000, probability: 0 });
  }
  return out;
}
