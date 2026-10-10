/** When VAD is used, and what it does to the audio. */

import type { SpeechSegment } from "./vad.js";

/** From this parameter count up, VAD is on by default; below, off. */
export const LARGE_MODEL_PARAMS_M = 500;

/** Parameter counts of the catalogue, in millions (Python's model registry). */
export const WHISPER_PARAMS_M: Readonly<Record<string, number>> = {
  tiny: 39, "tiny.en": 39, base: 74, "base.en": 74, small: 244, "small.en": 244,
  medium: 769, "medium.en": 769, "large-v1": 1550, "large-v2": 1550, "large-v3": 1550,
  "large-v3-turbo": 809, "distil-small.en": 166, "distil-medium.en": 394,
  "distil-large-v2": 756, "distil-large-v3": 756, "distil-large-v3.5": 756,
};

export type VADMode = "auto" | "on" | "off";

/**
 * Resolve the mode for a model; an unknown checkpoint gets the answer that cannot lose a passage.
 */
export function shouldUseVad(paramsM: number | null, mode: VADMode = "auto"): boolean {
  if (mode === "on") return true;
  if (mode === "off") return false;
  return paramsM !== null && paramsM >= LARGE_MODEL_PARAMS_M;
}

/**
 * Merge speech intervals into window-sized spans: two join when the silence between them is at most
 * `maxGapS` and the union still fits `maxDurationS` (28 s leaves headroom under Whisper's 30 s
 * window).
 */
export function groupSpeech(segments: readonly SpeechSegment[], maxDurationS = 28, maxGapS = 2): [number, number][] {
  if (segments.length === 0) return [];
  const ordered = [...segments].sort((a, b) => a.start - b.start);
  const groups: [number, number][] = [];
  let start = ordered[0]!.start;
  let end = ordered[0]!.end;
  for (const segment of ordered.slice(1)) {
    if (segment.start - end <= maxGapS && segment.end - start <= maxDurationS) {
      end = Math.max(end, segment.end);
    } else {
      groups.push([start, end]);
      start = segment.start;
      end = segment.end;
    }
  }
  groups.push([start, end]);
  return groups;
}

/** The fraction of the audio the VAD called speech. */
export function speechCoverage(segments: readonly SpeechSegment[], duration: number): number {
  if (duration <= 0 || segments.length === 0) return 0;
  const total = segments.reduce((sum, s) => sum + Math.max(0, s.end - s.start), 0);
  return Math.min(1, total / duration);
}

/** Widen each span, clamped to the audio, without letting them overlap. */
export function applyPadding(spans: readonly [number, number][], padMs: number, duration: number): [number, number][] {
  if (spans.length === 0 || padMs <= 0) return [...spans];
  const pad = padMs / 1000;
  const out: [number, number][] = [];
  for (const [start, end] of spans) {
    let low = Math.max(0, start - pad);
    const high = Math.min(duration, end + pad);
    const previous = out[out.length - 1];
    if (previous && low < previous[1]) low = previous[1];
    out.push([low, Math.max(high, low)]);
  }
  return out;
}

/** One kept stretch: where it sits in the clipped audio, and where it came from. */
export type SpeechMap = [clipStart: number, clipEnd: number, originalStart: number][];

/**
 * Keep only the speech, and the map back to the original timeline. Overlapping spans are not
 * duplicated.
 */
export function clipToSpeech(
  signal: Float32Array,
  spans: readonly [number, number][],
  sampleRate = 16000,
): { clipped: Float32Array; map: SpeechMap } {
  const parts: Float32Array[] = [];
  const map: SpeechMap = [];
  let position = 0;
  let lastEnd = 0;
  for (const [rawStart, end] of [...spans].sort((a, b) => a[0] - b[0] || a[1] - b[1])) {
    const start = Math.max(rawStart, lastEnd);
    if (end - start <= 0) continue;
    const first = Math.round(start * sampleRate);
    const last = Math.round(end * sampleRate);
    const part = signal.subarray(first, last);
    if (part.length === 0) continue;
    const length = part.length / sampleRate;
    map.push([position, position + length, first / sampleRate]);
    parts.push(part);
    position += length;
    lastEnd = end;
  }
  if (parts.length === 0) return { clipped: signal, map: [] };
  const clipped = new Float32Array(parts.reduce((n, p) => n + p.length, 0));
  let offset = 0;
  for (const part of parts) {
    clipped.set(part, offset);
    offset += part.length;
  }
  return { clipped, map };
}

/**
 * A time on the clipped audio, mapped back onto the original. A time past a span's end (a timestamp
 * can land a few milliseconds after the audio it closes) is held at that span's end.
 */
export function restoreTime(t: number, map: SpeechMap): number {
  if (map.length === 0) return t;
  let index = 0;
  for (let i = 0; i < map.length; i += 1) if (map[i]![0] <= t) index = i;
  const [clipStart, clipEnd, originalStart] = map[index]!;
  const local = Math.min(Math.max(t - clipStart, 0), clipEnd - clipStart);
  return Math.round((originalStart + local) * 1000) / 1000;
}
