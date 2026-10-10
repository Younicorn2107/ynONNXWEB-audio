/** Attaching punctuation to the word it belongs to. */

export interface Word {
  word: string;
  start: number;
  end: number;
  probability: number;
}

export const PREPEND_PUNCTUATIONS = "\"'“¿([{-";
export const APPEND_PUNCTUATIONS = "\"'.。,，!！?？:：”)]}、";

const round3 = (x: number) => Math.round(x * 1000) / 1000;

/**
 * Fold punctuation-only entries into their neighbour. The merged word keeps the OUTER bounds of
 * both pieces: the audio a word covers never shrinks because a comma was glued to it.
 */
export function mergePunctuation(words: readonly Word[], prepend = PREPEND_PUNCTUATIONS, append = APPEND_PUNCTUATIONS): Word[] {
  const out: Word[] = [];
  let pending: Word | null = null;
  const only = (text: string, set: string) => [...text].every((c) => set.includes(c));
  for (let word of words) {
    const stripped = word.word.trim();
    if (stripped && only(stripped, prepend)) {
      pending = pending ? { word: pending.word + word.word, start: pending.start, end: word.end, probability: word.probability } : word;
      continue;
    }
    if (stripped && only(stripped, append) && out.length > 0) {
      const previous = out[out.length - 1]!;
      out[out.length - 1] = { word: previous.word + stripped, start: previous.start, end: Math.max(previous.end, word.end), probability: previous.probability };
      continue;
    }
    if (pending) {
      word = { word: pending.word.trim() + word.word, start: pending.start, end: word.end, probability: word.probability };
      pending = null;
    }
    out.push(word);
  }

  if (pending) out.push(pending);
  return out
    .filter((w) => w.word.trim())
    .map((w) => ({ word: w.word.trim(), start: round3(w.start), end: round3(Math.max(w.end, w.start)), probability: w.probability }));
}

/**
 * The words of each segment, by the word's MIDPOINT: a word straddling a boundary belongs to the
 * side it spends most of its time in.
 */
export function splitWordsIntoSegments<S extends { end: number; words?: Word[] | null }>(
  words: readonly Word[],
  segments: S[],
  tolerance = 0.05,
): void {
  if (words.length === 0 || segments.length === 0) return;
  let cursor = 0;
  for (const segment of segments) {
    const owned: Word[] = [];
    while (cursor < words.length) {
      const word = words[cursor]!;
      if ((word.start + word.end) / 2 > segment.end + tolerance) break;
      owned.push(word);
      cursor += 1;
    }
    segment.words = owned.length > 0 ? owned : null;
  }
}

/**
 * Word spans from a segment's own bounds, spread by word LENGTH: a proportional split, NOT an
 * alignment, for when nothing better is available.
 */
export function proportionalWords(segment: { start: number; end: number; text: string }): Word[] {
  const pieces = segment.text.split(/\s+/).filter(Boolean);
  if (pieces.length === 0) return [];
  const total = pieces.reduce((s, p) => s + [...p].length, 0);
  const duration = Math.max(segment.end - segment.start, 0);
  let cursor = segment.start;
  return pieces.map((piece) => {
    const span = (duration * [...piece].length) / total;
    const word = { word: piece, start: round3(cursor), end: round3(Math.min(cursor + span, segment.end)), probability: 0 };
    cursor += span;
    return word;
  });
}
