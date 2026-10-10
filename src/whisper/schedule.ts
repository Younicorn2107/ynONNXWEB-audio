/** Planning what to decode, and in what order — the throughput half. */

import { SchedulingError } from "../errors.js";
import { WINDOW_SECONDS } from "../audio/mel.js";

export const MIN_SECTION_S = 90;
export const DEFAULT_OVERLAP_S = 5;

export interface AudioInfo {
  id: string;
  duration: number;
}

export interface Section {
  audioId: string;
  index: number;
  start: number;
  end: number;
  windows: [number, number][];
  /**
   * A time no window of this section may read past (where the language changes): windows stop there
   * instead of reading 30 s on.
   */
  limit?: number;
}

export interface WindowResult {
  audioId: string;
  sectionIndex: number;
  windowStart: number;
  tokens: number[];
  text: string;
  avgLogprob: number;
  compressionRatio: number;
  noSpeechProb: number;
  isSilent: boolean;
  /** End of the last CLOSED segment, or null when the window closed none. */
  lastSegmentEnd: number | null;
  temperature: number;
}

export class SectionState {
  seek: number;
  context: number[] = [];
  results: WindowResult[] = [];
  done = false;

  constructor(readonly section: Section) {
    this.seek = section.start;
  }

  get audioId(): string {
    return this.section.audioId;
  }

  /** Every window start from the seek to the end — batched mode only. */
  remainingWindows(chunkS: number = WINDOW_SECONDS): number[] {
    const starts: number[] = [];
    let cursor = this.seek;
    while (cursor < this.section.end - 1e-3) {
      starts.push(Number(cursor.toFixed(3)));
      cursor += chunkS;
    }
    return starts;
  }
}

/** Cuts audios into sections, and sections into windows. */
export class SectionScheduler {
  /**
   * Sections inside spans that must not be crossed (the portions of an audio that changes
   * language): each span is cut as an audio of its own would be — `budget` sections shared by
   * duration, none under 90 s, overlapping by `overlap` — and no window of a span's sections reads
   * past the span's end (`limit`). Within a span, sections still overlap and carry their prompt;
   * between two spans the cut is clean.
   */
  planSpans(
    audio: AudioInfo,
    spans: readonly { start: number; end: number }[],
    offset: number,
    options: { budget?: number; overlap?: number; chunkS?: number } = {},
  ): Section[] {
    const chunkS = options.chunkS ?? WINDOW_SECONDS;
    const overlap = options.overlap ?? DEFAULT_OVERLAP_S;
    const kept = spans.map((s) => ({ start: s.start, end: Math.min(s.end, audio.duration) })).filter((s) => s.end > s.start);
    const total = kept.reduce((sum, s) => sum + (s.end - s.start), 0) || 1;
    const budget = Math.max(1, Math.trunc(options.budget ?? 1));
    const out: Section[] = [];
    for (const span of kept) {
      const length = span.end - span.start;
      const count = this.sectionCount(length, Math.max(1, Math.round((budget * length) / total)));
      const step = length / count;
      for (let i = 0; i < count; i += 1) {
        const start = span.start + i * step;
        const end = i === count - 1 ? span.end : Math.min(span.end, start + step + overlap);
        out.push({
          audioId: audio.id,
          index: offset + out.length,
          start: Number(start.toFixed(3)),
          end: Number(end.toFixed(3)),
          windows: this.windows(start, end, chunkS),
          limit: Number(span.end.toFixed(3)),
        });
      }
    }
    return out;
  }

  plan(
    audios: readonly AudioInfo[],
    options: { totalBatch?: number; overlap?: number; chunkS?: number } = {},
  ): Section[] {
    if (audios.length === 0) return [];
    const budget = Math.max(1, Math.trunc(options.totalBatch ?? 1));
    const overlap = options.overlap ?? DEFAULT_OVERLAP_S;
    const chunkS = options.chunkS ?? WINDOW_SECONDS;

    const totalDuration = audios.reduce((sum, a) => sum + Math.max(a.duration, 0), 0) || 1;
    const sections: Section[] = [];
    for (const audio of audios) {
      const duration = Math.max(audio.duration, 0);
      const share = Math.max(1, Math.round((budget * duration) / totalDuration));
      const count = this.sectionCount(duration, share);
      sections.push(...this.cut(audio, count, overlap, chunkS, sections.length));
    }
    return sections;
  }

  /** How many sections this duration carries without dropping below 90 s. */
  private sectionCount(duration: number, requested: number): number {
    if (duration <= MIN_SECTION_S) return 1;
    const ceiling = Math.trunc(duration / MIN_SECTION_S);
    return Math.max(1, Math.min(requested, ceiling));
  }

  private cut(
    audio: AudioInfo,
    count: number,
    overlap: number,
    chunkS: number,
    offset: number,
  ): Section[] {
    const duration = Math.max(audio.duration, 0);
    const span = count ? duration / count : duration;
    const out: Section[] = [];
    for (let index = 0; index < count; index += 1) {
      const start = index * span;
      const end =
        index === count - 1 ? duration : Math.min(duration, start + span + overlap);
      out.push({
        audioId: audio.id,
        index: offset + index,
        start: Number(start.toFixed(3)),
        end: Number(end.toFixed(3)),
        windows: this.windows(start, end, chunkS),
      });
    }
    return out;
  }

  private windows(start: number, end: number, chunkS: number): [number, number][] {
    const out: [number, number][] = [];
    let cursor = start;
    while (cursor < end) {
      out.push([Number(cursor.toFixed(3)), Number(Math.min(cursor + chunkS, end).toFixed(3))]);
      cursor += chunkS;
    }
    return out.length > 0 ? out : [[Number(start.toFixed(3)), Number(end.toFixed(3))]];
  }
}

/** Groups the active window of every live section into one call. */
export class WaveScheduler {
  constructor(
    readonly chunkS: number = WINDOW_SECONDS,
    readonly batched = false,
  ) {}

  states(sections: readonly Section[]): SectionState[] {
    return sections.map((section) => new SectionState(section));
  }

  /** Yield the next wave until every section is done. */
  *waves(states: SectionState[], maxWindows: number): Generator<SectionState[]> {
    const cap = Math.max(1, Math.trunc(maxWindows));
    let cursor = 0;
    let stalled = 0;
    let seen: Map<SectionState, number> | null = null;

    for (;;) {
      const live = states.filter((state) => !state.done);
      if (live.length === 0) return;

      if (cursor >= live.length) {
        cursor = 0;
        const positions = new Map(live.map((state) => [state, state.seek] as const));
        const unchanged =
          seen !== null &&
          positions.size === seen.size &&
          [...positions].every(([state, seek]) => seen!.get(state) === seek);
        if (unchanged) {
          stalled += 1;
          if (stalled >= 2) {
            throw new SchedulingError(
              `no section advanced over two full passes; ${live.length} still open at ` +
                `${live.map((s) => s.seek.toFixed(2)).join(", ")}. Refusing to loop for ` +
                `ever — the caller must advance or skip every window it is handed.`,
            );
          }
        } else {
          stalled = 0;
        }
        seen = positions;
      }

      const wave = live.slice(cursor, cursor + cap);
      if (wave.length === 0) return;
      cursor += wave.length;
      yield wave;
    }
  }

  /** Move the seek forward and update the carried context. */
  advance(state: SectionState, result: WindowResult): void {
    const end = result.lastSegmentEnd;
    const floor = state.seek + this.chunkS * 0.5;
    if (end === null || end <= floor) state.seek += this.chunkS;
    else state.seek = Math.min(end, state.seek + this.chunkS);

    if (result.tokens.length > 0 && !result.isSilent) state.context = [...result.tokens];
    if (state.seek >= state.section.end - 1e-3) state.done = true;
  }

  /** Consume a window that produced nothing, keeping the section moving. */
  skip(state: SectionState): void {
    state.seek += this.chunkS;
    if (state.seek >= state.section.end - 1e-3) state.done = true;
  }

  /** Group every window result by the audio it belongs to. */
  static collect(states: readonly SectionState[]): Map<string, WindowResult[]> {
    const byAudio = new Map<string, WindowResult[]>();
    for (const state of states) {
      const list = byAudio.get(state.audioId) ?? [];
      list.push(...state.results);
      byAudio.set(state.audioId, list);
    }
    for (const results of byAudio.values()) {
      results.sort((a, b) =>
        a.sectionIndex === b.sectionIndex
          ? a.windowStart - b.windowStart
          : a.sectionIndex - b.sectionIndex,
      );
    }
    return byAudio;
  }
}
