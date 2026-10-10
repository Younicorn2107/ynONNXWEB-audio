/** The timeline every analysis writes into (spec § 4). */

import { ConfigurationError } from "@youni/onnxweb-core/core";

export type AnnotationKind = "speech" | "language" | "speaker" | "emotion" | "event" | "music" | "word" | "profile";
export type Granularity = "frame" | "word" | "segment" | "clip";

/** The version of the timeline's schema and of its canonical labels (spec § 5). */
export const TIMELINE_SCHEMA_VERSION = "1.1";

export interface Annotation {
  /** Stable within the result: `<kind>-<n>`, numbered in time order per kind. */
  id: string;
  kind: AnnotationKind;
  /** Seconds on the ORIGINAL audio; the interval is half-open, [start, end). */
  start: number;
  end: number;
  /** The canonical label: `"fr"`, `"SPEAKER_00"`, `"applause"`, a word… */
  label: string | null;
  /** A number, a JSON object (the profile's measurements), or null. */
  value: number | Record<string, unknown> | null;
  /** 0..1, or null. */
  confidence: number | null;
  /** The model, service or measurement that produced it. */
  source: string;
  granularity: Granularity;
  /** The speaker concerned (`SPEAKER_00`…), or null. */
  speakerId: string | null;
  /** The model's own label when it differs from the canonical one. */
  sourceLabel?: string;
}

export interface TimelineData {
  schemaVersion: string;
  annotations: Annotation[];
}

const KIND_ORDER: readonly AnnotationKind[] = ["speech", "language", "speaker", "emotion", "event", "music", "word", "profile"];

export class Timeline {
  readonly annotations: Annotation[] = [];
  readonly #counts = new Map<AnnotationKind, number>();

  /** Add an annotation; its id is given here. Rejects an empty or reversed interval. */
  add(
    annotation: Omit<Annotation, "id" | "value" | "confidence" | "label" | "speakerId"> &
      Partial<Pick<Annotation, "id" | "value" | "confidence" | "label" | "speakerId">>,
  ): Annotation {
    if (!(annotation.start >= 0) || !(annotation.end > annotation.start)) {
      if (annotation.end === annotation.start) return { id: "", label: null, value: null, confidence: null, speakerId: null, ...annotation };
      throw new ConfigurationError(`an annotation needs 0 <= start < end, got [${annotation.start}, ${annotation.end})`);
    }
    const count = this.#counts.get(annotation.kind) ?? 0;
    this.#counts.set(annotation.kind, count + 1);
    const full: Annotation = { label: null, value: null, confidence: null, speakerId: null, ...annotation, id: annotation.id ?? `${annotation.kind}-${count}` };
    this.annotations.push(full);
    return full;
  }

  /** The annotations in the contractual order: (start, end, kind, id). */
  sorted(): Annotation[] {
    return [...this.annotations].sort(
      (a, b) =>
        a.start - b.start ||
        a.end - b.end ||
        KIND_ORDER.indexOf(a.kind) - KIND_ORDER.indexOf(b.kind) ||
        (a.id < b.id ? -1 : a.id > b.id ? 1 : 0),
    );
  }

  of(kind: AnnotationKind): Annotation[] {
    return this.sorted().filter((a) => a.kind === kind);
  }

  /** The annotations of one kind overlapping [start, end). */
  during(kind: AnnotationKind, start: number, end: number): Annotation[] {
    return this.of(kind).filter((a) => a.start < end && a.end > start);
  }

  toJSON(): TimelineData {
    return { schemaVersion: TIMELINE_SCHEMA_VERSION, annotations: this.sorted() };
  }

  static from(data: TimelineData | readonly Annotation[]): Timeline {
    const timeline = new Timeline();
    for (const a of Array.isArray(data) ? data : (data as TimelineData).annotations) timeline.add(a);
    return timeline;
  }
}

/** An empty timeline, as a result carries it when nothing was asked for. */
export function emptyTimeline(): TimelineData {
  return { schemaVersion: TIMELINE_SCHEMA_VERSION, annotations: [] };
}

/** What `analyze` accepts (spec § 2.2). */
export const ANALYSES = ["speech", "profile", "language", "speakers", "words", "emotion", "events", "music"] as const;
export type Analysis = (typeof ANALYSES)[number];

/** The analyses this library can run today: the others need a model not shipped yet. */
export const AVAILABLE_ANALYSES: ReadonlySet<Analysis> = new Set(["speech", "profile", "language", "speakers", "words", "emotion", "events", "music"]);

function closest(value: string, candidates: readonly string[]): string[] {
  const distance = (a: string, b: string) => {
    const d = Array.from({ length: a.length + 1 }, (_, i) => [i, ...new Array<number>(b.length).fill(0)]);
    for (let j = 1; j <= b.length; j += 1) d[0]![j] = j;
    for (let i = 1; i <= a.length; i += 1) {
      for (let j = 1; j <= b.length; j += 1) {
        d[i]![j] = Math.min(d[i - 1]![j]! + 1, d[i]![j - 1]! + 1, d[i - 1]![j - 1]! + (a[i - 1] === b[j - 1] ? 0 : 1));
      }
    }
    return d[a.length]![b.length]!;
  };
  return [...candidates].sort((x, y) => distance(value, x) - distance(value, y)).slice(0, 2);
}

/**
 * The analyses asked for, checked BEFORE any download: an unknown value is refused with the close
 * ones, and one no shipped model serves is refused saying so.
 */
export function checkAnalyses(asked: readonly string[] | null | undefined): Analysis[] {
  const out: Analysis[] = [];
  for (const value of asked ?? []) {
    if (!(ANALYSES as readonly string[]).includes(value)) {
      throw new ConfigurationError(`'${value}' is not an analysis; did you mean ${closest(value, ANALYSES).map((c) => `'${c}'`).join(" or ")}? (${ANALYSES.join(", ")})`);
    }
    const analysis = value as Analysis;
    if (!AVAILABLE_ANALYSES.has(analysis)) {
      throw new ConfigurationError(
        `'${analysis}' needs a model this library does not ship yet; available: ${[...AVAILABLE_ANALYSES].join(", ")}`,
      );
    }
    if (!out.includes(analysis)) out.push(analysis);
  }
  return out;
}
