/** Renderings of an audio result (spec § 6). */

import { ExportError, toCsv, toJson, toSrt, toTxt, toVtt, toXml } from "@youni/onnxweb-core/core";

import { displayName, type SpeakerInfo } from "./diarization/diarizer.js";
import type { Annotation, TimelineData } from "./timeline.js";

interface RenderableSegment {
  id?: number;
  start: number;
  end: number;
  text: string;
  speaker?: string | null;
}

export interface RenderableResult {
  text: string;
  language?: string | null;
  segments: readonly RenderableSegment[];
  timeline?: TimelineData | readonly Annotation[];
  /** Enrolled names by identity: shown in place of `SPEAKER_00`. */
  speakerInfo?: readonly SpeakerInfo[] | null;
}

function annotationsOf(result: RenderableResult): readonly Annotation[] {
  const t = result.timeline;
  if (!t) return [];
  return Array.isArray(t) ? t : (t as TimelineData).annotations;
}

/** How a speaker is shown: the enrolled name, else "Speaker N". */
function shownAs(result: RenderableResult, speaker: string): string {
  return displayName(speaker, result.speakerInfo?.find((i) => i.speakerId === speaker)?.name ?? null);
}

export const RENDER_FORMATS = ["json", "srt", "vtt", "txt", "csv", "xml", "md"] as const;
export type RenderFormat = (typeof RENDER_FORMATS)[number];

/** The label of the annotations of `kind` overlapping a segment the most, or null. */
function labelDuring(timeline: readonly Annotation[], kind: Annotation["kind"], start: number, end: number): string | null {
  let best: Annotation | null = null;
  let bestOverlap = 0;
  for (const a of timeline) {
    if (a.kind !== kind) continue;
    const overlap = Math.min(a.end, end) - Math.max(a.start, start);
    if (overlap > bestOverlap) {
      best = a;
      bestOverlap = overlap;
    }
  }
  return best?.label ?? null;
}

function spoken(result: RenderableResult): RenderableSegment[] {
  return result.segments.map((s) => ({ ...s, text: s.speaker ? `${shownAs(result, s.speaker)}: ${s.text.trim()}` : s.text.trim(), speaker: null }));
}

/** `mm:ss` (or `h:mm:ss`). */
function clock(seconds: number): string {
  const t = Math.max(0, Math.round(seconds));
  const h = Math.floor(t / 3600);
  const m = Math.floor((t % 3600) / 60);
  const ss = String(t % 60).padStart(2, "0");
  return h > 0 ? `${h}:${String(m).padStart(2, "0")}:${ss}` : `${String(m).padStart(2, "0")}:${ss}`;
}

/**
 * Markdown by speaker turn: a line on the whole (length, languages, speakers), then a heading per
 * change of speaker with its start, its emotion and its language when they were analysed; music and
 * sound events in brackets where they happen — never inside the spoken text.
 */
export function renderMarkdown(result: RenderableResult): string {
  const timeline = annotationsOf(result);
  const lines: string[] = [];
  const end = result.segments.length > 0 ? result.segments[result.segments.length - 1]!.end : 0;
  const languages = [...new Set(timeline.filter((a) => a.kind === "language" && a.label).map((a) => a.label!))];
  if (languages.length === 0 && result.language) languages.push(result.language);
  const speakers = new Set(result.segments.map((s) => s.speaker).filter((s): s is string => !!s));
  const summary = [`${clock(end)}`, languages.length > 0 ? languages.join(", ") : null, speakers.size > 0 ? `${speakers.size} speaker${speakers.size > 1 ? "s" : ""}` : null].filter(Boolean);
  lines.push(`_${summary.join(" · ")}_`, "");
  let current: string | null | undefined;
  const several = languages.length > 1;
  for (const segment of result.segments) {
    const speaker = segment.speaker ?? null;
    if (speaker !== null && speaker !== current) {
      const emotion = labelDuring(timeline, "emotion", segment.start, segment.end);
      const language = several ? labelDuring(timeline, "language", segment.start, segment.end) : null;
      const notes = [emotion, language].filter(Boolean).join(", ");
      if (lines[lines.length - 1] !== "") lines.push("");
      lines.push(`## ${shownAs(result, speaker)}${notes ? ` (${notes})` : ""} · ${clock(segment.start)}`);
    }
    current = speaker;
    for (const event of timeline.filter((a) => (a.kind === "music" || a.kind === "event") && a.start >= segment.start && a.start < segment.end)) {
      lines.push(event.kind === "music" ? `[music${event.label ? ` : "${event.label}"` : ""}]` : `[${event.label ?? "event"}]`);
    }
    lines.push(segment.text.trim());
  }
  return `${lines.join("\n").trimEnd()}\n`;
}

/** One row per segment, with speaker, language and emotion when they exist. */
export function renderCsv(result: RenderableResult): string {
  const timeline = annotationsOf(result);
  return toCsv(
    result.segments.map((s) => {
      const row: Record<string, unknown> = { id: s.id ?? null, start: s.start, end: s.end };
      if (s.speaker) row["speaker"] = shownAs(result, s.speaker);
      const language = labelDuring(timeline, "language", s.start, s.end) ?? result.language ?? null;
      if (language) row["language"] = language;
      const emotion = labelDuring(timeline, "emotion", s.start, s.end);
      if (emotion) row["emotion"] = emotion;
      row["text"] = s.text.trim();
      return row;
    }),
  );
}

/** A result in one format. */
export function renderResult(result: RenderableResult, format: string): string {
  switch (format.toLowerCase().replace(/^\./, "")) {
    case "json":
      return toJson(result);
    case "xml":
      return toXml(result);
    case "srt":
      return toSrt(spoken(result));
    case "vtt":
      return toVtt(spoken(result));
    case "txt":
    case "text":
      return toTxt(result);
    case "csv":
      return renderCsv(result);
    case "md":
    case "markdown":
      return renderMarkdown(result);
    default:
      throw new ExportError(`Unknown format '${format}'. Known: ${RENDER_FORMATS.join(", ")}`);
  }
}
