/** Resuming a long transcription, wave by wave. */

import type { KeyValueStore } from "@youni/onnxweb-core/core";

import type { SectionState, WindowResult } from "./schedule.js";

/** Shown where a section is not finished yet. */
export const GAP = "[…]";

/** The readable text is rewritten at most this often. */
const TEXT_EVERY_MS = 2000;

interface Entry {
  section: string;
  seek: number;
  context: number[];
  results: WindowResult[];
}

const keyOf = (state: SectionState) => `${state.audioId}#${state.section.index}`;

function clock(seconds: number): string {
  const whole = Math.trunc(seconds);
  const pad = (n: number) => String(n).padStart(2, "0");
  return `${pad(Math.trunc(whole / 3600))}:${pad(Math.trunc((whole % 3600) / 60))}:${pad(whole % 60)}`;
}

export class WaveJournal {
  /** Per section: windows already journaled, and the seek they left. */
  readonly #saved = new Map<string, [number, number]>();
  #textWritten = Number.NEGATIVE_INFINITY;
  #next = 0;

  /**
   * `prefix` names this run (`transcribe/<fingerprint>`): another fingerprint is another journal.
   */
  constructor(
    readonly store: KeyValueStore,
    readonly prefix: string,
  ) {}

  async #entries(): Promise<Entry[]> {
    const keys = (await this.store.keys(`${this.prefix}/entry/`)).sort(
      (a, b) => Number(a.slice(a.lastIndexOf("/") + 1)) - Number(b.slice(b.lastIndexOf("/") + 1)),
    );
    this.#next = keys.length === 0 ? 0 : Number(keys[keys.length - 1]!.slice(keys[keys.length - 1]!.lastIndexOf("/") + 1)) + 1;
    const batches = await Promise.all(keys.map((key) => this.store.get(key)));
    return batches.flatMap((batch) => (Array.isArray(batch) ? (batch as Entry[]) : []));
  }

  /** Rewind the section states onto the journal; returns how many sections resumed. */
  async restore(states: readonly SectionState[]): Promise<number> {
    const bySection = new Map<string, Entry[]>();
    for (const entry of await this.#entries()) bySection.set(entry.section, [...(bySection.get(entry.section) ?? []), entry]);
    let resumed = 0;
    for (const state of states) {
      const found = bySection.get(keyOf(state));
      if (!found || found.length === 0) continue;
      const last = found[found.length - 1]!;
      state.seek = last.seek;
      state.context = [...last.context];
      state.results = found.flatMap((entry) => entry.results);
      state.done = state.seek >= state.section.end - 1e-3;
      this.#saved.set(keyOf(state), [state.results.length, state.seek]);
      resumed += 1;
    }
    return resumed;
  }

  /** Append what moved since the last save. */
  async save(states: readonly SectionState[]): Promise<void> {
    const entries: Entry[] = [];
    for (const state of states) {
      const key = keyOf(state);
      const [count, seek] = this.#saved.get(key) ?? [0, state.section.start];
      if (state.results.length === count && state.seek === seek) continue;
      entries.push({ section: key, seek: state.seek, context: [...state.context], results: state.results.slice(count) });
      this.#saved.set(key, [state.results.length, state.seek]);
    }
    if (entries.length > 0) {
      await this.store.set(`${this.prefix}/entry/${this.#next}`, entries);
      this.#next += 1;
    }
    const now = performance.now();
    if (now - this.#textWritten >= TEXT_EVERY_MS) {
      await this.store.set(`${this.prefix}/text`, WaveJournal.readable(states));
      this.#textWritten = now;
    }
  }

  /** The text so far, one paragraph per section, in time order. */
  static readable(states: readonly SectionState[]): string {
    const ordered = [...states].sort((a, b) => a.audioId.localeCompare(b.audioId) || a.section.start - b.section.start);
    const paragraphs = ordered.map((state) => {
      let text = state.results
        .map((r) => r.text.trim())
        .filter(Boolean)
        .join(" ");
      if (!state.done) text = `${text} ${GAP}`.trim();
      return `[${state.audioId} ${clock(state.section.start)}–${clock(state.section.end)}] ${text}`;
    });
    return `${paragraphs.join("\n\n")}\n`;
  }

  /** The readable text last written, or null. */
  text(): Promise<string | null> {
    return this.store.get(`${this.prefix}/text`).then((t) => (typeof t === "string" ? t : null));
  }

  /** The transcription finished: the journal and its text go. */
  async complete(): Promise<void> {
    for (const key of await this.store.keys(`${this.prefix}/`)) await this.store.delete(key);
  }
}
