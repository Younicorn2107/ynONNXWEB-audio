/**
 * The live contract (spec § 4.2, § 6.3): what every live run publishes, the Python libraries' own
 * shape, so a page reads a local session and a server's `WS /stream` the same way.
 */

import { fromPythonNames, toPythonNames } from "@youni/onnxweb-core";

import { StreamingError } from "./errors.js";

export type LiveEventKind = "partial" | "final" | "retract";

export interface LiveEvent {
  kind: LiveEventKind;
  streamId: string;
  /** Absent for an event about the whole stream rather than one unit. */
  segmentId?: string;
  /** 0 at the unit's first emission, then +1 at each event about it. */
  revision: number;
  /** `kind !== "partial"`, spelled out as the spec has it. */
  final: boolean;
  /**
   * Who produced it: `single` / `fast` / `large` / `post_final` for audio, the library or the step
   * otherwise.
   */
  producer: string;
  mode?: string;
  /** Which input of a batch the unit belongs to (batches finish out of order). */
  itemId?: string;
  text?: string;
  start?: number;
  end?: number;
  data: Record<string, unknown>;
}

/** What a unit carries: everything an event has beyond its identity. */
export interface LiveContent {
  producer?: string;
  itemId?: string;
  text?: string;
  start?: number;
  end?: number;
  data?: Record<string, unknown>;
}

/** The wire form: one JSON object in the Python field names, absent fields left out. */
export function toWire(event: LiveEvent): string {
  return JSON.stringify(toPythonNames(event, ["data"]));
}

/** One Server-Sent Events message (`event:` is the kind), as the Python servers send it. */
export function toSse(event: LiveEvent): string {
  return `event: ${event.kind}\ndata: ${toWire(event)}\n\n`;
}

/** An event read from the wire (a server's `WS /stream` or SSE route). */
export function fromWire(json: string | Record<string, unknown>): LiveEvent {
  const raw = typeof json === "string" ? (JSON.parse(json) as Record<string, unknown>) : json;
  const event = fromPythonNames(raw, ["data"]) as LiveEvent;
  if (event.kind !== "partial" && event.kind !== "final" && event.kind !== "retract") throw new StreamingError(`'${String(event.kind)}' is not a live event kind`);
  return { ...event, data: event.data ?? {} };
}

function token(bytes: number): string {
  const out = new Uint8Array(bytes);
  globalThis.crypto.getRandomValues(out);
  return Array.from(out, (b) => b.toString(16).padStart(2, "0")).join("");
}

/**
 * Revisions and protocol for the units of one stream: each unit's revision counted, a `final`
 * without content carrying the unit's last state, and nothing accepted after a unit's `final` or
 * `retract`.
 */
export class LiveEmitter {
  readonly streamId: string;
  readonly #revisions = new Map<string, number>();
  readonly #last = new Map<string, LiveEvent>();
  readonly #finished = new Set<string>();
  readonly #listeners: ((event: LiveEvent) => void)[] = [];
  #queue: LiveEvent[] = [];

  constructor(
    streamId?: string,
    readonly producer = "single",
    readonly mode?: string,
  ) {
    this.streamId = streamId ?? token(8);
  }

  /** Called with every event, in order, as it is emitted. */
  on(listener: (event: LiveEvent) => void): () => void {
    this.#listeners.push(listener);
    return () => {
      const at = this.#listeners.indexOf(listener);
      if (at >= 0) this.#listeners.splice(at, 1);
    };
  }

  /** The events emitted since the last call, in order. */
  events(): LiveEvent[] {
    const out = this.#queue;
    this.#queue = [];
    return out;
  }

  /** Units emitted and not final yet, in emission order. */
  get openUnits(): string[] {
    return [...this.#last.keys()].filter((unit) => !this.#finished.has(unit));
  }

  /** The unit as it now stands; a new `segmentId` when none is given. */
  partial(segmentId: string | null, content: LiveContent = {}): LiveEvent {
    return this.#emit("partial", segmentId ?? token(6), content);
  }

  /** The unit's last state. Without content, its last partial made final. */
  final(segmentId: string | null, content: LiveContent = {}): LiveEvent {
    return this.#emit("final", segmentId ?? token(6), content);
  }

  /** Withdraw a unit already emitted. */
  retract(segmentId: string, producer?: string): LiveEvent {
    if (!this.#last.has(segmentId)) throw new StreamingError(`retract of '${segmentId}', never emitted`);
    return this.#emit("retract", segmentId, producer ? { producer } : {});
  }

  /** Every unit still open made final, in emission order. */
  close(): LiveEvent[] {
    return this.openUnits.map((unit) => this.final(unit));
  }

  #emit(kind: LiveEventKind, unit: string, content: LiveContent): LiveEvent {
    if (this.#finished.has(unit)) throw new StreamingError(`'${unit}' is final: no ${kind} may follow`);
    const revision = (this.#revisions.get(unit) ?? -1) + 1;
    this.#revisions.set(unit, revision);
    const previous = this.#last.get(unit);
    const { producer, ...rest } = content;

    const carried: LiveContent =
      kind === "final" && previous
        ? {
            ...(previous.itemId !== undefined ? { itemId: previous.itemId } : {}),
            ...(previous.text !== undefined ? { text: previous.text } : {}),
            ...(previous.start !== undefined ? { start: previous.start } : {}),
            ...(previous.end !== undefined ? { end: previous.end } : {}),
            data: previous.data,
          }
        : {};
    const body = kind === "retract" ? {} : rest;
    const event: LiveEvent = {
      kind,
      streamId: this.streamId,
      segmentId: unit,
      revision,
      final: kind !== "partial",
      producer: producer ?? previous?.producer ?? this.producer,
      ...(this.mode !== undefined ? { mode: this.mode } : {}),
      ...carried,
      ...body,

      data: body.data ?? carried.data ?? {},
    };
    this.#last.set(unit, event);
    if (event.final) this.#finished.add(unit);
    this.#queue.push(event);
    for (const listener of this.#listeners) listener(event);
    return event;
  }
}
