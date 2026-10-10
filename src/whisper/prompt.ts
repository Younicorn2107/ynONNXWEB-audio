/** Assembling the decoder's opening tokens. */

import { assemblePrompt, quantizeContext } from "@youni/onnxweb-core/core";

import type { WhisperTokens } from "./tokens.js";

export const DEFAULT_MAX_CONTEXT = 224;

export interface PromptOptions {
  language?: string | null;
  task?: "transcribe" | "translate";
  withoutTimestamps?: boolean;
  /** Ids from the previous window, oldest first. */
  context?: readonly number[];
  maxContext?: number;
}

/** Build the prompt. */
export function buildPrompt(tokens: WhisperTokens, options: PromptOptions = {}): number[] {
  const start: number[] = [tokens.sot];

  if (tokens.isMultilingual) {
    const code = options.language ?? null;
    const id = code ? tokens.language(code) : null;
    if (id !== null) {
      start.push(id);
      const task = options.task === "translate" ? tokens.translate : tokens.transcribe;
      if (task !== null) start.push(task);
    }
  }

  if (options.withoutTimestamps === true && tokens.noTimestamps !== null) {
    start.push(tokens.noTimestamps);
  }

  const maxContext = options.maxContext ?? DEFAULT_MAX_CONTEXT;
  const context = options.context ?? [];
  if (context.length === 0 || maxContext <= 0 || tokens.sotPrev === null) return start;

  return assemblePrompt([tokens.sotPrev], quantizeContext(context), start, maxContext);
}

/** Strip a prompt's leading context so only the transcript remains. */
export function withoutContext(tokens: WhisperTokens, ids: readonly number[]): number[] {
  const sot = ids.lastIndexOf(tokens.sot);
  return sot === -1 ? [...ids] : ids.slice(sot);
}
