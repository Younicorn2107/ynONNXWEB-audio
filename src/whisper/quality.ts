/** Deciding whether a decoded window is good enough to keep. */

export const TEMPERATURE_SCALE: readonly number[] = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0];

export interface QualityConfig {
  compressionRatioThreshold: number;
  logProbThreshold: number;
  noSpeechThreshold: number;
}

export const DEFAULT_QUALITY: QualityConfig = {
  compressionRatioThreshold: 2.4,
  logProbThreshold: -1.0,
  noSpeechThreshold: 0.6,
};

/** Sum divided by (n + 1) — the faster-whisper formula, kept verbatim. */
export function avgLogprob(tokenLogprobs: readonly number[]): number {
  if (tokenLogprobs.length === 0) return 0;
  let total = 0;
  for (const value of tokenLogprobs) total += value;
  return total / (tokenLogprobs.length + 1);
}

/** The same formula when only the cumulative sum survived the search. */
export function avgLogprobFromSum(total: number, tokenCount: number): number {
  if (tokenCount <= 0) return 0;
  return total / (tokenCount + 1);
}

/** `len(utf-8) / len(deflate(utf-8))`. */
export function compressionRatioSync(text: string): number {
  if (text.length === 0) return 0;
  const bytes = new TextEncoder().encode(text);
  if (bytes.length === 0) return 0;

  const window = 1024;
  const minMatch = 4;
  let literals = 0;
  let i = 0;
  while (i < bytes.length) {
    let best = 0;
    const start = Math.max(0, i - window);
    for (let candidate = start; candidate < i; candidate += 1) {
      let length = 0;
      while (
        length < 255 &&
        i + length < bytes.length &&
        bytes[candidate + length] === bytes[i + length]
      ) {
        length += 1;
      }
      if (length > best) best = length;
      if (best >= 64) break;
    }
    if (best >= minMatch) {
      literals += 3;
      i += best;
    } else {
      literals += 1;
      i += 1;
    }
  }
  return bytes.length / Math.max(literals, 1);
}

/** The real ratio, using the browser's deflate when it exists. */
export async function compressionRatio(text: string): Promise<number> {
  if (text.length === 0) return 0;
  const bytes = new TextEncoder().encode(text);
  const CS = (globalThis as { CompressionStream?: new (f: string) => unknown })
    .CompressionStream;
  if (!CS) return compressionRatioSync(text);
  try {
    const stream = new Response(
      new Blob([bytes as unknown as BlobPart])
        .stream()
        .pipeThrough(new (CS as new (f: string) => TransformStream)("deflate")),
    );
    const compressed = await stream.arrayBuffer();
    return bytes.length / Math.max(compressed.byteLength, 1);
  } catch {
    return compressionRatioSync(text);
  }
}

export interface WindowScore {
  text: string;
  avgLogprob: number;
  compressionRatio: number;
  noSpeechProb: number;
}

/** Too compressible OR too improbable — retry at a warmer temperature. */
export function isDegenerate(result: WindowScore, config: QualityConfig = DEFAULT_QUALITY): boolean {
  if (result.compressionRatio > config.compressionRatioThreshold) return true;
  return result.avgLogprob < config.logProbThreshold;
}

/** Both conditions, never either one. */
export function isSilent(
  noSpeechProb: number,
  averageLogprob: number,
  config: QualityConfig = DEFAULT_QUALITY,
): boolean {
  if (noSpeechProb <= config.noSpeechThreshold) return false;
  return averageLogprob < config.logProbThreshold;
}

/** The temperature ladder: try cold, warm up only if the result is degenerate. */
export async function withTemperatureFallback<T extends WindowScore>(
  attempt: (temperature: number) => Promise<T>,
  options?: { scale?: readonly number[]; quality?: QualityConfig },
): Promise<{ result: T; temperature: number; attempts: number }> {
  const scale = options?.scale ?? TEMPERATURE_SCALE;
  const quality = options?.quality ?? DEFAULT_QUALITY;

  let last: T | null = null;
  let lastTemperature = scale[0] ?? 0;
  for (const [index, temperature] of scale.entries()) {
    const result = await attempt(temperature);
    last = result;
    lastTemperature = temperature;
    if (!isDegenerate(result, quality)) {
      return { result, temperature, attempts: index + 1 };
    }
  }

  return { result: last as T, temperature: lastTemperature, attempts: scale.length };
}
