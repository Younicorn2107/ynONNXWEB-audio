/** Getting a file the user dropped into the mono float32 the mel expects. */

import { AudioLoadError } from "@youni/onnxweb-core/core";

export const TARGET_RATE = 16000;

/** Linear resampling to 16 kHz. */
export function resampleTo16k(input: Float32Array, inputRate: number): Float32Array {
  if (inputRate === TARGET_RATE) return input;
  if (inputRate <= 0) throw new AudioLoadError(`invalid sample rate ${inputRate}`);

  const ratio = inputRate / TARGET_RATE;
  const length = Math.floor(input.length / ratio);
  const out = new Float32Array(length);
  for (let i = 0; i < length; i += 1) {
    const position = i * ratio;
    const index = Math.floor(position);
    const fraction = position - index;
    const a = input[index] ?? 0;
    const b = input[index + 1] ?? a;
    out[i] = a + (b - a) * fraction;
  }
  return out;
}

/** Decode any browser-playable audio to mono 16 kHz float32. */
export async function decodeAudio(
  data: ArrayBuffer | Blob | File,
  options?: { sampleRate?: number },
): Promise<{ audio: Float32Array; duration: number }> {
  const target = options?.sampleRate ?? TARGET_RATE;

  const Offline =
    (globalThis as { OfflineAudioContext?: typeof OfflineAudioContext }).OfflineAudioContext ??
    (globalThis as { webkitOfflineAudioContext?: typeof OfflineAudioContext }).webkitOfflineAudioContext;
  const Ctor =
    (globalThis as { AudioContext?: typeof AudioContext; webkitAudioContext?: typeof AudioContext })
      .AudioContext ??
    (globalThis as { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;
  if (!Offline && !Ctor) {
    throw new AudioLoadError(
      "no AudioContext in this environment, so audio cannot be decoded here. " +
        "In Node, decode the file yourself and pass a Float32Array to transcribe().",
    );
  }

  const buffer = data instanceof ArrayBuffer ? data : await data.arrayBuffer();
  const context: BaseAudioContext = Offline ? new Offline(1, 1, target) : new Ctor!();
  try {
    const decoded = await context.decodeAudioData(buffer.slice(0));
    const channels: Float32Array[] = [];
    for (let c = 0; c < decoded.numberOfChannels; c += 1) {
      channels.push(decoded.getChannelData(c));
    }
    const length = decoded.length;
    const mono = new Float32Array(length);
    for (const channel of channels) {
      for (let i = 0; i < length; i += 1) mono[i] = mono[i]! + channel[i]!;
    }
    if (channels.length > 1) {
      for (let i = 0; i < length; i += 1) mono[i] = mono[i]! / channels.length;
    }

    const audio = target === decoded.sampleRate ? mono : resampleTo16k(mono, decoded.sampleRate);

    return { audio, duration: audio.length / target };
  } catch (cause) {
    if (cause instanceof AudioLoadError) throw cause;
    throw new AudioLoadError(
      `the browser could not decode this audio: ${String(cause)}. ` +
        `Supported formats are whatever this browser can play.`,
      { cause },
    );
  } finally {
    const closable = context as BaseAudioContext & { close?: () => Promise<void> };
    await closable.close?.().catch(() => undefined);
  }
}
