/** The Kaldi fbank features speaker embedders such as wespeaker expect. */

import { RealDFT } from "../audio/mel.js";

export const FBANK_SAMPLE_RATE = 16000;
const FRAME_LENGTH_MS = 25;
const FRAME_SHIFT_MS = 10;
const PREEMPHASIS = 0.97;
export const NUM_MEL_BINS = 80;
const LOW_FREQ = 20;
/** Kaldi's log floor (FLT_EPSILON). */
const EPSILON = 1.1920928955078125e-7;

const mel = (freq: number) => 1127 * Math.log(1 + freq / 700);

/** Kaldi's window: Povey's (a Hann raised to 0.85, the default) or a plain Hann ("hanning"). */
export type FbankWindow = "povey" | "hanning";

function frameWindow(size: number, kind: FbankWindow): Float32Array {
  const out = new Float32Array(size);
  const power = kind === "povey" ? 0.85 : 1;
  for (let n = 0; n < size; n += 1) out[n] = (0.5 - 0.5 * Math.cos((2 * Math.PI * n) / (size - 1))) ** power;
  return out;
}

/** Kaldi's triangular filters on the FFT bins, in mel coordinates, no area normalization. */
function filterBank(nFft: number, sampleRate: number, bins: number): Float32Array[] {
  const melLow = mel(LOW_FREQ);
  const melHigh = mel(sampleRate / 2);
  const step = (melHigh - melLow) / (bins + 1);
  const freqs = nFft / 2 + 1;
  const mels = Float64Array.from({ length: freqs }, (_, k) => mel((k * sampleRate) / nFft));
  return Array.from({ length: bins }, (_, i) => {
    const left = melLow + step * i;
    const centre = melLow + step * (i + 1);
    const right = melLow + step * (i + 2);
    const row = new Float32Array(freqs);
    for (let k = 0; k < freqs; k += 1) {
      const rising = (mels[k]! - left) / (centre - left);
      const falling = (right - mels[k]!) / (right - centre);
      row[k] = Math.max(0, Math.min(rising, falling));
    }
    return row;
  });
}

const banks = new Map<string, Float32Array[]>();

/**
 * `[frames, bins]` Kaldi-style log-mel coefficients, row-major; the mean over time removed (`cmn`).
 */
export function fbank(signal: Float32Array, sampleRate = FBANK_SAMPLE_RATE, bins = NUM_MEL_BINS, cmn = true, windowKind: FbankWindow = "povey"): { data: Float32Array; frames: number; bins: number } {
  const length = Math.round((sampleRate * FRAME_LENGTH_MS) / 1000);
  const shift = Math.round((sampleRate * FRAME_SHIFT_MS) / 1000);
  if (signal.length < length) return { data: new Float32Array(0), frames: 0, bins };
  const count = 1 + Math.floor((signal.length - length) / shift);
  let nFft = 1;
  while (nFft < length) nFft *= 2;
  const key = `${nFft}/${sampleRate}/${bins}`;
  if (!banks.has(key)) banks.set(key, filterBank(nFft, sampleRate, bins));
  const bank = banks.get(key)!;
  const window = frameWindow(length, windowKind);
  const dft = new RealDFT(nFft);
  const half = nFft / 2 + 1;
  const re = new Float64Array(half);
  const im = new Float64Array(half);
  const frame = new Float32Array(length);
  const padded = new Float64Array(nFft);
  const out = new Float32Array(count * bins);
  for (let f = 0; f < count; f += 1) {
    let mean = 0;
    for (let i = 0; i < length; i += 1) mean += signal[f * shift + i]!;
    mean = Math.fround(mean / length);
    for (let i = 0; i < length; i += 1) frame[i] = signal[f * shift + i]! - mean;

    for (let i = length - 1; i >= 0; i -= 1) {
      const previous = frame[i > 0 ? i - 1 : 0]!;
      padded[i] = Math.fround(Math.fround(frame[i]! - Math.fround(PREEMPHASIS * previous)) * window[i]!);
    }
    padded.fill(0, length);
    dft.run(padded, re, im);
    for (let b = 0; b < bins; b += 1) {
      const row = bank[b]!;
      let energy = 0;
      for (let k = 0; k < half; k += 1) energy += (re[k]! * re[k]! + im[k]! * im[k]!) * row[k]!;
      out[f * bins + b] = Math.log(Math.max(energy, EPSILON));
    }
  }
  if (cmn) {
    for (let b = 0; b < bins; b += 1) {
      let sum = 0;
      for (let f = 0; f < count; f += 1) sum += out[f * bins + b]!;
      const mean = sum / count;
      for (let f = 0; f < count; f += 1) out[f * bins + b] = out[f * bins + b]! - mean;
    }
  }
  return { data: out, frames: count, bins };
}
