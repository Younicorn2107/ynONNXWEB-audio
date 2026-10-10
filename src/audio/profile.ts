/** What an audio contains, measured before deciding how to transcribe it. */

import { RealDFT } from "./mel.js";

export const SAMPLE_RATE = 16000;
/** 20 ms: short enough to be speech or silence, long enough not to be noise. */
export const FRAME_MS = 20;
/** The high cut-off of a telephone line. Energy above it carries the fricatives. */
export const TELEPHONE_HIGH_HZ = 3400;
/** A frame is silent this many dB below the speech level (95th percentile of frame energies). */
export const SILENCE_BELOW_SPEECH_DB = 30;
/** Only silences at least this long count (pauses between words are speech). */
export const MIN_SILENCE_S = 1;
/** A sample is clipped from this absolute value up. */
export const CLIP_LEVEL = 0.99;

export interface AudioProfile {
  durationS: number;
  /**
   * Signal-to-noise ESTIMATE in dB: 90th against 10th percentile of frame energies, long silences
   * excluded. Not calibrated (no noise ~19, 20 dB added ~18, 10 dB ~13, 5 dB ~10), but monotonic.
   */
  snrEstimateDb: number;
  /** Share of the energy above 3 400 Hz. A telephone band keeps almost none. */
  energyAboveTelephone: number;
  /** The band actually present: above it the spectrum stays 50 dB below its peak. */
  bandwidthHz: number;
  speechLevelDbfs: number;
  peakDbfs: number;
  clippingRatio: number;
  /** Share of the duration in silences of at least one second. */
  silenceRatio: number;
  longestSilenceS: number;
}

/** Python's `round`: halves go to the even neighbour (5031.25 → 5031.2). */
const round = (value: number, digits: number) => {
  const scale = 10 ** digits;
  const scaled = value * scale;
  const floor = Math.floor(scaled);
  const diff = scaled - floor;
  const rounded = Math.abs(diff - 0.5) < 1e-9 ? (floor % 2 === 0 ? floor : floor + 1) : Math.round(scaled);
  return rounded / scale;
};

const db = (value: number, floor = 1e-12) => 10 * Math.log10(Math.max(value, floor));

/** `np.percentile` with linear interpolation. */
export function percentile(values: ArrayLike<number>, q: number): number {
  const sorted = Float64Array.from(values).sort();
  if (sorted.length === 0) return Number.NaN;
  const position = ((sorted.length - 1) * q) / 100;
  const low = Math.floor(position);
  const high = Math.min(low + 1, sorted.length - 1);
  return sorted[low]! + (sorted[high]! - sorted[low]!) * (position - low);
}

function energies(signal: Float32Array, frame: number): Float64Array {
  const count = Math.floor(signal.length / frame);
  const out = new Float64Array(count);
  for (let f = 0; f < count; f += 1) {
    let sum = 0;
    for (let i = f * frame; i < (f + 1) * frame; i += 1) sum += signal[i]! * signal[i]!;
    out[f] = sum / frame;
  }
  return out;
}

/** Keep only the runs of `true` at least `minLength` long. */
function longRuns(mask: readonly boolean[], minLength: number): boolean[] {
  const kept = new Array<boolean>(mask.length).fill(false);
  let start = -1;
  for (let i = 0; i <= mask.length; i += 1) {
    if (i < mask.length && mask[i]) {
      if (start < 0) start = i;
    } else if (start >= 0) {
      if (i - start >= minLength) kept.fill(true, start, i);
      start = -1;
    }
  }
  return kept;
}

function longestRun(mask: readonly boolean[]): number {
  let best = 0;
  let run = 0;
  for (const value of mask) {
    run = value ? run + 1 : 0;
    best = Math.max(best, run);
  }
  return best;
}

/**
 * Mean power spectrum of the louder half of the frames, at most 400 of them spread over the audio.
 */
function speechSpectrum(signal: Float32Array, frameEnergies: Float64Array, frame: number, sampleRate: number) {
  const nFft = 1024;
  const median = percentile(frameEnergies, 50);
  let speech: number[] = [];
  frameEnergies.forEach((e, i) => {
    if (e >= median) speech.push(i);
  });
  if (speech.length === 0) speech = Array.from(frameEnergies, (_, i) => i);
  const step = Math.max(1, Math.floor(speech.length / 400));

  const window = new Float64Array(nFft);
  for (let i = 0; i < nFft; i += 1) window[i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / (nFft - 1));
  const dft = new RealDFT(nFft);
  const bins = nFft / 2 + 1;
  const total = new Float64Array(bins);
  const re = new Float64Array(bins);
  const im = new Float64Array(bins);
  const chunk = new Float64Array(nFft);
  let used = 0;
  for (let k = 0; k < speech.length; k += step) {
    const begin = speech[k]! * frame;
    if (begin + nFft > signal.length) continue;
    for (let i = 0; i < nFft; i += 1) chunk[i] = signal[begin + i]! * window[i]!;
    dft.run(chunk, re, im);
    for (let b = 0; b < bins; b += 1) total[b] = total[b]! + re[b]! * re[b]! + im[b]! * im[b]!;
    used += 1;
  }
  const frequencies = Float64Array.from({ length: bins }, (_, b) => (b * sampleRate) / nFft);
  for (let b = 0; b < bins; b += 1) total[b] = total[b]! / Math.max(used, 1);
  return { frequencies, power: total };
}

/** Measure a mono signal. Too short for a percentile to mean anything: a neutral profile. */
export function profileAudio(signal: Float32Array, sampleRate = SAMPLE_RATE): AudioProfile {
  const frame = Math.max(1, Math.floor((FRAME_MS * sampleRate) / 1000));
  const duration = signal.length / sampleRate;
  const frameEnergies = energies(signal, frame);
  let peak = 0;
  let clipped = 0;
  for (const x of signal) {
    const a = Math.abs(x);
    if (a > peak) peak = a;
    if (a >= CLIP_LEVEL) clipped += 1;
  }

  if (frameEnergies.length < 50) {
    return {
      durationS: duration,
      snrEstimateDb: 0,
      energyAboveTelephone: 0,
      bandwidthHz: sampleRate / 2,
      speechLevelDbfs: -120,
      peakDbfs: signal.length ? db(peak * peak) : -120,
      clippingRatio: 0,
      silenceRatio: 0,
      longestSilenceS: 0,
    };
  }

  const reference = percentile(frameEnergies, 95);
  const threshold = Math.max(reference, 1e-12) * 10 ** (-SILENCE_BELOW_SPEECH_DB / 10);
  const belowSpeech = Array.from(frameEnergies, (e) => e < threshold);
  const longSilences = longRuns(belowSpeech, Math.max(1, Math.floor((MIN_SILENCE_S * 1000) / FRAME_MS)));

  const notSilent = frameEnergies.filter((_, i) => !longSilences[i]);
  const active = notSilent.length >= 50 ? notSilent : frameEnergies;
  const snr = db(percentile(active, 90)) - db(percentile(active, 10));
  const voiced = frameEnergies.filter((_, i) => !belowSpeech[i]);
  const speech = voiced.length > 0 ? voiced : frameEnergies;
  const speechMean = speech.reduce((s, e) => s + e, 0) / speech.length;

  const { frequencies, power } = speechSpectrum(signal, frameEnergies, frame, sampleRate);
  let totalPower = 0;
  let above = 0;
  let maxPower = 0;
  power.forEach((p, b) => {
    totalPower += p;
    if (frequencies[b]! > TELEPHONE_HIGH_HZ) above += p;
    if (p > maxPower) maxPower = p;
  });
  let bandwidth = 0;
  power.forEach((p, b) => {
    if (p >= maxPower * 10 ** (-50 / 10)) bandwidth = Math.max(bandwidth, frequencies[b]!);
  });

  const silent = longSilences.filter(Boolean).length;
  return {
    durationS: round(duration, 3),
    snrEstimateDb: round(snr, 2),
    energyAboveTelephone: totalPower > 0 ? round(above / totalPower, 5) : 0,
    bandwidthHz: round(bandwidth, 1),
    speechLevelDbfs: round(db(speechMean), 2),
    peakDbfs: round(db(peak * peak), 2),
    clippingRatio: round(clipped / signal.length, 6),
    silenceRatio: round(silent / longSilences.length, 4),
    longestSilenceS: round((longestRun(longSilences) * frame) / sampleRate, 2),
  };
}
