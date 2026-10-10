/** Log-mel spectrogram, ported from `ynonnx_core.audio.features`. */

export interface MelConfig {
  sampleRate: number;
  nFft: number;
  hopLength: number;
  nMels: number;
}

export const WHISPER_MEL: MelConfig = {
  sampleRate: 16000,
  nFft: 400,
  hopLength: 160,
  nMels: 80,
};

/** large-v3 and large-v3-turbo take 128 mel bands, every earlier size takes 80. */
export const WHISPER_MEL_V3: MelConfig = { ...WHISPER_MEL, nMels: 128 };

/** OpenAI's two vocabulary sizes, and the band count each one implies. */
const VOCAB_BANDS: Readonly<Record<number, number>> = {
  51865: 80,
  51866: 128,
};

/** How many mel bands this model wants. */
export function melBandsFor(options: {
  encoderDims?: readonly (number | string)[] | undefined;
  logitsVocabSize?: number | null | undefined;
}): number {
  const dims = options.encoderDims;
  if (dims && dims.length >= 2 && typeof dims[1] === "number" && dims[1] > 0) {
    return dims[1];
  }
  return VOCAB_BANDS[options.logitsVocabSize ?? 0] ?? 80;
}

/** The mel config for a model, given what its manifest and graph declare. */
export function melConfigFor(options: {
  encoderDims?: readonly (number | string)[] | undefined;
  logitsVocabSize?: number | null | undefined;
}): MelConfig {
  return { ...WHISPER_MEL, nMels: melBandsFor(options) };
}

/** 30 seconds at 16 kHz with a 160-sample hop. */
export const WINDOW_FRAMES = 3000;
export const WINDOW_SAMPLES = 480_000;
export const WINDOW_SECONDS = 30;

/** In-place iterative radix-2 Cooley-Tukey, on a power-of-two size. */
class FFT {
  readonly size: number;
  private readonly cos: Float64Array;
  private readonly sin: Float64Array;
  private readonly reverse: Uint32Array;

  constructor(size: number) {
    if ((size & (size - 1)) !== 0) throw new Error(`FFT size ${size} is not a power of two`);
    this.size = size;
    this.cos = new Float64Array(size / 2);
    this.sin = new Float64Array(size / 2);
    for (let i = 0; i < size / 2; i += 1) {
      this.cos[i] = Math.cos((-2 * Math.PI * i) / size);
      this.sin[i] = Math.sin((-2 * Math.PI * i) / size);
    }
    this.reverse = new Uint32Array(size);
    const bits = Math.log2(size);
    for (let i = 0; i < size; i += 1) {
      let value = 0;
      for (let bit = 0; bit < bits; bit += 1) {
        value |= ((i >> bit) & 1) << (bits - 1 - bit);
      }
      this.reverse[i] = value;
    }
  }

  /** Transform `real` (length `size`) in place; `imag` must start at zero. */
  run(real: Float64Array, imag: Float64Array): void {
    const n = this.size;
    for (let i = 0; i < n; i += 1) {
      const j = this.reverse[i]!;
      if (j > i) {
        const tr = real[i]!;
        real[i] = real[j]!;
        real[j] = tr;
        const ti = imag[i]!;
        imag[i] = imag[j]!;
        imag[j] = ti;
      }
    }
    for (let span = 2; span <= n; span <<= 1) {
      const step = n / span;
      const half = span >> 1;
      for (let start = 0; start < n; start += span) {
        for (let k = 0; k < half; k += 1) {
          const twiddle = k * step;
          const wr = this.cos[twiddle]!;
          const wi = this.sin[twiddle]!;
          const a = start + k;
          const b = a + half;
          const xr = real[b]! * wr - imag[b]! * wi;
          const xi = real[b]! * wi + imag[b]! * wr;
          real[b] = real[a]! - xr;
          imag[b] = imag[a]! - xi;
          real[a] = real[a]! + xr;
          imag[a] = imag[a]! + xi;
        }
      }
    }
  }
}

/** The real DFT of an arbitrary length, via Bluestein's chirp-z transform. */
export class RealDFT {
  private readonly n: number;
  private readonly m: number;
  private readonly fft: FFT;
  private readonly chirpRe: Float64Array;
  private readonly chirpIm: Float64Array;
  private readonly kernelRe: Float64Array;
  private readonly kernelIm: Float64Array;

  constructor(n: number) {
    this.n = n;
    let m = 1;
    while (m < 2 * n - 1) m <<= 1;
    this.m = m;
    this.fft = new FFT(m);

    this.chirpRe = new Float64Array(n);
    this.chirpIm = new Float64Array(n);
    for (let j = 0; j < n; j += 1) {
      const angle = (-Math.PI * ((j * j) % (2 * n))) / n;
      this.chirpRe[j] = Math.cos(angle);
      this.chirpIm[j] = Math.sin(angle);
    }

    const kernelRe = new Float64Array(m);
    const kernelIm = new Float64Array(m);
    kernelRe[0] = this.chirpRe[0]!;
    kernelIm[0] = -this.chirpIm[0]!;
    for (let j = 1; j < n; j += 1) {
      const re = this.chirpRe[j]!;
      const im = -this.chirpIm[j]!;
      kernelRe[j] = re;
      kernelIm[j] = im;
      kernelRe[m - j] = re;
      kernelIm[m - j] = im;
    }
    this.fft.run(kernelRe, kernelIm);
    this.kernelRe = kernelRe;
    this.kernelIm = kernelIm;
    this.are = new Float64Array(m);
    this.aim = new Float64Array(m);
  }

  /** Scratch buffers, reused frame after frame (a 30 s window is 3 000 frames). */
  private readonly are: Float64Array;
  private readonly aim: Float64Array;

  /**
   * Bins `0 .. n/2` of the DFT of `input` (length `n`), written into `outRe`/`outIm`. Only the
   * non-negative frequencies, like `np.fft.rfft`.
   */
  run(input: Float64Array, outRe: Float64Array, outIm: Float64Array): void {
    this.#transform(input, null);
    const bins = (this.n >> 1) + 1;
    outRe.set(this.are.subarray(0, bins));
    outIm.set(this.aim.subarray(0, bins));
  }

  /**
   * Two real signals in one complex transform: `a` as the real part, `b` as the imaginary one,
   * pulled apart by symmetry — `A[k] = (Z[k] + conj(Z[n−k]))/2`, `B[k] = (Z[k] − conj(Z[n−k]))/2i`.
   * Half the work of two `run`s.
   */
  runPair(a: Float64Array, b: Float64Array, outA: [Float64Array, Float64Array], outB: [Float64Array, Float64Array]): void {
    this.#transform(a, b);
    const { n, are, aim } = this;
    const bins = (n >> 1) + 1;
    for (let k = 0; k < bins; k += 1) {
      const j = k === 0 ? 0 : n - k;
      const zr = are[k]!;
      const zi = aim[k]!;
      const cr = are[j]!;
      const ci = -aim[j]!;
      outA[0][k] = (zr + cr) / 2;
      outA[1][k] = (zi + ci) / 2;
      outB[0][k] = (zi - ci) / 2;
      outB[1][k] = -(zr - cr) / 2;
    }
  }

  /** The full DFT (bins `0 .. n−1`) of `re + i·im` into `this.are`/`this.aim`. */
  #transform(re: Float64Array, im: Float64Array | null): void {
    const { n, m, are, aim } = this;
    for (let j = 0; j < n; j += 1) {
      const x = re[j]!;
      const y = im ? im[j]! : 0;
      are[j] = x * this.chirpRe[j]! - y * this.chirpIm[j]!;
      aim[j] = x * this.chirpIm[j]! + y * this.chirpRe[j]!;
    }
    are.fill(0, n);
    aim.fill(0, n);
    this.fft.run(are, aim);

    for (let i = 0; i < m; i += 1) {
      const r = are[i]! * this.kernelRe[i]! - aim[i]! * this.kernelIm[i]!;
      const s = are[i]! * this.kernelIm[i]! + aim[i]! * this.kernelRe[i]!;
      are[i] = r;
      aim[i] = s;
    }

    for (let i = 0; i < m; i += 1) aim[i] = -aim[i]!;
    this.fft.run(are, aim);
    const scale = 1 / m;
    for (let k = 0; k < n; k += 1) {
      const r = are[k]! * scale;
      const s = -aim[k]! * scale;
      are[k] = r * this.chirpRe[k]! - s * this.chirpIm[k]!;
      aim[k] = r * this.chirpIm[k]! + s * this.chirpRe[k]!;
    }
  }
}

const MIN_LOG_HZ = 1000;
const MIN_LOG_MEL = 15;
const LOGSTEP = Math.log(6.4) / 27;

function hzToMel(hz: number): number {
  if (hz >= MIN_LOG_HZ) return MIN_LOG_MEL + Math.log(hz / MIN_LOG_HZ) / LOGSTEP;
  return hz / (200 / 3);
}

function melToHz(mel: number): number {
  if (mel >= MIN_LOG_MEL) return MIN_LOG_HZ * Math.exp(LOGSTEP * (mel - MIN_LOG_MEL));
  return mel * (200 / 3);
}

const filterCache = new Map<string, Float32Array>();

/** Slaney-scale mel filterbank, `[nMels, nFft/2 + 1]` flattened row-major. */
export function melFilters(config: MelConfig = WHISPER_MEL): Float32Array {
  const key = `${config.sampleRate}:${config.nFft}:${config.nMels}`;
  const cached = filterCache.get(key);
  if (cached) return cached;

  const nFreqs = config.nFft / 2 + 1;
  const freqs = new Float64Array(nFreqs);
  for (let i = 0; i < nFreqs; i += 1) {
    freqs[i] = (i * (config.sampleRate / 2)) / (nFreqs - 1);
  }

  const melMin = hzToMel(0);
  const melMax = hzToMel(config.sampleRate / 2);
  const points = config.nMels + 2;
  const hzPoints = new Float64Array(points);
  for (let i = 0; i < points; i += 1) {
    hzPoints[i] = melToHz(melMin + ((melMax - melMin) * i) / (points - 1));
  }

  const weights = new Float32Array(config.nMels * nFreqs);
  for (let m = 0; m < config.nMels; m += 1) {
    const left = hzPoints[m]!;
    const centre = hzPoints[m + 1]!;
    const right = hzPoints[m + 2]!;

    const enorm = 2 / (right - left);
    for (let f = 0; f < nFreqs; f += 1) {
      const hz = freqs[f]!;
      const lower = (hz - left) / (centre - left);
      const upper = (right - hz) / (right - centre);
      const value = Math.max(0, Math.min(lower, upper));
      weights[m * nFreqs + f] = value * enorm;
    }
  }
  filterCache.set(key, weights);
  return weights;
}

const dfts = new Map<number, RealDFT>();
function dftFor(n: number): RealDFT {
  let dft = dfts.get(n);
  if (!dft) dfts.set(n, (dft = new RealDFT(n)));
  return dft;
}

const spanCache = new WeakMap<Float32Array, [number, number][]>();
/** Per filter, the bins `[from, to)` outside which its weight is zero. */
function filterSpans(filters: Float32Array, nMels: number, nFreqs: number): [number, number][] {
  const known = spanCache.get(filters);
  if (known) return known;
  const spans: [number, number][] = [];
  for (let m = 0; m < nMels; m += 1) {
    let from = 0;
    let to = nFreqs;
    while (from < to && filters[m * nFreqs + from] === 0) from += 1;
    while (to > from && filters[m * nFreqs + to - 1] === 0) to -= 1;
    spans.push([from, to]);
  }
  spanCache.set(filters, spans);
  return spans;
}

function hann(nFft: number): Float64Array {
  const window = new Float64Array(nFft);
  for (let i = 0; i < nFft; i += 1) {
    window[i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / nFft);
  }
  return window;
}

function padOrTrim(signal: Float32Array, target: number): Float32Array {
  if (signal.length === target) return signal;
  if (signal.length > target) return signal.subarray(0, target);
  const out = new Float32Array(target);
  out.set(signal);
  return out;
}

export interface MelResult {
  /** Row-major `[nMels, frames]`. */
  data: Float32Array;
  nMels: number;
  frames: number;
}

/** Log-mel of one window. */
export function logMel(
  audio: Float32Array,
  config: MelConfig = WHISPER_MEL,
  padTo: number | null = WINDOW_FRAMES,
): MelResult {
  let signal = audio;
  if (padTo !== null) signal = padOrTrim(signal, padTo * config.hopLength);
  if (signal.length === 0) signal = new Float32Array(1);

  const pad = config.nFft >> 1;
  const canReflect = signal.length > pad;
  const padded = new Float32Array(signal.length + 2 * pad);
  padded.set(signal, pad);
  for (let i = 0; i < pad; i += 1) {
    padded[pad - 1 - i] = canReflect ? signal[i + 1]! : 0;
    padded[pad + signal.length + i] = canReflect ? signal[signal.length - 2 - i]! : 0;
  }

  const window = hann(config.nFft);
  const frameCount = 1 + Math.floor((padded.length - config.nFft) / config.hopLength);
  const nFreqs = config.nFft / 2 + 1;

  const dft = dftFor(config.nFft);
  const frameA = new Float64Array(config.nFft);
  const frameB = new Float64Array(config.nFft);
  const outA: [Float64Array, Float64Array] = [new Float64Array(nFreqs), new Float64Array(nFreqs)];
  const outB: [Float64Array, Float64Array] = [new Float64Array(nFreqs), new Float64Array(nFreqs)];

  const usable = Math.max(0, frameCount - 1);
  const power = new Float64Array(usable * nFreqs);

  const live: number[] = [];
  for (let index = 0; index < usable; index += 1) {
    const offset = index * config.hopLength;
    for (let i = 0; i < config.nFft; i += 1) {
      if (padded[offset + i] !== 0) {
        live.push(index);
        break;
      }
    }
  }
  const fill = (index: number, frame: Float64Array) => {
    const offset = index * config.hopLength;
    for (let i = 0; i < config.nFft; i += 1) frame[i] = padded[offset + i]! * window[i]!;
  };
  const store = (index: number, [re, im]: [Float64Array, Float64Array]) => {
    for (let bin = 0; bin < nFreqs; bin += 1) power[index * nFreqs + bin] = re[bin]! * re[bin]! + im[bin]! * im[bin]!;
  };
  for (let at = 0; at < live.length; at += 2) {
    const a = live[at]!;
    const b = live[at + 1];
    fill(a, frameA);
    if (b === undefined) {
      dft.run(frameA, outA[0], outA[1]);
      store(a, outA);
    } else {
      fill(b, frameB);
      dft.runPair(frameA, frameB, outA, outB);
      store(a, outA);
      store(b, outB);
    }
  }

  const filters = melFilters(config);
  const spans = filterSpans(filters, config.nMels, nFreqs);
  const mel = new Float32Array(config.nMels * usable);
  let maximum = -Infinity;
  for (let m = 0; m < config.nMels; m += 1) {
    const [from, to] = spans[m]!;
    for (let t = 0; t < usable; t += 1) {
      let sum = 0;
      for (let f = from; f < to; f += 1) {
        sum += filters[m * nFreqs + f]! * power[t * nFreqs + f]!;
      }
      const value = Math.log10(Math.max(sum, 1e-10));
      mel[m * usable + t] = value;
      if (value > maximum) maximum = value;
    }
  }

  const floor = maximum - 8;
  for (let i = 0; i < mel.length; i += 1) {
    mel[i] = (Math.max(mel[i]!, floor) + 4) / 4;
  }

  if (padTo === null) return { data: mel, nMels: config.nMels, frames: usable };

  if (usable === padTo) return { data: mel, nMels: config.nMels, frames: padTo };

  const out = new Float32Array(config.nMels * padTo);
  for (let m = 0; m < config.nMels; m += 1) {
    for (let t = 0; t < padTo; t += 1) {
      out[m * padTo + t] = mel[m * usable + Math.min(t, usable - 1)] ?? 0;
    }
  }
  return { data: out, nMels: config.nMels, frames: padTo };
}

/** `[batch, nMels, frames]` flattened, every row the same width. */
export function logMelBatch(
  audios: readonly Float32Array[],
  config: MelConfig = WHISPER_MEL,
  padTo: number | null = WINDOW_FRAMES,
): { data: Float32Array; dims: [number, number, number] } {
  if (audios.length === 0) {
    return { data: new Float32Array(0), dims: [0, config.nMels, 0] };
  }
  const mels = audios.map((audio) => logMel(audio, config, padTo));
  const frames = mels[0]!.frames;
  const out = new Float32Array(audios.length * config.nMels * frames);
  for (const [index, mel] of mels.entries()) {
    out.set(mel.data, index * config.nMels * frames);
  }
  return { data: out, dims: [audios.length, config.nMels, frames] };
}
