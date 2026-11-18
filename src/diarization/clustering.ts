/** Grouping speaker embeddings into speakers. */

import { seededRandom } from "@youni/onnxweb-core/core";

/**
 * Where agglomerative clustering stops: 0.3 average cosine similarity (a cosine distance of 0.7).
 * Measured by the Python library on VoxConverse dev (two sets of 12 recordings, DER with a 0.25 s
 * collar): 0.5 gave 40.8 / 43.8 % (16 to 35 speakers too many: short overlapping chunks give noisy
 * prints), 0.3 alone 8.8 / 6.1 %, 0.3 with clusters under 5 s of speech absorbed 7.2 / 4.2 %, 0.2
 * 6.7 / 7.7 %.
 */
export const DEFAULT_CLUSTER_THRESHOLD = 0.3;

/** Rows L2-normalized in float64, as the Python side normalizes before SciPy's cosine metric. */
function unitRows(embeddings: readonly Float32Array[]): Float64Array[] {
  return embeddings.map((row) => {
    let norm = 0;
    for (const x of row) norm += x * x;
    const scale = 1 / Math.max(Math.sqrt(norm), 1e-12);
    return Float64Array.from(row, (x) => x * scale);
  });
}

/** Pairwise cosine similarity of the rows (normalized here), clipped to [-1, 1]. */
export function cosineSimilarityMatrix(embeddings: readonly Float32Array[]): Float64Array[] {
  const normalized = unitRows(embeddings);
  return normalized.map((a) =>
    Float64Array.from(normalized, (b) => {
      let dot = 0;
      for (let k = 0; k < a.length; k += 1) dot += a[k]! * b[k]!;
      return Math.min(1, Math.max(-1, dot));
    }),
  );
}

export interface AgglomerativeOptions {
  /** Merge while the average cosine similarity stays at or above this (0.3). */
  threshold?: number;
  numSpeakers?: number | null;
  minSpeakers?: number | null;
  maxSpeakers?: number | null;
}

/**
 * One merge of the dendrogram: two clusters, each named by one of its rows, at a cosine distance.
 */
export interface Merge {
  a: number;
  b: number;
  distance: number;
}

/**
 * The average-linkage dendrogram of the rows on cosine distance, merges in increasing distance:
 * SciPy's `linkage(method="average", metric="cosine")`.
 */
export function averageLinkage(embeddings: readonly Float32Array[]): Merge[] {
  const n = embeddings.length;
  if (n <= 1) return [];
  const rows = unitRows(embeddings);
  const dims = rows[0]!.length;
  const d = new Float64Array((n * (n - 1)) / 2);

  const offset = (i: number) => i * n - (i * (i + 1)) / 2;
  const at = (i: number, j: number) => (i < j ? offset(i) + j - i - 1 : offset(j) + i - j - 1);
  for (let i = 0; i < n; i += 1) {
    const a = rows[i]!;
    const base = offset(i) - i - 1;
    for (let j = i + 1; j < n; j += 1) {
      const b = rows[j]!;
      let dot = 0;
      for (let k = 0; k < dims; k += 1) dot += a[k]! * b[k]!;
      d[base + j] = 1 - dot;
    }
  }
  const size = new Float64Array(n).fill(1);
  const active = new Uint8Array(n).fill(1);
  const chain: number[] = [];
  const merges: Merge[] = [];
  for (let step = 0; step < n - 1; step += 1) {
    if (chain.length === 0) chain.push(active.indexOf(1));
    let x: number;
    let y: number;
    let best: number;
    for (;;) {
      x = chain[chain.length - 1]!;

      y = chain.length >= 2 ? chain[chain.length - 2]! : -1;
      best = y >= 0 ? d[at(x, y)]! : Number.POSITIVE_INFINITY;
      for (let i = 0; i < n; i += 1) {
        if (!active[i] || i === x) continue;
        const value = d[at(x, i)]!;
        if (value < best) {
          best = value;
          y = i;
        }
      }
      if (chain.length >= 2 && y === chain[chain.length - 2]) break;
      chain.push(y);
    }
    chain.pop();
    chain.pop();
    if (x > y) [x, y] = [y, x];
    merges.push({ a: x, b: y, distance: best });

    const nx = size[x]!;
    const ny = size[y]!;
    active[x] = 0;
    for (let i = 0; i < n; i += 1) {
      if (!active[i] || i === y) continue;
      d[at(i, y)] = (nx * d[at(i, x)]! + ny * d[at(i, y)]!) / (nx + ny);
    }
    size[y] = nx + ny;
  }

  return merges
    .map((merge, order) => ({ merge, order }))
    .sort((p, q) => p.merge.distance - q.merge.distance || p.order - q.order)
    .map(({ merge }) => merge);
}

/** Flat labels, by first appearance, after the first `count` merges of a sorted dendrogram. */
function cutTree(rows: number, merges: readonly Merge[], count: number): number[] {
  const parent = Int32Array.from({ length: rows }, (_, i) => i);
  const root = (i: number): number => {
    while (parent[i] !== i) {
      parent[i] = parent[parent[i]!]!;
      i = parent[i]!;
    }
    return i;
  };
  for (let k = 0; k < count; k += 1) parent[root(merges[k]!.a)] = root(merges[k]!.b);
  const names = new Map<number, number>();
  return Array.from({ length: rows }, (_, i) => {
    const r = root(i);
    if (!names.has(r)) names.set(r, names.size);
    return names.get(r)!;
  });
}

/**
 * Average-linkage clustering on cosine similarity: one label per row, the tree cut as SciPy's
 * `fcluster` cuts it. Average rather than single linkage, which chains: one ambiguous chunk between
 * two speakers would merge them for the rest of the recording. Clusters merge while their average
 * similarity stays at or above `threshold` (a cosine distance of at most 1 − threshold);
 * `numSpeakers` imposes the count, `minSpeakers` and `maxSpeakers` bound it.
 */
export function agglomerative(embeddings: readonly Float32Array[], options: AgglomerativeOptions = {}): number[] {
  const count = embeddings.length;
  if (count <= 1) return new Array<number>(count).fill(0);
  const merges = averageLinkage(embeddings);
  const into = (clusters: number) => cutTree(count, merges, count - Math.max(1, Math.min(Math.trunc(clusters), count)));
  if (options.numSpeakers !== null && options.numSpeakers !== undefined) return into(options.numSpeakers);
  const limit = 1 - (options.threshold ?? DEFAULT_CLUSTER_THRESHOLD);
  let below = 0;
  while (below < merges.length && merges[below]!.distance <= limit) below += 1;
  const found = count - below;
  const ceiling = options.maxSpeakers || count;
  const floor = Math.max(1, options.minSpeakers || 1);
  if (found > ceiling) return into(ceiling);
  if (found < floor) return into(floor);
  return cutTree(count, merges, below);
}

/** Seconds covered by the union of `spans`: overlapping chunks counted once. */
export function covered(spans: readonly (readonly [number, number])[]): number {
  let total = 0;
  let reach = Number.NEGATIVE_INFINITY;
  for (const [start, end] of [...spans].sort((a, b) => a[0] - b[0] || a[1] - b[1])) {
    if (end <= reach) continue;
    total += end - Math.max(start, reach);
    reach = end;
  }
  return total;
}

/** A speaker's print: the normalized sum of the normalized vectors. */
export function meanPrint(vectors: readonly ArrayLike<number>[]): Float64Array {
  const sum = new Float64Array(vectors[0]?.length ?? 0);
  for (const v of vectors) {
    let norm = 0;
    for (let k = 0; k < v.length; k += 1) norm += v[k]! * v[k]!;
    const scale = 1 / Math.max(Math.sqrt(norm), 1e-12);
    for (let k = 0; k < v.length; k += 1) sum[k] = sum[k]! + v[k]! * scale;
  }
  let norm = 0;
  for (const x of sum) norm += x * x;
  const scale = 1 / Math.max(Math.sqrt(norm), 1e-12);
  return sum.map((x) => x * scale);
}

/**
 * Clusters holding less than `minSeconds` of speech merged into the closest of the others (by
 * centroid). Short noisy chunks — a laugh, a cough, music under speech — cluster on their own and
 * become one-chunk speakers; a real speaker holds the floor longer. The speech a cluster holds is
 * the union of its chunks: they overlap (1.5 s every 0.75 s), and their sum counted each second
 * about twice. Nothing changes when no cluster, or every cluster, reaches the minimum.
 */
export function absorbSmallClusters(
  embeddings: readonly Float32Array[],
  labels: readonly number[],
  spans: readonly (readonly [number, number])[],
  minSeconds: number,
): number[] {
  const out = [...labels];
  if (minSeconds <= 0 || out.length === 0) return out;
  const byLabel = new Map<number, (readonly [number, number])[]>();
  out.forEach((label, i) => {
    if (!byLabel.has(label)) byLabel.set(label, []);
    byLabel.get(label)!.push(spans[i]!);
  });
  const big = [...byLabel].filter(([, chunks]) => covered(chunks) >= minSeconds).map(([label]) => label);
  if (big.length === 0 || big.length === byLabel.size) return out;
  const membersOf = (label: number) => labels.flatMap((l, i) => (l === label ? [embeddings[i]!] : []));
  const centroids = big.map((label) => meanPrint(membersOf(label)));
  for (const label of byLabel.keys()) {
    if (big.includes(label)) continue;
    const print = meanPrint(membersOf(label));
    let best = 0;
    let score = Number.NEGATIVE_INFINITY;
    centroids.forEach((centroid, k) => {
      let dot = 0;
      for (let j = 0; j < print.length; j += 1) dot += centroid[j]! * print[j]!;
      if (dot > score) [best, score] = [k, dot];
    });
    out.forEach((l, i) => {
      if (l === label) out[i] = big[best]!;
    });
  }
  return out;
}

/**
 * Eigenvalues (ascending) and eigenvectors (columns) of a symmetric matrix, by cyclic Jacobi
 * rotations.
 */
export function symmetricEigen(matrix: readonly Float64Array[]): { values: number[]; vectors: Float64Array[] } {
  const n = matrix.length;
  const a = matrix.map((row) => Float64Array.from(row));
  const v = Array.from({ length: n }, (_, i) => Float64Array.from({ length: n }, (_, j) => (i === j ? 1 : 0)));
  for (let sweep = 0; sweep < 100; sweep += 1) {
    let off = 0;
    for (let p = 0; p < n; p += 1) for (let q = p + 1; q < n; q += 1) off += a[p]![q]! ** 2;
    if (off < 1e-22) break;
    for (let p = 0; p < n; p += 1) {
      for (let q = p + 1; q < n; q += 1) {
        const apq = a[p]![q]!;
        if (Math.abs(apq) < 1e-300) continue;
        const theta = (a[q]![q]! - a[p]![p]!) / (2 * apq);
        const t = Math.sign(theta || 1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
        const c = 1 / Math.sqrt(t * t + 1);
        const s = t * c;
        for (let k = 0; k < n; k += 1) {
          const akp = a[k]![p]!;
          const akq = a[k]![q]!;
          a[k]![p] = c * akp - s * akq;
          a[k]![q] = s * akp + c * akq;
        }
        for (let k = 0; k < n; k += 1) {
          const apk = a[p]![k]!;
          const aqk = a[q]![k]!;
          a[p]![k] = c * apk - s * aqk;
          a[q]![k] = s * apk + c * aqk;
        }
        for (let k = 0; k < n; k += 1) {
          const vkp = v[k]![p]!;
          const vkq = v[k]![q]!;
          v[k]![p] = c * vkp - s * vkq;
          v[k]![q] = s * vkp + c * vkq;
        }
      }
    }
  }
  const order = Array.from({ length: n }, (_, i) => i).sort((x, y) => a[x]![x]! - a[y]![y]!);
  return {
    values: order.map((i) => a[i]![i]!),

    vectors: v.map((row) => Float64Array.from(order, (i) => row[i]!)),
  };
}

/** k-means++ initialization, then Lloyd's algorithm. */
export function kmeans(points: readonly Float64Array[], clusters: number, iterations = 50, seed = 0): number[] {
  const count = points.length;
  if (clusters >= count) return Array.from({ length: count }, (_, i) => i);
  const random = seededRandom(seed);
  const distance = (a: Float64Array, b: Float64Array) => a.reduce((s, x, k) => s + (x - b[k]!) ** 2, 0);
  const centres: Float64Array[] = [Float64Array.from(points[Math.floor(random() * count)]!)];
  for (let c = 1; c < clusters; c += 1) {
    const d = points.map((p) => Math.min(...centres.map((centre) => distance(p, centre))));
    const total = d.reduce((s, x) => s + x, 0);
    if (total <= 0) {
      centres.push(Float64Array.from(points[Math.floor(random() * count)]!));
      continue;
    }
    let pick = random() * total;
    let index = 0;
    while (index < count - 1 && pick >= d[index]!) {
      pick -= d[index]!;
      index += 1;
    }
    centres.push(Float64Array.from(points[index]!));
  }
  let labels = new Array<number>(count).fill(0);
  for (let it = 0; it < iterations; it += 1) {
    const next = points.map((p) => {
      let best = 0;
      for (let k = 1; k < clusters; k += 1) if (distance(p, centres[k]!) < distance(p, centres[best]!)) best = k;
      return best;
    });
    if (next.every((l, i) => l === labels[i])) break;
    labels = next;
    for (let k = 0; k < clusters; k += 1) {
      const members = points.filter((_, i) => labels[i] === k);
      if (members.length === 0) continue;
      centres[k] = Float64Array.from(members[0]!, (_, j) => members.reduce((s, m) => s + m[j]!, 0) / members.length);
    }
  }
  return labels;
}

/** Normalized spectral clustering, with an eigengap speaker count. */
export function spectral(embeddings: readonly Float32Array[], options: { numSpeakers?: number | null; maxSpeakers?: number; threshold?: number } = {}): number[] {
  const count = embeddings.length;
  if (count <= 1) return new Array<number>(count).fill(0);
  const threshold = options.threshold ?? 0.5;
  const affinity = cosineSimilarityMatrix(embeddings).map((row, i) =>
    Float64Array.from(row, (x, j) => (i === j ? 0 : Math.max(x, 0) < threshold ? 0 : Math.max(x, 0))),
  );
  const inverseSqrt = affinity.map((row) => 1 / Math.sqrt(Math.max(row.reduce((s, x) => s + x, 0), 1e-12)));
  const laplacian = affinity.map((row, i) => Float64Array.from(row, (x, j) => (i === j ? 1 : 0) - inverseSqrt[i]! * x * inverseSqrt[j]!));
  const { values, vectors } = symmetricEigen(laplacian);
  let speakers = options.numSpeakers ?? null;
  if (speakers === null) {
    const limit = Math.min(options.maxSpeakers ?? 8, count - 1);
    let best = 0;
    let bestGap = Number.NEGATIVE_INFINITY;
    for (let k = 0; k < limit; k += 1) {
      const gap = values[k + 1]! - values[k]!;
      if (gap > bestGap) {
        bestGap = gap;
        best = k;
      }
    }
    speakers = limit > 0 ? best + 1 : 1;
  }
  speakers = Math.max(1, Math.min(speakers, count));
  const features = vectors.map((row) => {
    const head = row.slice(0, speakers!);
    const norm = Math.max(Math.sqrt(head.reduce((s, x) => s + x * x, 0)), 1e-12);
    return head.map((x) => x / norm);
  });
  return kmeans(features, speakers);
}

/**
 * Renumber so speaker 0 is whoever speaks first: a transcript opening on SPEAKER_03 reads like a
 * bug.
 */
export function relabelByFirstAppearance(labels: readonly number[]): number[] {
  const mapping = new Map<number, number>();
  return labels.map((label) => {
    if (!mapping.has(label)) mapping.set(label, mapping.size);
    return mapping.get(label)!;
  });
}
