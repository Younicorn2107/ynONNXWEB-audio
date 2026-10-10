/** Grouping speaker embeddings into speakers. */

import { seededRandom } from "@youni/onnxweb-core/core";

/** Pairwise cosine similarity of the rows (normalized here), clipped to [-1, 1]. */
export function cosineSimilarityMatrix(embeddings: readonly Float32Array[]): Float64Array[] {
  const normalized = embeddings.map((row) => {
    let norm = 0;
    for (const x of row) norm += x * x;
    const scale = 1 / Math.max(Math.sqrt(norm), 1e-12);
    return Float64Array.from(row, (x) => x * scale);
  });
  return normalized.map((a) =>
    Float64Array.from(normalized, (b) => {
      let dot = 0;
      for (let k = 0; k < a.length; k += 1) dot += a[k]! * b[k]!;
      return Math.min(1, Math.max(-1, dot));
    }),
  );
}

export interface AgglomerativeOptions {
  threshold?: number;
  numSpeakers?: number | null;
  minSpeakers?: number | null;
  maxSpeakers?: number | null;
}

/**
 * Average-linkage clustering on cosine similarity: one label per row. Average rather than single
 * linkage, which chains: one ambiguous segment between two speakers would merge them for the rest
 * of the recording.
 */
export function agglomerative(embeddings: readonly Float32Array[], options: AgglomerativeOptions = {}): number[] {
  const count = embeddings.length;
  if (count <= 1) return new Array<number>(count).fill(0);
  const threshold = options.threshold ?? 0.7;
  const similarity = cosineSimilarityMatrix(embeddings);
  const clusters: number[][] = Array.from({ length: count }, (_, i) => [i]);
  const floor = Math.max(1, options.minSpeakers ?? 1);
  const ceiling = options.maxSpeakers || count;
  const target = options.numSpeakers ?? null;

  while (clusters.length > floor) {
    let best: [number, number, number] = [-1, -1, Number.NEGATIVE_INFINITY];
    for (let a = 0; a < clusters.length; a += 1) {
      for (let b = a + 1; b < clusters.length; b += 1) {
        let sum = 0;
        for (const i of clusters[a]!) for (const j of clusters[b]!) sum += similarity[i]![j]!;
        const linkage = sum / (clusters[a]!.length * clusters[b]!.length);
        if (linkage > best[2]) best = [a, b, linkage];
      }
    }
    const [a, b, linkage] = best;
    if (a < 0) break;
    const mustMerge = (target !== null && clusters.length > target) || clusters.length > ceiling;
    if (!mustMerge && linkage < threshold) break;
    clusters[a] = [...clusters[a]!, ...clusters[b]!];
    clusters.splice(b, 1);
    if (target !== null && clusters.length === target) break;
  }
  const labels = new Array<number>(count).fill(0);
  clusters.forEach((members, label) => members.forEach((i) => (labels[i] = label)));
  return labels;
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
