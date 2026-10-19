import { Matrix, createRng, type Clusterer, type Rng } from '@sim/core';

/**
 * k-means: alternate between assigning every point to its nearest centre and moving every centre
 * to the mean of its points (Lloyd's algorithm), which can only lower the within-cluster sum of
 * squares (the inertia). The result depends on where the centres start, so the usual remedies are
 * k-means++ seeding and several restarts. Conventions follow scikit-learn's `KMeans`.
 */
export type KMeansInit = 'kmeans++' | 'random' | Matrix;

export interface KMeansOptions {
  k: number;
  init?: KMeansInit;
  maxIterations?: number;
  /** Stop when the centres move less than this share of the data variance (scikit-learn: 1e-4). */
  tolerance?: number;
  /** Restarts from different initialisations; the lowest inertia wins. */
  restarts?: number;
  seed?: number;
}

const dist2 = (x: Matrix, i: number, c: Matrix, j: number): number => {
  let s = 0;
  for (let d = 0; d < x.cols; d++) {
    const v = x.get(i, d) - c.get(j, d);
    s += v * v;
  }
  return s;
};

/** k-means++ seeding: each new centre is drawn with probability proportional to its squared distance to the nearest one chosen so far. */
export function kmeansPlusPlus(x: Matrix, k: number, rng: Rng): Matrix {
  const n = x.rows;
  const centers = new Matrix(k, x.cols);
  const first = Math.floor(rng.next() * n);
  centers.data.set(x.data.subarray(first * x.cols, (first + 1) * x.cols), 0);
  const best = new Float64Array(n).fill(Infinity);
  for (let c = 1; c < k; c++) {
    let total = 0;
    for (let i = 0; i < n; i++) {
      best[i] = Math.min(best[i] as number, dist2(x, i, centers, c - 1));
      total += best[i] as number;
    }
    let target = rng.next() * total;
    let pick = n - 1;
    for (let i = 0; i < n; i++) {
      target -= best[i] as number;
      if (target <= 0) {
        pick = i;
        break;
      }
    }
    centers.data.set(x.data.subarray(pick * x.cols, (pick + 1) * x.cols), c * x.cols);
  }
  return centers;
}

/** `k` distinct data points chosen at random as the starting centres. */
export function randomCenters(x: Matrix, k: number, rng: Rng): Matrix {
  const chosen = new Set<number>();
  while (chosen.size < Math.min(k, x.rows)) chosen.add(Math.floor(rng.next() * x.rows));
  const centers = new Matrix(k, x.cols);
  [...chosen].forEach((r, c) =>
    centers.data.set(x.data.subarray(r * x.cols, (r + 1) * x.cols), c * x.cols),
  );
  return centers;
}

/** Assigns each row to its nearest centre; returns the labels and the inertia. */
export function assign(
  x: Matrix,
  centers: Matrix,
): { labels: Int32Array; inertia: number; dists: Float64Array } {
  const labels = new Int32Array(x.rows);
  const dists = new Float64Array(x.rows);
  let inertia = 0;
  for (let i = 0; i < x.rows; i++) {
    let best = 0;
    let bestD = Infinity;
    for (let c = 0; c < centers.rows; c++) {
      const d = dist2(x, i, centers, c);
      if (d < bestD) {
        bestD = d;
        best = c;
      }
    }
    labels[i] = best;
    dists[i] = bestD;
    inertia += bestD;
  }
  return { labels, inertia, dists };
}

export interface KMeansState {
  iteration: number;
  centers: Matrix;
  labels: Int32Array;
  inertia: number;
  /** How far the centres moved in the last update (squared, summed). */
  shift: number;
}

/** One run of Lloyd's algorithm, yielding the state after every iteration (the first is the initial assignment). */
export function* kmeansSteps(
  x: Matrix,
  initial: Matrix,
  options: { maxIterations?: number; tolerance?: number } = {},
): Generator<KMeansState, void, undefined> {
  const { maxIterations = 300 } = options;
  const k = initial.rows;
  let variance = 0;
  for (let d = 0; d < x.cols; d++) {
    let m = 0;
    for (let i = 0; i < x.rows; i++) m += x.get(i, d);
    m /= x.rows;
    let v = 0;
    for (let i = 0; i < x.rows; i++) v += (x.get(i, d) - m) ** 2;
    variance += v / x.rows;
  }
  const tol = (options.tolerance ?? 1e-4) * (variance / x.cols);
  let centers = new Matrix(k, x.cols, Float64Array.from(initial.data));
  let { labels, inertia } = assign(x, centers);
  yield { iteration: 0, centers, labels, inertia, shift: Infinity };
  for (let it = 1; it <= maxIterations; it++) {
    // update: move every centre to the mean of its points
    const next = new Matrix(k, x.cols);
    const counts = new Int32Array(k);
    for (let i = 0; i < x.rows; i++) {
      const c = labels[i] as number;
      counts[c] = (counts[c] as number) + 1;
      for (let d = 0; d < x.cols; d++)
        next.data[c * x.cols + d] = (next.data[c * x.cols + d] as number) + x.get(i, d);
    }
    for (let c = 0; c < k; c++)
      if (counts[c])
        for (let d = 0; d < x.cols; d++)
          next.data[c * x.cols + d] = (next.data[c * x.cols + d] as number) / (counts[c] as number);
    // an empty cluster is re-seeded at the point farthest from its centre
    const empty = Array.from({ length: k }, (_, c) => c).filter((c) => !counts[c]);
    if (empty.length) {
      const far = assign(x, centers).dists;
      const order = Array.from({ length: x.rows }, (_, i) => i).sort(
        (a, b) => (far[b] as number) - (far[a] as number),
      );
      empty.forEach((c, j) =>
        next.data.set(
          x.data.subarray((order[j] as number) * x.cols, ((order[j] as number) + 1) * x.cols),
          c * x.cols,
        ),
      );
    }
    let shift = 0;
    for (let i = 0; i < next.data.length; i++)
      shift += ((next.data[i] as number) - (centers.data[i] as number)) ** 2;
    centers = next;
    const previous = labels;
    ({ labels, inertia } = assign(x, centers));
    yield { iteration: it, centers, labels, inertia, shift };
    const strict = labels.every((v, i) => v === previous[i]);
    if (strict || shift <= tol) return;
  }
}

/** The starting centres of the first restart of a fit with this seed (what the animation starts from). */
export function initialCenters(x: Matrix, k: number, init: KMeansInit, seed: number): Matrix {
  if (init instanceof Matrix) return init;
  const rng = createRng(seed);
  return init === 'random' ? randomCenters(x, k, rng) : kmeansPlusPlus(x, k, rng);
}

export class KMeans implements Clusterer {
  centers: Matrix = new Matrix(0, 0);
  labels: Int32Array = new Int32Array(0);
  inertia = 0;
  iterations = 0;

  constructor(readonly options: KMeansOptions) {}

  fit(x: Matrix): this {
    const { k, restarts = 1, seed = 1 } = this.options;
    const rng = createRng(seed);
    const init = this.options.init ?? 'kmeans++';
    let best: KMeansState | null = null;
    const runs = init instanceof Matrix ? 1 : restarts;
    for (let r = 0; r < runs; r++) {
      const start =
        init instanceof Matrix
          ? init
          : init === 'random'
            ? randomCenters(x, k, rng)
            : kmeansPlusPlus(x, k, rng);
      let last: KMeansState | null = null;
      for (const s of kmeansSteps(x, start, this.options)) last = s;
      if (last && (!best || last.inertia < best.inertia)) best = last;
    }
    const final = best as KMeansState;
    this.centers = final.centers;
    this.labels = final.labels;
    this.inertia = final.inertia;
    // scikit-learn counts the first assignment pass as iteration 1
    this.iterations = final.iteration + 1;
    return this;
  }

  predict(x: Matrix): Int32Array {
    return assign(x, this.centers).labels;
  }

  /** Distances from every row to every centre. */
  transform(x: Matrix): Matrix {
    const out = new Matrix(x.rows, this.centers.rows);
    for (let i = 0; i < x.rows; i++)
      for (let c = 0; c < this.centers.rows; c++)
        out.set(i, c, Math.sqrt(dist2(x, i, this.centers, c)));
    return out;
  }
}

/** Inertia of the best of `restarts` k-means fits for each k in `ks` (the "elbow" curve). */
export function elbowCurve(
  x: Matrix,
  ks: readonly number[],
  options: Omit<KMeansOptions, 'k'> = {},
): number[] {
  return ks.map(
    (k) => new KMeans({ ...options, k, restarts: options.restarts ?? 5 }).fit(x).inertia,
  );
}
