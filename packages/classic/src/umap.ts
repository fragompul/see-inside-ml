import { Matrix, createRng, nelderMead, symmetricEigen, type Rng } from '@sim/core';
import { pairwiseDistances } from './hierarchical';

/**
 * UMAP, written from the paper. Build a *fuzzy* neighbour graph (each point's links to its nearest
 * neighbours get a strength between 0 and 1, rescaled per point so every point has at least one
 * full-strength link), then lay the points out in 2-D so that linked points attract and random
 * pairs repel, by stochastic gradient descent on a cross-entropy between the two graphs.
 */

/** The curve `1 / (1 + a·x^(2b))` that approximates `exp(−(x − minDist)/spread)` beyond `minDist` (umap-learn's `find_ab_params`). */
export function findAB(spread: number, minDist: number): { a: number; b: number } {
  const xs = Array.from({ length: 300 }, (_, i) => (i / 299) * spread * 3);
  const target = xs.map((x) => (x < minDist ? 1 : Math.exp(-(x - minDist) / spread)));
  const loss = (p: readonly number[]) => {
    const a = Math.exp(p[0] as number);
    const b = Math.exp(p[1] as number);
    return xs.reduce((s, x, i) => s + (1 / (1 + a * x ** (2 * b)) - (target[i] as number)) ** 2, 0);
  };
  const fit = nelderMead(loss, [Math.log(1.5), Math.log(0.9)], {
    step: 0.3,
    maxIterations: 600,
    tolerance: 1e-14,
  });
  return { a: Math.exp(fit.x[0] as number), b: Math.exp(fit.x[1] as number) };
}

export interface FuzzyGraph {
  /** Symmetric membership strengths `n × n`, in [0, 1]. */
  weights: Float64Array;
  /** Distance to the nearest neighbour (each point's own scale). */
  rho: Float64Array;
  sigma: Float64Array;
  neighbors: number[][];
}

/** Fuzzy simplicial set of the data: per-point kernels calibrated to `log2(k)`, combined by fuzzy union. */
export function fuzzyGraph(x: Matrix, neighbors: number): FuzzyGraph {
  const n = x.rows;
  const k = Math.min(neighbors, n - 1);
  const d = pairwiseDistances(x);
  const nb: number[][] = [];
  const rho = new Float64Array(n);
  const sigma = new Float64Array(n);
  const directed = new Float64Array(n * n);
  const target = Math.log2(k);
  for (let i = 0; i < n; i++) {
    const order = Array.from({ length: n }, (_, j) => j)
      .filter((j) => j !== i)
      .sort((a, b) => (d[i * n + a] as number) - (d[i * n + b] as number) || a - b)
      .slice(0, k);
    nb.push(order);
    const dist = order.map((j) => d[i * n + j] as number);
    rho[i] = dist.find((v) => v > 0) ?? 0;
    // binary search for sigma so that Σ exp(−max(0, dist − rho)/sigma) = log2(k)
    let lo = 0;
    let hi = Infinity;
    let mid = 1;
    for (let step = 0; step < 64; step++) {
      let psum = 0;
      for (const v of dist) psum += Math.exp(-Math.max(0, v - (rho[i] as number)) / mid);
      if (Math.abs(psum - target) < 1e-5) break;
      if (psum > target) {
        hi = mid;
        mid = (lo + hi) / 2;
      } else {
        lo = mid;
        mid = hi === Infinity ? mid * 2 : (lo + hi) / 2;
      }
    }
    sigma[i] = Math.max(mid, 1e-3);
    order.forEach((j, r) => {
      const v = Math.exp(
        -Math.max(0, (dist[r] as number) - (rho[i] as number)) / (sigma[i] as number),
      );
      directed[i * n + j] = v;
    });
  }
  const weights = new Float64Array(n * n);
  for (let i = 0; i < n; i++)
    for (let j = 0; j < n; j++) {
      const a = directed[i * n + j] as number;
      const b = directed[j * n + i] as number;
      weights[i * n + j] = a + b - a * b; // fuzzy union
    }
  return { weights, rho, sigma, neighbors: nb };
}

export interface UmapOptions {
  neighbors?: number;
  minDist?: number;
  spread?: number;
  epochs?: number;
  /** Negative (repulsive) samples per attractive one. */
  negativeSamples?: number;
  seed?: number;
  /** Start from the spectral layout of the graph (default) or random positions. */
  init?: 'spectral' | 'random';
}

export interface UmapState {
  epoch: number;
  embedding: Matrix;
}

function initialLayout(
  graph: FuzzyGraph,
  n: number,
  mode: 'spectral' | 'random',
  rng: Rng,
): Matrix {
  const out = new Matrix(n, 2);
  if (mode === 'spectral') {
    try {
      const deg = new Float64Array(n);
      for (let i = 0; i < n; i++)
        for (let j = 0; j < n; j++)
          deg[i] = (deg[i] as number) + (graph.weights[i * n + j] as number);
      const m = new Matrix(n, n);
      for (let i = 0; i < n; i++)
        for (let j = 0; j < n; j++)
          m.set(
            i,
            j,
            (graph.weights[i * n + j] as number) /
              Math.sqrt((deg[i] as number) * (deg[j] as number) || 1),
          );
      const { vectors } = symmetricEigen(m);
      // vectors are sorted largest first: skip the trivial one, take the next two
      for (let i = 0; i < n; i++) {
        out.set(i, 0, vectors.get(i, 1));
        out.set(i, 1, vectors.get(i, 2));
      }
    } catch {
      mode = 'random';
    }
  }
  if (mode === 'random')
    for (let i = 0; i < out.data.length; i++) out.data[i] = rng.uniform(-10, 10);
  // scale into [0, 10] and add a little noise, as umap-learn does
  let max = 0;
  for (const v of out.data) max = Math.max(max, Math.abs(v));
  const expand = max > 0 ? 10 / max : 1;
  for (let i = 0; i < out.data.length; i++)
    out.data[i] = (out.data[i] as number) * expand + rng.normal() * 1e-4;
  return out;
}

const clip = (v: number) => Math.max(-4, Math.min(4, v));

/** The layout optimisation as a generator: state 0 is the starting layout, then one per `every` epochs. */
export function* umapSteps(
  x: Matrix,
  options: UmapOptions = {},
  every = 10,
): Generator<UmapState, void, undefined> {
  const n = x.rows;
  const {
    neighbors = 15,
    minDist = 0.1,
    spread = 1,
    epochs = 200,
    negativeSamples = 5,
    seed = 1,
    init = 'spectral',
  } = options;
  const rng = createRng(seed);
  const { a, b } = findAB(spread, minDist);
  const graph = fuzzyGraph(x, neighbors);
  const emb = initialLayout(graph, n, init, rng);
  // edges with their sampling schedule: strong edges are visited every epoch, weak ones rarely
  const edges: { i: number; j: number; w: number }[] = [];
  let maxW = 0;
  for (let i = 0; i < n; i++)
    for (let j = i + 1; j < n; j++) {
      const w = graph.weights[i * n + j] as number;
      if (w > 0) {
        edges.push({ i, j, w });
        maxW = Math.max(maxW, w);
      }
    }
  const live = edges.filter((e) => e.w >= maxW / epochs);
  const perSample = live.map((e) => maxW / e.w);
  const nextSample = Float64Array.from(perSample);
  yield { epoch: 0, embedding: new Matrix(n, 2, Float64Array.from(emb.data)) };
  for (let epoch = 1; epoch <= epochs; epoch++) {
    const alpha = 1 - (epoch - 1) / epochs;
    live.forEach((e, idx) => {
      if ((nextSample[idx] as number) > epoch) return;
      // attraction along the edge
      const ci = e.i * 2;
      const cj = e.j * 2;
      let d2 = 0;
      for (let c = 0; c < 2; c++)
        d2 += ((emb.data[ci + c] as number) - (emb.data[cj + c] as number)) ** 2;
      if (d2 > 0) {
        const coeff = (-2 * a * b * d2 ** (b - 1)) / (1 + a * d2 ** b);
        for (let c = 0; c < 2; c++) {
          const g =
            clip(coeff * ((emb.data[ci + c] as number) - (emb.data[cj + c] as number))) * alpha;
          emb.data[ci + c] = (emb.data[ci + c] as number) + g;
          emb.data[cj + c] = (emb.data[cj + c] as number) - g;
        }
      }
      nextSample[idx] = (nextSample[idx] as number) + (perSample[idx] as number);
      // repulsion from random points
      for (let s = 0; s < negativeSamples; s++) {
        const k = Math.floor(rng.next() * n);
        if (k === e.i) continue;
        const ck = k * 2;
        let dd = 0;
        for (let c = 0; c < 2; c++)
          dd += ((emb.data[ci + c] as number) - (emb.data[ck + c] as number)) ** 2;
        const coeffR = (2 * b) / ((0.001 + dd) * (1 + a * dd ** b));
        for (let c = 0; c < 2; c++) {
          const g =
            dd > 0
              ? clip(coeffR * ((emb.data[ci + c] as number) - (emb.data[ck + c] as number)))
              : 4;
          emb.data[ci + c] = (emb.data[ci + c] as number) + g * alpha;
        }
      }
    });
    if (epoch % every === 0 || epoch === epochs)
      yield { epoch, embedding: new Matrix(n, 2, Float64Array.from(emb.data)) };
  }
}

export function umap(x: Matrix, options: UmapOptions = {}): Matrix {
  let last: UmapState | null = null;
  for (const s of umapSteps(x, options, 1000)) last = s;
  return (last as UmapState).embedding;
}
