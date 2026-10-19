import { Matrix, createRng, lossValue, symmetricEigen, type Rng } from '@sim/core';
import type { NeuralNetwork } from './network';

/**
 * Looking at a network's loss as a landscape. The loss is a function of every weight, far too many
 * dimensions to see; a **slice** evaluates it on a plane through weight space and shows that plane
 * as a map. The plane is spanned by two directions: random ones (filter-normalised, as in Li et al.,
 * *Visualizing the Loss Landscape of Neural Nets*), or the two directions along which the training
 * path moved the most (PCA of the trajectory), which makes the descent visible.
 */

/** All trainable parameters as one flat vector (weights and biases, layer by layer). */
export function flattenParams(net: NeuralNetwork): Float64Array {
  const ps = net.parameters;
  const out = new Float64Array(ps.reduce((n, p) => n + p.data.length, 0));
  let at = 0;
  for (const p of ps) {
    out.set(p.data, at);
    at += p.data.length;
  }
  return out;
}

/** Writes a flat vector back into the network (the inverse of {@link flattenParams}). */
export function assignParams(net: NeuralNetwork, flat: ArrayLike<number>): void {
  let at = 0;
  for (const p of net.parameters) {
    for (let i = 0; i < p.data.length; i++) p.data[i] = flat[at + i] as number;
    at += p.data.length;
  }
}

/** Boundaries `[start, end)` of every weight matrix and bias vector inside the flat vector. */
function blocks(net: NeuralNetwork): { start: number; end: number; isBias: boolean }[] {
  const out: { start: number; end: number; isBias: boolean }[] = [];
  let at = 0;
  for (const layer of net.layers)
    for (const p of layer.parameters) {
      out.push({ start: at, end: at + p.data.length, isBias: p === layer.biasParam });
      at += p.data.length;
    }
  return out;
}

const dot = (a: ArrayLike<number>, b: ArrayLike<number>) => {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += (a[i] as number) * (b[i] as number);
  return s;
};
const norm = (a: ArrayLike<number>) => Math.sqrt(dot(a, a));

/** Two orthonormal vectors from two arbitrary ones (Gram–Schmidt); the second is null when degenerate. */
function orthonormal(a: Float64Array, b: Float64Array): [Float64Array, Float64Array | null] {
  const na = norm(a);
  const u1 = na > 0 ? a.map((v) => v / na) : a;
  const proj = dot(b, u1);
  const rest = b.map((v, i) => v - proj * (u1[i] as number));
  const nr = norm(rest);
  return [u1, nr > 1e-12 ? rest.map((v) => v / nr) : null];
}

/** Random direction whose weight blocks have the same norm as the weights (biases stay fixed). */
function filterNormalised(net: NeuralNetwork, centre: Float64Array, rng: Rng): Float64Array {
  const d = new Float64Array(centre.length);
  for (const b of blocks(net)) {
    if (b.isBias) continue;
    let sum = 0;
    for (let i = b.start; i < b.end; i++) {
      d[i] = rng.normal();
      sum += (d[i] as number) ** 2;
    }
    let wn = 0;
    for (let i = b.start; i < b.end; i++) wn += (centre[i] as number) ** 2;
    const scale = sum > 0 ? Math.sqrt(wn / sum) : 0;
    for (let i = b.start; i < b.end; i++) d[i] = (d[i] as number) * scale;
  }
  return d;
}

/** The two leading principal directions of `trajectory − centre`, via the small Gram matrix. */
function pathDirections(
  trajectory: readonly Float64Array[],
  centre: Float64Array,
): [Float64Array, Float64Array] | null {
  const t = trajectory.length;
  if (t < 3) return null;
  const rows = trajectory.map((w) => w.map((v, i) => v - (centre[i] as number)));
  const gram = new Matrix(t, t);
  for (let i = 0; i < t; i++)
    for (let j = i; j < t; j++) {
      const v = dot(rows[i] as Float64Array, rows[j] as Float64Array);
      gram.set(i, j, v);
      gram.set(j, i, v);
    }
  const { values, vectors } = symmetricEigen(gram);
  // eigenvalues come back sorted descending (largest first); build directions as Xᵀ·u / √λ
  const dirs: Float64Array[] = [];
  for (let k = 0; k < 2 && k < values.length; k++) {
    if ((values[k] as number) <= 1e-12) break;
    const d = new Float64Array(centre.length);
    for (let i = 0; i < t; i++) {
      const w = vectors.get(i, k);
      const row = rows[i] as Float64Array;
      for (let j = 0; j < d.length; j++) d[j] = (d[j] as number) + w * (row[j] as number);
    }
    const n = norm(d);
    dirs.push(d.map((v) => v / n));
  }
  return dirs.length === 2 ? [dirs[0] as Float64Array, dirs[1] as Float64Array] : null;
}

export type LandscapeKind = 'random' | 'path';

export interface LandscapeOptions {
  /** Grid points per side (odd keeps the current weights at the centre cell). */
  steps?: number;
  kind?: LandscapeKind;
  seed?: number;
  /** Half-width of the plane in weight-space units; default adapts to the path or the weight norm. */
  radius?: number;
}

export interface Landscape {
  kind: LandscapeKind;
  steps: number;
  /** Coordinates along both axes (the same for each), from −radius to +radius. */
  axis: Float64Array;
  radius: number;
  /** `values[row * steps + col]`: loss at `centre + axis[col]·u1 + axis[row]·u2`. */
  values: Float32Array;
  /** Loss at the current weights (the centre of the plane). */
  centreLoss: number;
  /** Where each stored snapshot of the training path falls on the plane, oldest first. */
  path: { a: Float32Array; b: Float32Array } | null;
}

/**
 * Evaluates the training loss over a plane through the current weights. `trajectory` holds earlier
 * weight vectors (oldest first); the network's weights are restored before returning.
 */
export function weightSpaceLandscape(
  net: NeuralNetwork,
  X: Matrix,
  y: Matrix,
  trajectory: readonly Float64Array[],
  options: LandscapeOptions = {},
): Landscape {
  const steps = Math.max(5, options.steps ?? 25) | 1;
  const centre = flattenParams(net);
  const rng = createRng(options.seed ?? 1);

  let kind: LandscapeKind = options.kind ?? 'path';
  let u1: Float64Array;
  let u2: Float64Array;
  const fromPath = kind === 'path' ? pathDirections(trajectory, centre) : null;
  if (fromPath) [u1, u2] = fromPath;
  else {
    kind = 'random';
    const [a, b] = orthonormal(
      filterNormalised(net, centre, rng),
      filterNormalised(net, centre, rng),
    );
    u1 = a;
    u2 = b ?? a.map((_, i) => (i === 0 ? 1 : 0));
  }

  // where the path sits on this plane (the current weights are the origin)
  const project = (w: Float64Array) => {
    const diff = w.map((v, i) => v - (centre[i] as number));
    return [dot(diff, u1), dot(diff, u2)] as const;
  };
  const pathPoints = trajectory.map(project);
  const extent = Math.max(0, ...pathPoints.flat().map((v) => Math.abs(v)));
  const radius = options.radius ?? Math.max(extent * 1.25, 0.15 * norm(centre), 0.5);
  const axis = Float64Array.from(
    { length: steps },
    (_, i) => -radius + (2 * radius * i) / (steps - 1),
  );

  const values = new Float32Array(steps * steps);
  const probe = new Float64Array(centre.length);
  for (let r = 0; r < steps; r++)
    for (let c = 0; c < steps; c++) {
      const a = axis[c] as number;
      const b = axis[r] as number;
      for (let i = 0; i < probe.length; i++)
        probe[i] = (centre[i] as number) + a * (u1[i] as number) + b * (u2[i] as number);
      assignParams(net, probe);
      values[r * steps + c] = lossValue(net.loss, net.forward(X, false), y);
    }
  assignParams(net, centre);
  const centreLoss = lossValue(net.loss, net.forward(X, false), y);
  return {
    kind,
    steps,
    axis,
    radius,
    values,
    centreLoss,
    path:
      pathPoints.length > 0
        ? {
            a: Float32Array.from(pathPoints, (p) => p[0]),
            b: Float32Array.from(pathPoints, (p) => p[1]),
          }
        : null,
  };
}

export interface Histogram {
  min: number;
  max: number;
  counts: number[];
}

export interface LayerHistograms {
  weights: Histogram;
  gradients: Histogram;
  /** Fraction of the weights that are exactly zero (L1 regularisation, dead units). */
  zeroShare: number;
  /** Root mean square of the weights and of the gradients. */
  weightRms: number;
  gradientRms: number;
}

function histogram(values: ArrayLike<number>, bins: number, symmetric = true): Histogram {
  let lo = Infinity;
  let hi = -Infinity;
  for (let i = 0; i < values.length; i++) {
    lo = Math.min(lo, values[i] as number);
    hi = Math.max(hi, values[i] as number);
  }
  if (!Number.isFinite(lo) || lo === hi) {
    const c = Number.isFinite(lo) ? lo : 0;
    lo = c - 1e-9 - Math.abs(c) * 0.01;
    hi = c + 1e-9 + Math.abs(c) * 0.01;
  }
  if (symmetric) {
    const m = Math.max(Math.abs(lo), Math.abs(hi));
    lo = -m;
    hi = m;
  }
  const counts = new Array<number>(bins).fill(0);
  const width = (hi - lo) / bins;
  for (let i = 0; i < values.length; i++)
    counts[Math.min(bins - 1, Math.floor(((values[i] as number) - lo) / width))]! += 1;
  return { min: lo, max: hi, counts };
}

const rms = (v: ArrayLike<number>) => (v.length ? Math.sqrt(dot(v, v) / v.length) : 0);

/** Distribution of the weights and of the latest gradients, one pair of histograms per layer. */
export function layerHistograms(net: NeuralNetwork, bins = 24): LayerHistograms[] {
  return net.layers.map((layer) => {
    const w = layer.weights.data;
    const g = layer.weightParam.grad;
    let zeros = 0;
    for (const v of w) if (v === 0) zeros++;
    return {
      weights: histogram(w, bins),
      gradients: histogram(g, bins),
      zeroShare: w.length ? zeros / w.length : 0,
      weightRms: rms(w),
      gradientRms: rms(g),
    };
  });
}
