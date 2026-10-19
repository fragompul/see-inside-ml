import { Matrix, createRng, solve } from '@sim/core';

/**
 * Explaining one prediction. Shapley values split the difference between a prediction and the average
 * prediction fairly among the features (an idea from cooperative game theory); KernelSHAP estimates them
 * by weighted regression on sampled coalitions; LIME fits a simple local model around the instance.
 * All three only need a function `f` that maps rows to numbers.
 */
export type Model = (x: Matrix) => ArrayLike<number>;

const rowOf = (x: Matrix, i: number): number[] =>
  Array.from(x.data.subarray(i * x.cols, (i + 1) * x.cols));

/** The prediction for the instance when only the features in `coalition` are known; the rest are drawn from the background rows. */
export function coalitionValue(
  f: Model,
  instance: readonly number[],
  background: Matrix,
  coalition: ReadonlySet<number>,
): number {
  const probe = new Matrix(background.rows, background.cols);
  for (let b = 0; b < background.rows; b++)
    for (let j = 0; j < background.cols; j++)
      probe.set(b, j, coalition.has(j) ? (instance[j] as number) : background.get(b, j));
  const pred = f(probe);
  let s = 0;
  for (let b = 0; b < background.rows; b++) s += pred[b] as number;
  return s / background.rows;
}

export interface Attribution {
  /** Contribution of each feature. */
  values: number[];
  /** The average prediction over the background (what you would say knowing nothing about the instance). */
  base: number;
  /** The prediction for the instance. */
  prediction: number;
}

/** Exact Shapley values: average each feature's marginal contribution over all subsets (2^p evaluations of the model). */
export function shapleyExact(
  f: Model,
  instance: readonly number[],
  background: Matrix,
): Attribution {
  const p = instance.length;
  if (p > 12) throw new RangeError('Exact Shapley values need 12 features or fewer');
  const value = new Float64Array(1 << p);
  for (let mask = 0; mask < 1 << p; mask++) {
    const set = new Set<number>();
    for (let j = 0; j < p; j++) if (mask & (1 << j)) set.add(j);
    value[mask] = coalitionValue(f, instance, background, set);
  }
  const fact = [1];
  for (let i = 1; i <= p; i++) fact.push((fact[i - 1] as number) * i);
  const values = new Array<number>(p).fill(0);
  for (let j = 0; j < p; j++) {
    for (let mask = 0; mask < 1 << p; mask++) {
      if (mask & (1 << j)) continue;
      let size = 0;
      for (let k = 0; k < p; k++) if (mask & (1 << k)) size++;
      const weight =
        ((fact[size] as number) * (fact[p - size - 1] as number)) / (fact[p] as number);
      values[j] =
        (values[j] as number) +
        weight * ((value[mask | (1 << j)] as number) - (value[mask] as number));
    }
  }
  return { values, base: value[0] as number, prediction: value[(1 << p) - 1] as number };
}

/**
 * KernelSHAP: sample coalitions, evaluate the model on each, and fit a linear model whose coefficients are
 * the feature contributions, weighting each coalition by the Shapley kernel. The two constraints (the
 * contributions add up to prediction - base) are enforced exactly.
 */
export function kernelShap(
  f: Model,
  instance: readonly number[],
  background: Matrix,
  samples: number,
  seed = 1,
): Attribution {
  const p = instance.length;
  const rng = createRng(seed);
  const base = coalitionValue(f, instance, background, new Set());
  const full = coalitionValue(
    f,
    instance,
    background,
    new Set(Array.from({ length: p }, (_, j) => j)),
  );
  // distribution over coalition sizes proportional to the kernel mass of that size: (p - 1) / (s (p - s))
  const sizes = Array.from({ length: p - 1 }, (_, i) => i + 1);
  const mass = sizes.map((s) => (p - 1) / (s * (p - s)));
  const total = mass.reduce((a, b) => a + b, 0);
  const rows: { z: number[]; y: number }[] = [];
  const seen = new Set<string>();
  const draw = (): number[] => {
    let u = rng.next() * total;
    let size = sizes[sizes.length - 1] as number;
    for (let i = 0; i < sizes.length; i++) {
      u -= mass[i] as number;
      if (u <= 0) {
        size = sizes[i] as number;
        break;
      }
    }
    const idx = Array.from({ length: p }, (_, j) => j);
    for (let i = p - 1; i > 0; i--) {
      const j = Math.floor(rng.next() * (i + 1));
      [idx[i], idx[j]] = [idx[j] as number, idx[i] as number];
    }
    const z = new Array<number>(p).fill(0);
    for (let k = 0; k < size; k++) z[idx[k] as number] = 1;
    return z;
  };
  for (let n = 0; n < samples; n++) {
    const z = draw();
    const key = z.join('');
    if (seen.has(key)) continue;
    seen.add(key);
    const set = new Set(z.map((v, j) => (v ? j : -1)).filter((j) => j >= 0));
    rows.push({ z, y: coalitionValue(f, instance, background, set) });
  }
  // eliminate the last feature with the efficiency constraint: phi_p = (full - base) - sum of the others
  const m = rows.length;
  const q = p - 1;
  const a = new Matrix(q, q);
  const b = new Matrix(q, 1);
  for (const { z, y } of rows) {
    const size = z.reduce((s, v) => s + v, 0);
    // each sampled coalition counts equally: sizes were already drawn by kernel mass
    const w = 1 / Math.max(m, 1);
    void size;
    const zl = z[p - 1] as number;
    const target = y - base - zl * (full - base);
    for (let i = 0; i < q; i++) {
      const xi = (z[i] as number) - zl;
      b.set(i, 0, b.get(i, 0) + w * xi * target);
      for (let j = 0; j < q; j++) a.set(i, j, a.get(i, j) + w * xi * ((z[j] as number) - zl));
    }
  }
  for (let i = 0; i < q; i++) a.set(i, i, a.get(i, i) + 1e-9);
  const phi = Array.from(solve(a, b).data);
  const last = full - base - phi.reduce((s, v) => s + v, 0);
  return { values: [...phi, last], base, prediction: full };
}

export interface LimeOptions {
  samples: number;
  /** Width of the neighbourhood in standard deviations of each feature. */
  width: number;
  /** Kernel width on the standardised distance. */
  kernelWidth: number;
  ridge: number;
  seed: number;
}

export const DEFAULT_LIME: LimeOptions = {
  samples: 500,
  width: 0.5,
  kernelWidth: 1.5,
  ridge: 1,
  seed: 1,
};

/** LIME: perturb the instance, weight the perturbations by closeness, fit a weighted ridge regression; its coefficients explain. */
export function lime(
  f: Model,
  instance: readonly number[],
  data: Matrix,
  options: LimeOptions = DEFAULT_LIME,
): { coefficients: number[]; intercept: number; fidelity: number } {
  const p = instance.length;
  const rng = createRng(options.seed);
  const sd = Array.from({ length: p }, (_, j) => {
    let m = 0;
    for (let i = 0; i < data.rows; i++) m += data.get(i, j) / data.rows;
    let v = 0;
    for (let i = 0; i < data.rows; i++) v += (data.get(i, j) - m) ** 2 / data.rows;
    return Math.sqrt(v) || 1;
  });
  const n = options.samples;
  const probe = new Matrix(n, p);
  const z = new Matrix(n, p);
  for (let i = 0; i < n; i++)
    for (let j = 0; j < p; j++) {
      const d = rng.normal() * options.width;
      z.set(i, j, d);
      probe.set(i, j, (instance[j] as number) + d * (sd[j] as number));
    }
  const y = f(probe);
  const w = Array.from({ length: n }, (_, i) => {
    let d2 = 0;
    for (let j = 0; j < p; j++) d2 += z.get(i, j) ** 2;
    return Math.exp(-d2 / (options.kernelWidth ** 2 * p));
  });
  // weighted ridge on the standardised offsets, with an intercept
  const q = p + 1;
  const a = new Matrix(q, q);
  const b = new Matrix(q, 1);
  for (let i = 0; i < n; i++) {
    const row = [1, ...Array.from({ length: p }, (_, j) => z.get(i, j))];
    for (let r = 0; r < q; r++) {
      b.set(r, 0, b.get(r, 0) + (w[i] as number) * (row[r] as number) * (y[i] as number));
      for (let c = 0; c < q; c++)
        a.set(r, c, a.get(r, c) + (w[i] as number) * (row[r] as number) * (row[c] as number));
    }
  }
  for (let r = 1; r < q; r++) a.set(r, r, a.get(r, r) + options.ridge);
  const beta = solve(a, b);
  // coefficients per unit of the feature (not per standard deviation) so they compare with a linear model's weights
  const coefficients = Array.from({ length: p }, (_, j) => beta.get(j + 1, 0) / (sd[j] as number));
  // local fidelity: weighted R^2 of the surrogate
  let sw = 0;
  let sy = 0;
  for (let i = 0; i < n; i++) {
    sw += w[i] as number;
    sy += (w[i] as number) * (y[i] as number);
  }
  const ym = sy / sw;
  let ssr = 0;
  let sst = 0;
  for (let i = 0; i < n; i++) {
    let pred = beta.get(0, 0);
    for (let j = 0; j < p; j++) pred += beta.get(j + 1, 0) * z.get(i, j);
    ssr += (w[i] as number) * ((y[i] as number) - pred) ** 2;
    sst += (w[i] as number) * ((y[i] as number) - ym) ** 2;
  }
  return { coefficients, intercept: beta.get(0, 0), fidelity: sst > 0 ? 1 - ssr / sst : 1 };
}

export { rowOf as instanceOf };
