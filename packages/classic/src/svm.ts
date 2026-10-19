import { Matrix, type Classifier, type Regressor } from '@sim/core';

/**
 * Support vector machines trained with sequential minimal optimisation (SMO), following libsvm
 * (second-order working-set selection), so the support vectors, dual coefficients and intercept
 * match scikit-learn's `SVC` / `SVR`.
 */
export type KernelName = 'linear' | 'rbf' | 'poly' | 'sigmoid';

export interface KernelOptions {
  kernel?: KernelName;
  /** `'scale'` = 1 / (d · var(X)), `'auto'` = 1 / d, or a number. */
  gamma?: 'scale' | 'auto' | number;
  degree?: number;
  coef0?: number;
}

export interface SvmOptions extends KernelOptions {
  /** Penalty on margin violations (larger = harder margin). */
  C?: number;
  /** KKT tolerance; scikit-learn's default is 1e-3. */
  tol?: number;
  maxIterations?: number;
}

function resolveGamma(g: KernelOptions['gamma'], x: Matrix): number {
  if (typeof g === 'number') return g;
  if (g === 'auto') return 1 / x.cols;
  let mean = 0;
  for (const v of x.data) mean += v;
  mean /= x.data.length;
  let variance = 0;
  for (const v of x.data) variance += (v - mean) ** 2;
  variance /= x.data.length;
  return variance > 0 ? 1 / (x.cols * variance) : 1 / x.cols;
}

/** A kernel function `k(a, b)` with the parameters resolved for the training data `x`. */
export function makeKernel(
  options: KernelOptions,
  x: Matrix,
): (a: ArrayLike<number>, b: ArrayLike<number>) => number {
  const { kernel = 'rbf', degree = 3, coef0 = 0 } = options;
  const gamma = resolveGamma(options.gamma ?? 'scale', x);
  switch (kernel) {
    case 'linear':
      return (a, b) => {
        let s = 0;
        for (let j = 0; j < a.length; j++) s += (a[j] as number) * (b[j] as number);
        return s;
      };
    case 'poly':
      return (a, b) => {
        let s = 0;
        for (let j = 0; j < a.length; j++) s += (a[j] as number) * (b[j] as number);
        return (gamma * s + coef0) ** degree;
      };
    case 'sigmoid':
      return (a, b) => {
        let s = 0;
        for (let j = 0; j < a.length; j++) s += (a[j] as number) * (b[j] as number);
        return Math.tanh(gamma * s + coef0);
      };
    default:
      return (a, b) => {
        let s = 0;
        for (let j = 0; j < a.length; j++) s += ((a[j] as number) - (b[j] as number)) ** 2;
        return Math.exp(-gamma * s);
      };
  }
}

export const resolvedGamma = (options: KernelOptions, x: Matrix): number =>
  resolveGamma(options.gamma ?? 'scale', x);

interface SolveResult {
  alpha: Float64Array;
  rho: number;
  iterations: number;
}

/**
 * Minimises ½αᵀQα + pᵀα subject to 0 ≤ α ≤ C and yᵀα = 0, with Q = (y y ᵀ) ∘ K, by SMO.
 * `kernelAt(i, j)` returns K for the *underlying* rows, so SVR can reuse it for its doubled problem.
 */
function smo(
  y: Int8Array,
  p: Float64Array,
  kernelAt: (i: number, j: number) => number,
  C: number,
  tol: number,
  maxIterations: number,
): SolveResult {
  const l = y.length;
  const TAU = 1e-12;
  const alpha = new Float64Array(l);
  const G = Float64Array.from(p);
  const QD = new Float64Array(l);
  for (let i = 0; i < l; i++) QD[i] = kernelAt(i, i);
  // the full row of Q on demand (n is a few hundred: no cache needed)
  const row = (i: number): Float64Array => {
    const out = new Float64Array(l);
    for (let j = 0; j < l; j++) out[j] = (y[i] as number) * (y[j] as number) * kernelAt(i, j);
    return out;
  };

  let iterations = 0;
  for (; iterations < maxIterations; iterations++) {
    // working-set selection (second order)
    let gmax = -Infinity;
    let gmaxIdx = -1;
    for (let t = 0; t < l; t++) {
      const up = y[t] === 1 ? (alpha[t] as number) < C : (alpha[t] as number) > 0;
      if (up) {
        const v = -(y[t] as number) * (G[t] as number);
        if (v >= gmax) {
          gmax = v;
          gmaxIdx = t;
        }
      }
    }
    const i = gmaxIdx;
    const qi = i >= 0 ? row(i) : new Float64Array(l);
    let j = -1;
    let gmax2 = -Infinity;
    let objMin = Infinity;
    for (let t = 0; t < l; t++) {
      if (y[t] === 1) {
        if ((alpha[t] as number) > 0) {
          const gradDiff = gmax + (G[t] as number);
          if ((G[t] as number) >= gmax2) gmax2 = G[t] as number;
          if (gradDiff > 0) {
            let quad =
              (QD[i] as number) + (QD[t] as number) - 2 * (y[i] as number) * (qi[t] as number);
            if (quad <= 0) quad = TAU;
            const obj = -(gradDiff * gradDiff) / quad;
            if (obj <= objMin) {
              j = t;
              objMin = obj;
            }
          }
        }
      } else if ((alpha[t] as number) < C) {
        const gradDiff = gmax - (G[t] as number);
        if (-(G[t] as number) >= gmax2) gmax2 = -(G[t] as number);
        if (gradDiff > 0) {
          let quad =
            (QD[i] as number) + (QD[t] as number) + 2 * (y[i] as number) * (qi[t] as number);
          if (quad <= 0) quad = TAU;
          const obj = -(gradDiff * gradDiff) / quad;
          if (obj <= objMin) {
            j = t;
            objMin = obj;
          }
        }
      }
    }
    if (gmax + gmax2 < tol || j < 0 || i < 0) break;

    const qj = row(j);
    const oldI = alpha[i] as number;
    const oldJ = alpha[j] as number;
    if (y[i] !== y[j]) {
      let quad = (QD[i] as number) + (QD[j] as number) + 2 * (qi[j] as number);
      if (quad <= 0) quad = TAU;
      const delta = (-(G[i] as number) - (G[j] as number)) / quad;
      const diff = (alpha[i] as number) - (alpha[j] as number);
      alpha[i] = (alpha[i] as number) + delta;
      alpha[j] = (alpha[j] as number) + delta;
      if (diff > 0) {
        if ((alpha[j] as number) < 0) {
          alpha[j] = 0;
          alpha[i] = diff;
        }
      } else if ((alpha[i] as number) < 0) {
        alpha[i] = 0;
        alpha[j] = -diff;
      }
      if (diff > 0) {
        if ((alpha[i] as number) > C) {
          alpha[i] = C;
          alpha[j] = C - diff;
        }
      } else if ((alpha[j] as number) > C) {
        alpha[j] = C;
        alpha[i] = C + diff;
      }
    } else {
      let quad = (QD[i] as number) + (QD[j] as number) - 2 * (qi[j] as number);
      if (quad <= 0) quad = TAU;
      const delta = ((G[i] as number) - (G[j] as number)) / quad;
      const sum = (alpha[i] as number) + (alpha[j] as number);
      alpha[i] = (alpha[i] as number) - delta;
      alpha[j] = (alpha[j] as number) + delta;
      if (sum > C) {
        if ((alpha[i] as number) > C) {
          alpha[i] = C;
          alpha[j] = sum - C;
        }
      } else if ((alpha[j] as number) < 0) {
        alpha[j] = 0;
        alpha[i] = sum;
      }
      if (sum > C) {
        if ((alpha[j] as number) > C) {
          alpha[j] = C;
          alpha[i] = sum - C;
        }
      } else if ((alpha[i] as number) < 0) {
        alpha[i] = 0;
        alpha[j] = sum;
      }
    }
    const di = (alpha[i] as number) - oldI;
    const dj = (alpha[j] as number) - oldJ;
    for (let k = 0; k < l; k++)
      G[k] = (G[k] as number) + (qi[k] as number) * di + (qj[k] as number) * dj;
  }

  // the offset: average y·G over the free variables, or the middle of the feasible interval
  let ub = Infinity;
  let lb = -Infinity;
  let sumFree = 0;
  let free = 0;
  for (let i = 0; i < l; i++) {
    const yG = (y[i] as number) * (G[i] as number);
    const a = alpha[i] as number;
    if (a >= C) {
      if (y[i] === -1) ub = Math.min(ub, yG);
      else lb = Math.max(lb, yG);
    } else if (a <= 0) {
      if (y[i] === 1) ub = Math.min(ub, yG);
      else lb = Math.max(lb, yG);
    } else {
      free++;
      sumFree += yG;
    }
  }
  return { alpha, rho: free > 0 ? sumFree / free : (ub + lb) / 2, iterations };
}

/** Binary machine: support vectors, their signed dual coefficients and the offset. */
interface BinaryModel {
  /** Indices (into the training rows the model was given) of the support vectors. */
  support: number[];
  /** `αᵢ yᵢ` for each support vector. */
  coef: number[];
  rho: number;
  iterations: number;
}

function fitBinary(
  x: Matrix,
  labels: Int8Array,
  kernel: (a: ArrayLike<number>, b: ArrayLike<number>) => number,
  C: number,
  tol: number,
  maxIterations: number,
): BinaryModel {
  const n = x.rows;
  const rows = Array.from({ length: n }, (_, i) => x.data.subarray(i * x.cols, (i + 1) * x.cols));
  const cache = new Float64Array(n * n).fill(NaN);
  const k = (i: number, j: number) => {
    const idx = i * n + j;
    const hit = cache[idx] as number;
    if (!Number.isNaN(hit)) return hit;
    const v = kernel(rows[i] as Float64Array, rows[j] as Float64Array);
    cache[idx] = v;
    cache[j * n + i] = v;
    return v;
  };
  const { alpha, rho, iterations } = smo(
    labels,
    new Float64Array(n).fill(-1),
    k,
    C,
    tol,
    maxIterations,
  );
  const support: number[] = [];
  const coef: number[] = [];
  for (let i = 0; i < n; i++)
    if ((alpha[i] as number) > 0) {
      support.push(i);
      coef.push((alpha[i] as number) * (labels[i] as number));
    }
  return { support, coef, rho, iterations };
}

export class SVC implements Classifier {
  classes = 2;
  /** Indices of the support vectors in the training set (binary problems and one-vs-one pairs merged). */
  supportIndices: number[] = [];
  /** Number of SMO iterations of the (first) binary machine. */
  iterations = 0;
  private train: Matrix = new Matrix(0, 0);
  private kernel: (a: ArrayLike<number>, b: ArrayLike<number>) => number = () => 0;
  /** One machine per pair of classes (a, b), a < b; positive output votes for b. */
  private machines: { a: number; b: number; rows: number[]; model: BinaryModel }[] = [];

  constructor(readonly options: SvmOptions = {}) {}

  fit(x: Matrix, y: ArrayLike<number>): this {
    const { C = 1, tol = 1e-3, maxIterations = 100000 } = this.options;
    let k = 0;
    for (let i = 0; i < y.length; i++) k = Math.max(k, (y[i] as number) + 1);
    this.classes = Math.max(2, k);
    this.train = x;
    this.kernel = makeKernel(this.options, x);
    this.machines = [];
    const support = new Set<number>();
    for (let a = 0; a < this.classes; a++)
      for (let b = a + 1; b < this.classes; b++) {
        const rows: number[] = [];
        for (let i = 0; i < x.rows; i++) if (y[i] === a || y[i] === b) rows.push(i);
        const sub = new Matrix(rows.length, x.cols);
        rows.forEach((r, i) =>
          sub.data.set(x.data.subarray(r * x.cols, (r + 1) * x.cols), i * x.cols),
        );
        const labels = Int8Array.from(rows, (r) => (y[r] === b ? 1 : -1));
        const model = fitBinary(sub, labels, this.kernel, C, tol, maxIterations);
        model.support.forEach((s) => support.add(rows[s] as number));
        this.machines.push({ a, b, rows, model });
      }
    this.supportIndices = [...support].sort((p, q) => p - q);
    this.iterations = this.machines[0]?.model.iterations ?? 0;
    return this;
  }

  /** Signed distance-like score of the (first, for two classes the only) machine; positive favours class 1. */
  decision(x: Matrix, machine = 0): Float64Array {
    const m = this.machines[machine];
    if (!m) return new Float64Array(x.rows);
    return Float64Array.from({ length: x.rows }, (_, i) => {
      const q = x.data.subarray(i * x.cols, (i + 1) * x.cols);
      let s = -m.model.rho;
      m.model.support.forEach((sv, t) => {
        const r = m.rows[sv] as number;
        s +=
          (m.model.coef[t] as number) *
          this.kernel(q, this.train.data.subarray(r * this.train.cols, (r + 1) * this.train.cols));
      });
      return s;
    });
  }

  /** Dual coefficients `αᵢ yᵢ` of the binary machine, aligned with `supportIndices` of that machine. */
  binaryModel(machine = 0): { support: number[]; coef: number[]; intercept: number } | null {
    const m = this.machines[machine];
    return m
      ? {
          support: m.model.support.map((s) => m.rows[s] as number),
          coef: [...m.model.coef],
          intercept: -m.model.rho,
        }
      : null;
  }

  /** One-vs-one votes per class. */
  votes(x: Matrix): Matrix {
    const out = new Matrix(x.rows, this.classes);
    this.machines.forEach((m, idx) => {
      const d = this.decision(x, idx);
      for (let i = 0; i < x.rows; i++) {
        const winner = (d[i] as number) > 0 ? m.b : m.a;
        out.set(i, winner, out.get(i, winner) + 1);
      }
    });
    return out;
  }

  predict(x: Matrix): Int32Array {
    const v = this.votes(x);
    return Int32Array.from({ length: x.rows }, (_, i) => {
      let best = 0;
      for (let c = 1; c < this.classes; c++) if (v.get(i, c) > v.get(i, best)) best = c;
      return best;
    });
  }

  /**
   * Squashed scores, **not** calibrated probabilities: the sigmoid of the margin for two classes
   * and the share of one-vs-one votes otherwise. Good for shading, not for decisions that need real
   * probabilities.
   */
  predictProba(x: Matrix): Matrix {
    const out = new Matrix(x.rows, this.classes);
    if (this.classes === 2) {
      const d = this.decision(x);
      for (let i = 0; i < x.rows; i++) {
        const p1 = 1 / (1 + Math.exp(-2 * (d[i] as number)));
        out.set(i, 0, 1 - p1);
        out.set(i, 1, p1);
      }
      return out;
    }
    const v = this.votes(x);
    for (let i = 0; i < x.rows; i++) {
      let total = 0;
      for (let c = 0; c < this.classes; c++) total += v.get(i, c) + 1e-3;
      for (let c = 0; c < this.classes; c++) out.set(i, c, (v.get(i, c) + 1e-3) / total);
    }
    return out;
  }
}

export interface SvrOptions extends SvmOptions {
  /** Half-width of the tube inside which errors cost nothing. */
  epsilon?: number;
}

/** ε-insensitive support vector regression. */
export class SVR implements Regressor {
  support: number[] = [];
  /** `αᵢ⁺ − αᵢ⁻` for each support vector. */
  coef: number[] = [];
  intercept = 0;
  private train: Matrix = new Matrix(0, 0);
  private kernel: (a: ArrayLike<number>, b: ArrayLike<number>) => number = () => 0;

  constructor(readonly options: SvrOptions = {}) {}

  fit(x: Matrix, y: ArrayLike<number>): this {
    const { C = 1, tol = 1e-3, maxIterations = 100000, epsilon = 0.1 } = this.options;
    const n = x.rows;
    this.train = x;
    this.kernel = makeKernel(this.options, x);
    const rows = Array.from({ length: n }, (_, i) => x.data.subarray(i * x.cols, (i + 1) * x.cols));
    const cache = new Float64Array(n * n).fill(NaN);
    const k = (i: number, j: number) => {
      const a = i % n;
      const b = j % n;
      const idx = a * n + b;
      const hit = cache[idx] as number;
      if (!Number.isNaN(hit)) return hit;
      const v = this.kernel(rows[a] as Float64Array, rows[b] as Float64Array);
      cache[idx] = v;
      cache[b * n + a] = v;
      return v;
    };
    // 2n variables: αᵢ⁺ (sign +1, linear term ε − zᵢ) and αᵢ⁻ (sign −1, linear term ε + zᵢ)
    const signs = new Int8Array(2 * n);
    const lin = new Float64Array(2 * n);
    for (let i = 0; i < n; i++) {
      signs[i] = 1;
      lin[i] = epsilon - (y[i] as number);
      signs[i + n] = -1;
      lin[i + n] = epsilon + (y[i] as number);
    }
    const { alpha, rho } = smo(signs, lin, k, C, tol, maxIterations);
    this.support = [];
    this.coef = [];
    for (let i = 0; i < n; i++) {
      const b = (alpha[i] as number) - (alpha[i + n] as number);
      if (b !== 0) {
        this.support.push(i);
        this.coef.push(b);
      }
    }
    this.intercept = -rho;
    return this;
  }

  predict(x: Matrix): Float64Array {
    return Float64Array.from({ length: x.rows }, (_, i) => {
      const q = x.data.subarray(i * x.cols, (i + 1) * x.cols);
      let s = this.intercept;
      this.support.forEach((sv, t) => {
        s +=
          (this.coef[t] as number) *
          this.kernel(
            q,
            this.train.data.subarray(sv * this.train.cols, (sv + 1) * this.train.cols),
          );
      });
      return s;
    });
  }
}
