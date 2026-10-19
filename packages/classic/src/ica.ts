import { Matrix, createRng, symmetricEigen } from '@sim/core';

/**
 * Independent component analysis (FastICA). PCA finds uncorrelated directions; ICA looks for
 * directions that are *statistically independent*, which is what unmixes signals that were added
 * together (the cocktail-party problem). The data are first whitened (so every direction has unit
 * variance and none is correlated), after which the unmixing matrix is a rotation; FastICA finds the
 * rotation that makes each output as non-Gaussian as possible, because a sum of independent
 * signals is always closer to Gaussian than the signals themselves.
 */
export type IcaNonlinearity = 'logcosh' | 'exp' | 'cube';

export interface IcaOptions {
  components?: number;
  nonlinearity?: IcaNonlinearity;
  maxIterations?: number;
  tolerance?: number;
  seed?: number;
}

/** Excess kurtosis of a sample (0 for a Gaussian; sources are usually far from 0). */
export function kurtosis(values: ArrayLike<number>): number {
  const n = values.length;
  let m = 0;
  for (let i = 0; i < n; i++) m += values[i] as number;
  m /= n;
  let m2 = 0;
  let m4 = 0;
  for (let i = 0; i < n; i++) {
    const d = (values[i] as number) - m;
    m2 += d * d;
    m4 += d ** 4;
  }
  m2 /= n;
  m4 /= n;
  return m4 / (m2 * m2) - 3;
}

/** Symmetric decorrelation: `W ← (W Wᵀ)^(−1/2) W`, which keeps the rows orthonormal. */
function decorrelate(w: Matrix): Matrix {
  const k = w.rows;
  const g = new Matrix(k, k);
  for (let i = 0; i < k; i++)
    for (let j = 0; j < k; j++) {
      let s = 0;
      for (let c = 0; c < w.cols; c++) s += w.get(i, c) * w.get(j, c);
      g.set(i, j, s);
    }
  const { values, vectors } = symmetricEigen(g);
  const out = new Matrix(k, w.cols);
  for (let i = 0; i < k; i++)
    for (let c = 0; c < w.cols; c++) {
      let s = 0;
      for (let p = 0; p < k; p++) {
        let inv = 0;
        for (let q = 0; q < k; q++)
          inv +=
            (vectors.get(i, q) * vectors.get(p, q)) /
            Math.sqrt(Math.max(values[q] as number, 1e-12));
        s += inv * w.get(p, c);
      }
      out.set(i, c, s);
    }
  return out;
}

export interface IcaState {
  iteration: number;
  /** Largest remaining change of an unmixing direction (0 = converged). */
  change: number;
  sources: Matrix;
}

export class FastICA {
  mean: Float64Array = new Float64Array(0);
  /** Whitening matrix `k × d` (rows scaled so every whitened coordinate has unit variance). */
  whitening: Matrix = new Matrix(0, 0);
  /** Rotation `k × k` found by FastICA, acting on whitened data. */
  rotation: Matrix = new Matrix(0, 0);
  /** Full unmixing matrix `k × d` (rotation × whitening): sources = (x − mean) · componentsᵀ. */
  components: Matrix = new Matrix(0, 0);
  iterations = 0;
  converged = false;

  constructor(readonly options: IcaOptions = {}) {}

  private whiten(x: Matrix): Matrix {
    const n = x.rows;
    const d = x.cols;
    this.mean = new Float64Array(d);
    for (let j = 0; j < d; j++) {
      for (let i = 0; i < n; i++) this.mean[j] = (this.mean[j] as number) + x.get(i, j) / n;
    }
    const cov = new Matrix(d, d);
    for (let a = 0; a < d; a++)
      for (let b = a; b < d; b++) {
        let s = 0;
        for (let i = 0; i < n; i++)
          s += (x.get(i, a) - (this.mean[a] as number)) * (x.get(i, b) - (this.mean[b] as number));
        cov.set(a, b, s / n);
        cov.set(b, a, s / n);
      }
    const { values, vectors } = symmetricEigen(cov);
    const k = Math.min(this.options.components ?? d, d);
    this.whitening = new Matrix(k, d);
    for (let c = 0; c < k; c++) {
      const scale = 1 / Math.sqrt(Math.max(values[c] as number, 1e-12));
      for (let j = 0; j < d; j++) this.whitening.set(c, j, vectors.get(j, c) * scale);
    }
    const z = new Matrix(n, k);
    for (let i = 0; i < n; i++)
      for (let c = 0; c < k; c++) {
        let s = 0;
        for (let j = 0; j < d; j++)
          s += (x.get(i, j) - (this.mean[j] as number)) * this.whitening.get(c, j);
        z.set(i, c, s);
      }
    return z;
  }

  /** The fixed-point iterations as a generator (state 0 is the random start). */
  *states(x: Matrix): Generator<IcaState, void, undefined> {
    const {
      nonlinearity = 'logcosh',
      maxIterations = 200,
      tolerance = 1e-5,
      seed = 1,
    } = this.options;
    const z = this.whiten(x);
    const n = z.rows;
    const k = z.cols;
    const rng = createRng(seed);
    let w = decorrelate(
      new Matrix(
        k,
        k,
        Float64Array.from({ length: k * k }, () => rng.normal()),
      ),
    );
    const sourcesOf = (rot: Matrix): Matrix => {
      const s = new Matrix(n, k);
      for (let i = 0; i < n; i++)
        for (let c = 0; c < k; c++) {
          let v = 0;
          for (let j = 0; j < k; j++) v += z.get(i, j) * rot.get(c, j);
          s.set(i, c, v);
        }
      return s;
    };
    this.rotation = w;
    yield { iteration: 0, change: Infinity, sources: sourcesOf(w) };
    for (let it = 1; it <= maxIterations; it++) {
      const next = new Matrix(k, k);
      for (let c = 0; c < k; c++) {
        let gp = 0;
        const acc = new Float64Array(k);
        for (let i = 0; i < n; i++) {
          let u = 0;
          for (let j = 0; j < k; j++) u += w.get(c, j) * z.get(i, j);
          let g: number;
          let dg: number;
          if (nonlinearity === 'logcosh') {
            g = Math.tanh(u);
            dg = 1 - g * g;
          } else if (nonlinearity === 'exp') {
            const e = Math.exp((-u * u) / 2);
            g = u * e;
            dg = (1 - u * u) * e;
          } else {
            g = u ** 3;
            dg = 3 * u * u;
          }
          gp += dg;
          for (let j = 0; j < k; j++) acc[j] = (acc[j] as number) + g * z.get(i, j);
        }
        for (let j = 0; j < k; j++) next.set(c, j, (acc[j] as number) / n - (gp / n) * w.get(c, j));
      }
      const rotated = decorrelate(next);
      // converged when every new direction is (up to sign) the old one
      let change = 0;
      for (let c = 0; c < k; c++) {
        let dot = 0;
        for (let j = 0; j < k; j++) dot += rotated.get(c, j) * w.get(c, j);
        change = Math.max(change, Math.abs(1 - Math.abs(dot)));
      }
      w = rotated;
      this.rotation = w;
      this.iterations = it;
      yield { iteration: it, change, sources: sourcesOf(w) };
      if (change < tolerance) {
        this.converged = true;
        break;
      }
    }
  }

  fit(x: Matrix): this {
    this.converged = false;
    for (const _ of this.states(x)) void _;
    const k = this.rotation.rows;
    const d = x.cols;
    this.components = new Matrix(k, d);
    for (let c = 0; c < k; c++)
      for (let j = 0; j < d; j++) {
        let s = 0;
        for (let q = 0; q < k; q++) s += this.rotation.get(c, q) * this.whitening.get(q, j);
        this.components.set(c, j, s);
      }
    return this;
  }

  /** The recovered independent signals of `x` (one column per source). */
  transform(x: Matrix): Matrix {
    const k = this.components.rows;
    const out = new Matrix(x.rows, k);
    for (let i = 0; i < x.rows; i++)
      for (let c = 0; c < k; c++) {
        let s = 0;
        for (let j = 0; j < x.cols; j++)
          s += (x.get(i, j) - (this.mean[j] as number)) * this.components.get(c, j);
        out.set(i, c, s);
      }
    return out;
  }
}

/** Best absolute correlation between each true source and any recovered one (1 = recovered up to sign and scale). */
export function sourceRecovery(trueSources: Matrix, recovered: Matrix): number[] {
  const col = (m: Matrix, c: number) => Array.from({ length: m.rows }, (_, i) => m.get(i, c));
  const corr = (a: number[], b: number[]) => {
    const ma = a.reduce((s, v) => s + v, 0) / a.length;
    const mb = b.reduce((s, v) => s + v, 0) / b.length;
    let num = 0;
    let da = 0;
    let db = 0;
    a.forEach((v, i) => {
      num += (v - ma) * ((b[i] as number) - mb);
      da += (v - ma) ** 2;
      db += ((b[i] as number) - mb) ** 2;
    });
    return num / Math.sqrt(da * db || 1);
  };
  return Array.from({ length: trueSources.cols }, (_, s) =>
    Math.max(
      ...Array.from({ length: recovered.cols }, (_, r) =>
        Math.abs(corr(col(trueSources, s), col(recovered, r))),
      ),
    ),
  );
}
