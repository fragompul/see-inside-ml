import { Matrix, createRng } from '@sim/core';

/**
 * Non-negative matrix factorisation: write a non-negative table `V` (n × m) as `W · H` with both
 * factors non-negative, `W` (n × k) and `H` (k × m). Because nothing can be subtracted, the rows of
 * `H` come out as additive *parts* (strokes of a digit, topics of documents) and `W` says how much
 * of each part every row uses. Multiplicative updates (Lee & Seung) keep the factors non-negative
 * and never increase the error.
 */
export type NmfLoss = 'frobenius' | 'kl';

export interface NmfOptions {
  components: number;
  loss?: NmfLoss;
  iterations?: number;
  seed?: number;
}

export interface NmfState {
  iteration: number;
  w: Matrix;
  h: Matrix;
  /** Frobenius: ‖V − WH‖²; KL: the generalised Kullback–Leibler divergence. */
  error: number;
}

const EPS = 1e-10;

function product(w: Matrix, h: Matrix): Matrix {
  const out = new Matrix(w.rows, h.cols);
  for (let i = 0; i < w.rows; i++)
    for (let k = 0; k < w.cols; k++) {
      const a = w.get(i, k);
      if (a === 0) continue;
      for (let j = 0; j < h.cols; j++)
        out.data[i * h.cols + j] = (out.data[i * h.cols + j] as number) + a * h.get(k, j);
    }
  return out;
}

export function nmfError(v: Matrix, w: Matrix, h: Matrix, loss: NmfLoss): number {
  const wh = product(w, h);
  let s = 0;
  for (let i = 0; i < v.data.length; i++) {
    const a = v.data[i] as number;
    const b = wh.data[i] as number;
    if (loss === 'frobenius') s += (a - b) ** 2;
    else s += (a > 0 ? a * Math.log(a / Math.max(b, EPS)) : 0) - a + b;
  }
  return s;
}

/** The multiplicative-update iterations as a generator (state 0 is the random start). */
export function* nmfSteps(v: Matrix, options: NmfOptions): Generator<NmfState, void, undefined> {
  const { components: k, loss = 'frobenius', iterations = 200, seed = 1 } = options;
  const n = v.rows;
  const m = v.cols;
  const rng = createRng(seed);
  let mean = 0;
  for (const a of v.data) mean += a;
  mean /= v.data.length;
  const scale = Math.sqrt(mean / k) || 1;
  let w = new Matrix(
    n,
    k,
    Float64Array.from({ length: n * k }, () => (0.1 + rng.next()) * scale),
  );
  let h = new Matrix(
    k,
    m,
    Float64Array.from({ length: k * m }, () => (0.1 + rng.next()) * scale),
  );
  yield { iteration: 0, w, h, error: nmfError(v, w, h, loss) };
  for (let it = 1; it <= iterations; it++) {
    if (loss === 'frobenius') {
      // H ← H ∘ (WᵀV) / (WᵀW H);  W ← W ∘ (V Hᵀ) / (W H Hᵀ)
      const wtv = new Matrix(k, m);
      const wtw = new Matrix(k, k);
      for (let a = 0; a < k; a++) {
        for (let b = 0; b < k; b++) {
          let s = 0;
          for (let i = 0; i < n; i++) s += w.get(i, a) * w.get(i, b);
          wtw.set(a, b, s);
        }
        for (let j = 0; j < m; j++) {
          let s = 0;
          for (let i = 0; i < n; i++) s += w.get(i, a) * v.get(i, j);
          wtv.set(a, j, s);
        }
      }
      const hn = new Matrix(k, m);
      for (let a = 0; a < k; a++)
        for (let j = 0; j < m; j++) {
          let den = 0;
          for (let b = 0; b < k; b++) den += wtw.get(a, b) * h.get(b, j);
          hn.set(a, j, (h.get(a, j) * wtv.get(a, j)) / (den + EPS));
        }
      h = hn;
      const vht = new Matrix(n, k);
      const hht = new Matrix(k, k);
      for (let a = 0; a < k; a++)
        for (let b = 0; b < k; b++) {
          let s = 0;
          for (let j = 0; j < m; j++) s += h.get(a, j) * h.get(b, j);
          hht.set(a, b, s);
        }
      for (let i = 0; i < n; i++)
        for (let a = 0; a < k; a++) {
          let s = 0;
          for (let j = 0; j < m; j++) s += v.get(i, j) * h.get(a, j);
          vht.set(i, a, s);
        }
      const wn = new Matrix(n, k);
      for (let i = 0; i < n; i++)
        for (let a = 0; a < k; a++) {
          let den = 0;
          for (let b = 0; b < k; b++) den += w.get(i, b) * hht.get(b, a);
          wn.set(i, a, (w.get(i, a) * vht.get(i, a)) / (den + EPS));
        }
      w = wn;
    } else {
      // KL: H ← H ∘ Wᵀ(V/WH) / Wᵀ1;  W ← W ∘ (V/WH)Hᵀ / 1Hᵀ
      let wh = product(w, h);
      const hn = new Matrix(k, m);
      for (let a = 0; a < k; a++) {
        let colSum = 0;
        for (let i = 0; i < n; i++) colSum += w.get(i, a);
        for (let j = 0; j < m; j++) {
          let num = 0;
          for (let i = 0; i < n; i++)
            num += (w.get(i, a) * v.get(i, j)) / Math.max(wh.get(i, j), EPS);
          hn.set(a, j, (h.get(a, j) * num) / (colSum + EPS));
        }
      }
      h = hn;
      wh = product(w, h);
      const wn = new Matrix(n, k);
      for (let a = 0; a < k; a++) {
        let rowSum = 0;
        for (let j = 0; j < m; j++) rowSum += h.get(a, j);
        for (let i = 0; i < n; i++) {
          let num = 0;
          for (let j = 0; j < m; j++)
            num += (v.get(i, j) * h.get(a, j)) / Math.max(wh.get(i, j), EPS);
          wn.set(i, a, (w.get(i, a) * num) / (rowSum + EPS));
        }
      }
      w = wn;
    }
    yield { iteration: it, w, h, error: nmfError(v, w, h, loss) };
  }
}

export class NMF {
  w: Matrix = new Matrix(0, 0);
  /** The parts: row `c` is the c-th building block. */
  h: Matrix = new Matrix(0, 0);
  error = 0;
  history: number[] = [];

  constructor(readonly options: NmfOptions) {}

  fit(v: Matrix): this {
    this.history = [];
    for (const s of nmfSteps(v, this.options)) {
      this.w = s.w;
      this.h = s.h;
      this.error = s.error;
      this.history.push(s.error);
    }
    return this;
  }

  /** `W · H`: the best additive approximation of the data. */
  reconstruction(): Matrix {
    return product(this.w, this.h);
  }
}
