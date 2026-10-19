import { Matrix, solve, type Classifier } from '@sim/core';
import { Glm } from './glm';

/**
 * Logistic regression, parametrised like scikit-learn: minimise `C · Σ log-loss + ½‖w‖²` (the
 * intercept is not penalised), so a small `C` means strong regularisation and `C = Infinity` none.
 * Two classes use the sigmoid (solved by IRLS, i.e. Newton's method); three or more use the
 * softmax with a damped Newton method on the full Hessian.
 */
export interface LogisticOptions {
  /** Inverse regularisation strength (default 1). */
  C?: number;
  maxIterations?: number;
}

export class LogisticRegression implements Classifier {
  classes = 2;
  /** `K × d` weights (one row for two classes: the log-odds of class 1). */
  weights: Matrix = new Matrix(0, 0);
  intercepts: Float64Array = new Float64Array(0);
  iterations = 0;

  constructor(readonly options: LogisticOptions = {}) {}

  private get lambda(): number {
    const c = this.options.C ?? 1;
    return Number.isFinite(c) && c > 0 ? 1 / c : 0;
  }

  fit(x: Matrix, y: ArrayLike<number>): this {
    let k = 0;
    for (let i = 0; i < y.length; i++) k = Math.max(k, (y[i] as number) + 1);
    this.classes = Math.max(2, k);
    if (this.classes === 2) this.fitBinary(x, y);
    else this.fitSoftmax(x, y);
    return this;
  }

  private fitBinary(x: Matrix, y: ArrayLike<number>): void {
    const glm = new Glm({
      family: 'binomial',
      l2: this.lambda,
      maxIterations: this.options.maxIterations ?? 100,
      tolerance: 1e-14,
    }).fit(x, y);
    this.iterations = glm.iterations;
    this.weights = new Matrix(1, x.cols, Float64Array.from(glm.coef));
    this.intercepts = Float64Array.of(glm.intercept);
  }

  private fitSoftmax(x: Matrix, y: ArrayLike<number>): void {
    const n = x.rows;
    const d = x.cols;
    const k = this.classes;
    const p = d + 1;
    const lambda = this.lambda;
    const theta = new Float64Array(k * p); // row-major [class][0 = intercept, 1…d = weights]
    const probs = (t: Float64Array): Float64Array => {
      const out = new Float64Array(n * k);
      for (let i = 0; i < n; i++) {
        let max = -Infinity;
        for (let c = 0; c < k; c++) {
          let s = t[c * p] as number;
          for (let j = 0; j < d; j++) s += x.get(i, j) * (t[c * p + 1 + j] as number);
          out[i * k + c] = s;
          max = Math.max(max, s);
        }
        let z = 0;
        for (let c = 0; c < k; c++) {
          const e = Math.exp((out[i * k + c] as number) - max);
          out[i * k + c] = e;
          z += e;
        }
        for (let c = 0; c < k; c++) out[i * k + c] = (out[i * k + c] as number) / z;
      }
      return out;
    };
    const loss = (t: Float64Array, pr: Float64Array): number => {
      let s = 0;
      for (let i = 0; i < n; i++)
        s -= Math.log(Math.max(pr[i * k + (y[i] as number)] as number, 1e-300));
      let reg = 0;
      for (let c = 0; c < k; c++) for (let j = 1; j < p; j++) reg += (t[c * p + j] as number) ** 2;
      return s + 0.5 * lambda * reg;
    };
    let pr = probs(theta);
    let current = loss(theta, pr);
    let it = 0;
    for (; it < (this.options.maxIterations ?? 100); it++) {
      const dim = k * p;
      const grad = new Float64Array(dim);
      const hess = new Matrix(dim, dim);
      for (let i = 0; i < n; i++) {
        const xi = (j: number) => (j === 0 ? 1 : x.get(i, j - 1));
        for (let c = 0; c < k; c++) {
          const pc = pr[i * k + c] as number;
          const r = pc - (y[i] === c ? 1 : 0);
          for (let a = 0; a < p; a++) grad[c * p + a] = (grad[c * p + a] as number) + r * xi(a);
          for (let l = 0; l < k; l++) {
            const pl = pr[i * k + l] as number;
            const s = pc * ((c === l ? 1 : 0) - pl);
            for (let a = 0; a < p; a++)
              for (let b = 0; b < p; b++) {
                const idx = (c * p + a) * dim + l * p + b;
                hess.data[idx] = (hess.data[idx] as number) + s * xi(a) * xi(b);
              }
          }
        }
      }
      for (let c = 0; c < k; c++)
        for (let j = 1; j < p; j++) {
          grad[c * p + j] = (grad[c * p + j] as number) + lambda * (theta[c * p + j] as number);
          hess.data[(c * p + j) * dim + c * p + j] =
            (hess.data[(c * p + j) * dim + c * p + j] as number) + lambda;
        }
      // the unpenalised softmax is invariant to adding a constant to every class's weights: keep H invertible
      for (let q = 0; q < dim; q++)
        hess.data[q * dim + q] = (hess.data[q * dim + q] as number) + 1e-9;
      const step = solve(hess, new Matrix(dim, 1, grad)).data;
      // damped Newton: halve the step until the loss decreases
      let t = 1;
      let next = theta;
      let nextPr = pr;
      let nextLoss = current;
      for (let tries = 0; tries < 30; tries++) {
        next = Float64Array.from(theta, (v, q) => v - t * (step[q] as number));
        nextPr = probs(next);
        nextLoss = loss(next, nextPr);
        if (nextLoss <= current + 1e-14 * Math.abs(current)) break;
        t /= 2;
      }
      const decrease = current - nextLoss;
      theta.set(next);
      pr = nextPr;
      current = nextLoss;
      if (decrease < 1e-14 * Math.max(1, Math.abs(current))) {
        it++;
        break;
      }
    }
    this.iterations = it;
    this.weights = new Matrix(k, d);
    this.intercepts = new Float64Array(k);
    for (let c = 0; c < k; c++) {
      this.intercepts[c] = theta[c * p] as number;
      for (let j = 0; j < d; j++) this.weights.set(c, j, theta[c * p + 1 + j] as number);
    }
  }

  /** Raw scores: the log-odds (one column) or the softmax logits (one per class). */
  decision(x: Matrix): Matrix {
    const rows = this.weights.rows;
    const out = new Matrix(x.rows, rows);
    for (let i = 0; i < x.rows; i++)
      for (let c = 0; c < rows; c++) {
        let s = this.intercepts[c] as number;
        for (let j = 0; j < x.cols; j++) s += x.get(i, j) * this.weights.get(c, j);
        out.set(i, c, s);
      }
    return out;
  }

  predictProba(x: Matrix): Matrix {
    const z = this.decision(x);
    const out = new Matrix(x.rows, this.classes);
    for (let i = 0; i < x.rows; i++) {
      if (this.classes === 2) {
        const p1 = 1 / (1 + Math.exp(-z.get(i, 0)));
        out.set(i, 0, 1 - p1);
        out.set(i, 1, p1);
      } else {
        let max = -Infinity;
        for (let c = 0; c < this.classes; c++) max = Math.max(max, z.get(i, c));
        let s = 0;
        for (let c = 0; c < this.classes; c++) {
          const e = Math.exp(z.get(i, c) - max);
          out.set(i, c, e);
          s += e;
        }
        for (let c = 0; c < this.classes; c++) out.set(i, c, out.get(i, c) / s);
      }
    }
    return out;
  }

  predict(x: Matrix): Int32Array {
    const p = this.predictProba(x);
    return Int32Array.from({ length: x.rows }, (_, i) => {
      let best = 0;
      for (let c = 1; c < this.classes; c++) if (p.get(i, c) > p.get(i, best)) best = c;
      return best;
    });
  }
}

/** Mean cross-entropy of a binary logistic model `σ(w·x + b)`, plus an optional ridge term. */
export function binaryLogLoss(
  x: Matrix,
  y: ArrayLike<number>,
  w: ArrayLike<number>,
  b: number,
  l2 = 0,
): number {
  let s = 0;
  for (let i = 0; i < x.rows; i++) {
    let z = b;
    for (let j = 0; j < x.cols; j++) z += x.get(i, j) * (w[j] as number);
    // log(1 + e^z) − y·z, evaluated without overflow
    s += Math.max(z, 0) + Math.log1p(Math.exp(-Math.abs(z))) - (y[i] as number) * z;
  }
  let reg = 0;
  for (let j = 0; j < x.cols; j++) reg += (w[j] as number) ** 2;
  return s / x.rows + 0.5 * l2 * reg;
}

export interface LogisticStep {
  step: number;
  weights: number[];
  intercept: number;
  loss: number;
}

/**
 * Batch gradient descent on the (optionally ridge-penalised) mean log-loss, yielding every step.
 * It heads for the same optimum that IRLS reaches in a handful of Newton steps.
 */
export function* logisticGradientDescent(
  x: Matrix,
  y: ArrayLike<number>,
  options: {
    lr: number;
    steps: number;
    l2?: number;
    start?: { weights: number[]; intercept: number };
  },
): Generator<LogisticStep, void, undefined> {
  const n = x.rows;
  const d = x.cols;
  const l2 = options.l2 ?? 0;
  let w = options.start?.weights.slice() ?? new Array<number>(d).fill(0);
  let b = options.start?.intercept ?? 0;
  yield { step: 0, weights: w.slice(), intercept: b, loss: binaryLogLoss(x, y, w, b, l2) };
  for (let t = 1; t <= options.steps; t++) {
    const gw = new Array<number>(d).fill(0);
    let gb = 0;
    for (let i = 0; i < n; i++) {
      let z = b;
      for (let j = 0; j < d; j++) z += x.get(i, j) * (w[j] as number);
      const r = 1 / (1 + Math.exp(-z)) - (y[i] as number);
      gb += r / n;
      for (let j = 0; j < d; j++) gw[j] = (gw[j] as number) + (r * x.get(i, j)) / n;
    }
    w = w.map((v, j) => v - options.lr * ((gw[j] as number) + l2 * v));
    b -= options.lr * gb;
    yield { step: t, weights: w.slice(), intercept: b, loss: binaryLogLoss(x, y, w, b, l2) };
  }
}
