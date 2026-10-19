import {
  Matrix,
  cholesky,
  choleskySolve,
  fCdf,
  gram,
  inverse,
  solve,
  tCdf,
  tPpf,
  type Regressor,
} from '@sim/core';

/**
 * Ordinary least squares (and its ridge-regularised cousin) by the normal equations, with the
 * statistical summary a statistics package prints: standard errors, t statistics, p-values,
 * confidence intervals, R², F test, AIC/BIC, leverage and Cook's distance.
 */
export interface LinearOptions {
  /** Fit an intercept (default true). */
  intercept?: boolean;
  /** Ridge penalty λ on the coefficients (never on the intercept): minimises ‖y − Xw‖² + λ‖w‖². */
  l2?: number;
}

export interface CoefficientRow {
  name: string;
  estimate: number;
  stdError: number;
  t: number;
  p: number;
  ci: [number, number];
}

export interface LinearSummary {
  n: number;
  /** Number of estimated parameters (including the intercept). */
  parameters: number;
  dfResidual: number;
  rss: number;
  tss: number;
  r2: number;
  adjustedR2: number;
  /** Estimated noise variance σ² = RSS / (n − p). */
  sigma2: number;
  fStatistic: number;
  fPValue: number;
  aic: number;
  bic: number;
  coefficients: CoefficientRow[];
}

export class LinearRegression implements Regressor {
  intercept = 0;
  coef: Float64Array = new Float64Array(0);
  private fitted: { x: Matrix; y: Float64Array } | null = null;

  constructor(readonly options: LinearOptions = {}) {}

  private get useIntercept(): boolean {
    return this.options.intercept !== false;
  }

  /** Design matrix: a leading column of ones when there is an intercept. */
  design(x: Matrix): Matrix {
    if (!this.useIntercept) return x;
    const d = new Matrix(x.rows, x.cols + 1);
    for (let i = 0; i < x.rows; i++) {
      d.set(i, 0, 1);
      for (let j = 0; j < x.cols; j++) d.set(i, j + 1, x.get(i, j));
    }
    return d;
  }

  fit(x: Matrix, y: ArrayLike<number>): this {
    const a = this.design(x);
    const g = gram(a);
    const lambda = this.options.l2 ?? 0;
    if (lambda > 0)
      for (let j = this.useIntercept ? 1 : 0; j < g.rows; j++) g.set(j, j, g.get(j, j) + lambda);
    const b = new Matrix(a.cols, 1);
    for (let i = 0; i < a.rows; i++)
      for (let j = 0; j < a.cols; j++)
        b.data[j] = (b.data[j] as number) + a.get(i, j) * (y[i] as number);
    let w: Matrix;
    try {
      w = choleskySolve(cholesky(g), b);
    } catch {
      w = solve(g, b); // singular up to rounding: LU still gives the (unstable) solution; the UI warns about collinearity
    }
    const wv = w.data;
    this.intercept = this.useIntercept ? (wv[0] as number) : 0;
    this.coef = Float64Array.from(wv.subarray(this.useIntercept ? 1 : 0));
    this.fitted = { x, y: Float64Array.from(y) };
    return this;
  }

  predict(x: Matrix): Float64Array {
    const out = new Float64Array(x.rows);
    for (let i = 0; i < x.rows; i++) {
      let s = this.intercept;
      for (let j = 0; j < x.cols; j++) s += x.get(i, j) * (this.coef[j] as number);
      out[i] = s;
    }
    return out;
  }

  residuals(): Float64Array {
    const f = this.requireFit();
    const pred = this.predict(f.x);
    return Float64Array.from(f.y, (v, i) => v - (pred[i] as number));
  }

  /** Leverage `hᵢᵢ` of every training point: how strongly it pulls the fit towards itself. */
  leverage(): Float64Array {
    const f = this.requireFit();
    const a = this.design(f.x);
    const inv = inverse(gram(a));
    const h = new Float64Array(a.rows);
    for (let i = 0; i < a.rows; i++) {
      let s = 0;
      for (let p = 0; p < a.cols; p++)
        for (let q = 0; q < a.cols; q++) s += a.get(i, p) * inv.get(p, q) * a.get(i, q);
      h[i] = s;
    }
    return h;
  }

  /** Cook's distance of every training point: how much the fit would move if it were removed. */
  cooksDistance(): Float64Array {
    const f = this.requireFit();
    const e = this.residuals();
    const h = this.leverage();
    const p = this.options.intercept === false ? f.x.cols : f.x.cols + 1;
    const rss = e.reduce((s, v) => s + v * v, 0);
    const s2 = rss / (f.x.rows - p);
    return Float64Array.from(
      e,
      (r, i) => ((r * r) / (p * s2)) * ((h[i] as number) / (1 - (h[i] as number)) ** 2),
    );
  }

  /** Variance inflation factor of each feature: how much collinearity inflates its coefficient's variance. */
  vif(): Float64Array {
    const f = this.requireFit();
    const d = f.x.cols;
    if (d < 2) return new Float64Array(d).fill(1);
    return Float64Array.from({ length: d }, (_, j) => {
      const others = new Matrix(f.x.rows, d - 1);
      for (let i = 0; i < f.x.rows; i++) {
        let c = 0;
        for (let k = 0; k < d; k++) if (k !== j) others.set(i, c++, f.x.get(i, k));
      }
      const target = Float64Array.from({ length: f.x.rows }, (_, i) => f.x.get(i, j));
      const aux = new LinearRegression().fit(others, target);
      const pred = aux.predict(others);
      let rss = 0;
      let tss = 0;
      const m = target.reduce((s, v) => s + v, 0) / target.length;
      for (let i = 0; i < target.length; i++) {
        rss += ((target[i] as number) - (pred[i] as number)) ** 2;
        tss += ((target[i] as number) - m) ** 2;
      }
      return 1 / Math.max(1e-12, rss / tss);
    });
  }

  /** The textbook regression table. Only meaningful for an unpenalised fit. */
  summary(names?: string[], level = 0.95): LinearSummary {
    const f = this.requireFit();
    const n = f.x.rows;
    const a = this.design(f.x);
    const p = a.cols;
    const e = this.residuals();
    const rss = e.reduce((s, v) => s + v * v, 0);
    const mean = this.useIntercept ? f.y.reduce((s, v) => s + v, 0) / n : 0;
    const tss = f.y.reduce((s, v) => s + (v - mean) ** 2, 0);
    const dfResidual = n - p;
    const sigma2 = rss / dfResidual;
    const inv = inverse(gram(a));
    const est = this.useIntercept ? [this.intercept, ...this.coef] : Array.from(this.coef);
    const labels = this.useIntercept
      ? ['intercept', ...(names ?? Array.from(this.coef, (_, j) => `x${j + 1}`))]
      : (names ?? Array.from(this.coef, (_, j) => `x${j + 1}`));
    const crit = tPpf(1 - (1 - level) / 2, dfResidual);
    const coefficients: CoefficientRow[] = est.map((b, j) => {
      const se = Math.sqrt(sigma2 * inv.get(j, j));
      const t = b / se;
      return {
        name: labels[j] as string,
        estimate: b,
        stdError: se,
        t,
        p: 2 * (1 - tCdf(Math.abs(t), dfResidual)),
        ci: [b - crit * se, b + crit * se],
      };
    });
    const dfModel = this.useIntercept ? p - 1 : p;
    const r2 = 1 - rss / tss;
    const fStatistic = (tss - rss) / dfModel / sigma2;
    const logLik = -(n / 2) * (Math.log(2 * Math.PI) + Math.log(rss / n) + 1);
    return {
      n,
      parameters: p,
      dfResidual,
      rss,
      tss,
      r2,
      adjustedR2: 1 - (1 - r2) * ((n - (this.useIntercept ? 1 : 0)) / dfResidual),
      sigma2,
      fStatistic,
      fPValue: 1 - fCdf(fStatistic, dfModel, dfResidual),
      aic: 2 * p - 2 * logLik, // statsmodels' convention: the noise variance is not counted
      bic: Math.log(n) * p - 2 * logLik,
      coefficients,
    };
  }

  private requireFit(): { x: Matrix; y: Float64Array } {
    if (!this.fitted) throw new Error('Call fit() first');
    return this.fitted;
  }
}

/** One recorded step of gradient descent on a line `y = a·x + b`. */
export interface LineStep {
  step: number;
  slope: number;
  intercept: number;
  loss: number;
}

/**
 * Fits `y = slope·x + intercept` by gradient descent on the mean squared error, yielding every
 * step so a lab can animate how the line settles on the least-squares solution. Inputs are
 * centred internally, which makes the descent well-behaved for any `x` range.
 */
export function* lineGradientDescent(
  x: ArrayLike<number>,
  y: ArrayLike<number>,
  options: { lr: number; steps: number; start?: { slope: number; intercept: number } },
): Generator<LineStep, void, undefined> {
  const n = x.length;
  let slope = options.start?.slope ?? 0;
  let intercept = options.start?.intercept ?? 0;
  const loss = () => {
    let s = 0;
    for (let i = 0; i < n; i++)
      s += ((y[i] as number) - (slope * (x[i] as number) + intercept)) ** 2;
    return s / n;
  };
  yield { step: 0, slope, intercept, loss: loss() };
  for (let t = 1; t <= options.steps; t++) {
    let gs = 0;
    let gi = 0;
    for (let i = 0; i < n; i++) {
      const err = slope * (x[i] as number) + intercept - (y[i] as number);
      gs += (2 / n) * err * (x[i] as number);
      gi += (2 / n) * err;
    }
    slope -= options.lr * gs;
    intercept -= options.lr * gi;
    yield { step: t, slope, intercept, loss: loss() };
  }
}
