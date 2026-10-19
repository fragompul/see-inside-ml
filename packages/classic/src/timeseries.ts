import { Matrix, nelderMead, solve } from '@sim/core';

/**
 * Time-series toolkit: autocorrelation, differencing, classical decomposition, exponential
 * smoothing (simple, Holt, Holt-Winters) and ARIMA fitted by conditional sum of squares.
 */

// ---- basics ------------------------------------------------------------------------------------

const mean = (x: readonly number[]): number => x.reduce((a, b) => a + b, 0) / x.length;

/** Sample autocorrelation up to `maxLag` (biased estimator, as in statsmodels' `acf`). */
export function acf(x: readonly number[], maxLag: number): number[] {
  const n = x.length;
  const m = mean(x);
  let c0 = 0;
  for (const v of x) c0 += (v - m) ** 2;
  const out: number[] = [];
  for (let k = 0; k <= Math.min(maxLag, n - 1); k++) {
    let s = 0;
    for (let i = 0; i + k < n; i++) s += ((x[i] as number) - m) * ((x[i + k] as number) - m);
    out.push(c0 > 0 ? s / c0 : 0);
  }
  return out;
}

/** Partial autocorrelation by the Durbin-Levinson recursion on the sample autocorrelations. */
export function pacf(x: readonly number[], maxLag: number): number[] {
  const rho = acf(x, maxLag);
  const out = [1];
  let phi: number[] = [];
  for (let k = 1; k < rho.length; k++) {
    let num = rho[k] as number;
    let den = 1;
    for (let j = 0; j < phi.length; j++) {
      num -= (phi[j] as number) * (rho[k - 1 - j] as number);
      den -= (phi[j] as number) * (rho[j + 1] as number);
    }
    const kk = den === 0 ? 0 : num / den;
    phi = [...phi.map((p, j) => p - kk * (phi[phi.length - 1 - j] as number)), kk];
    out.push(kk);
  }
  return out;
}

/** Approximate 95% band under the null of white noise. */
export const whiteNoiseBand = (n: number): number => 1.96 / Math.sqrt(n);

/** Ljung-Box statistic for the first `lags` autocorrelations (compare with chi-square on `lags` degrees of freedom). */
export function ljungBox(x: readonly number[], lags: number): number {
  const n = x.length;
  const r = acf(x, lags);
  let q = 0;
  for (let k = 1; k <= lags; k++) q += (r[k] as number) ** 2 / (n - k);
  return n * (n + 2) * q;
}

/** `d`-th order differences at `lag` (lag 12 removes a yearly pattern in monthly data). */
export function difference(x: readonly number[], d = 1, lag = 1): number[] {
  let out = [...x];
  for (let i = 0; i < d; i++) out = out.slice(lag).map((v, j) => v - (out[j] as number));
  return out;
}

/** Undoes one differencing step given the values that preceded the differenced series. */
export function integrate(
  diffs: readonly number[],
  previous: readonly number[],
  lag = 1,
): number[] {
  const out = [...previous.slice(-lag)];
  diffs.forEach((d, i) => out.push((out[i] as number) + d));
  return out.slice(lag);
}

// ---- classical decomposition -----------------------------------------------------------------------

export interface Decomposition {
  trend: (number | null)[];
  seasonal: number[];
  residual: (number | null)[];
  /** One value per position of the cycle. */
  pattern: number[];
}

/**
 * Classical decomposition (`seasonal_decompose` in statsmodels): trend by a centred moving average
 * over one period, seasonal pattern as the average detrended value per position (re-centred), and
 * what is left as the residual. Additive: `x = trend + seasonal + residual`; multiplicative: product.
 */
export function decompose(
  x: readonly number[],
  period: number,
  model: 'additive' | 'multiplicative' = 'additive',
): Decomposition {
  const n = x.length;
  const half = Math.floor(period / 2);
  const trend: (number | null)[] = new Array<number | null>(n).fill(null);
  for (let i = half; i < n - half; i++) {
    if (period % 2 === 1) {
      let s = 0;
      for (let j = -half; j <= half; j++) s += x[i + j] as number;
      trend[i] = s / period;
    } else {
      // 2 x period moving average: half weights on the two end points
      let s = 0.5 * ((x[i - half] as number) + (x[i + half] as number));
      for (let j = -half + 1; j <= half - 1; j++) s += x[i + j] as number;
      trend[i] = s / period;
    }
  }
  const detrended = x.map((v, i) => {
    const t = trend[i];
    return t === null || t === undefined ? null : model === 'additive' ? v - t : v / t;
  });
  const pattern = new Array<number>(period).fill(0);
  for (let p = 0; p < period; p++) {
    const vals: number[] = [];
    for (let i = p; i < n; i += period) {
      const d = detrended[i];
      if (d !== null && d !== undefined) vals.push(d);
    }
    pattern[p] = vals.length ? mean(vals) : 0;
  }
  const centre = mean(pattern);
  const pat = pattern.map((v) => (model === 'additive' ? v - centre : v / centre));
  const seasonal = x.map((_, i) => pat[i % period] as number);
  const residual = x.map((v, i) => {
    const t = trend[i];
    if (t === null || t === undefined) return null;
    return model === 'additive'
      ? v - t - (seasonal[i] as number)
      : v / (t * (seasonal[i] as number));
  });
  return { trend, seasonal, residual, pattern: pat };
}

// ---- exponential smoothing -----------------------------------------------------------------------

export type EtsKind = 'simple' | 'holt' | 'holt-winters';

export interface EtsParams {
  alpha: number;
  beta?: number;
  gamma?: number;
  /** Season length for Holt-Winters. */
  period?: number;
  seasonal?: 'additive' | 'multiplicative';
}

export interface EtsFit {
  kind: EtsKind;
  params: EtsParams;
  /** One-step-ahead fitted values for every observation. */
  fitted: number[];
  sse: number;
  level: number;
  trend: number;
  season: number[];
  forecast: (h: number) => number[];
}

/** Runs the smoothing recursions with fixed parameters. */
export function etsRun(x: readonly number[], kind: EtsKind, p: EtsParams): EtsFit {
  const n = x.length;
  const period = p.period ?? 12;
  const mult = p.seasonal === 'multiplicative';
  let level: number;
  let trend: number;
  let season: number[] = [];
  if (kind === 'holt-winters') {
    const first = x.slice(0, period);
    const second = x.slice(period, 2 * period);
    level = mean(first);
    trend = second.length === period ? (mean(second) - mean(first)) / period : 0;
    season = first.map((v) => (mult ? v / level : v - level));
  } else {
    level = x[0] as number;
    trend = kind === 'holt' ? (x[1] as number) - (x[0] as number) : 0;
  }
  const fitted: number[] = [];
  let sse = 0;
  for (let t = 0; t < n; t++) {
    const s = kind === 'holt-winters' ? (season[t % period] as number) : mult ? 1 : 0;
    const base = level + trend;
    const pred = kind === 'holt-winters' ? (mult ? base * s : base + s) : base;
    fitted.push(pred);
    const y = x[t] as number;
    sse += (y - pred) ** 2;
    const prevLevel = level;
    if (kind === 'holt-winters') {
      level = p.alpha * (mult ? y / s : y - s) + (1 - p.alpha) * base;
      trend = (p.beta ?? 0) * (level - prevLevel) + (1 - (p.beta ?? 0)) * trend;
      // Hyndman's form: the seasonal update compares the observation with the *prediction* of level + trend
      season[t % period] = (p.gamma ?? 0) * (mult ? y / base : y - base) + (1 - (p.gamma ?? 0)) * s;
    } else {
      level = p.alpha * y + (1 - p.alpha) * base;
      if (kind === 'holt')
        trend = (p.beta ?? 0) * (level - prevLevel) + (1 - (p.beta ?? 0)) * trend;
    }
  }
  const finalLevel = level;
  const finalTrend = trend;
  const finalSeason = [...season];
  return {
    kind,
    params: p,
    fitted,
    sse,
    level: finalLevel,
    trend: finalTrend,
    season: finalSeason,
    forecast: (h) =>
      Array.from({ length: h }, (_, i) => {
        const base = finalLevel + (i + 1) * finalTrend;
        if (kind !== 'holt-winters') return base;
        const s = finalSeason[(n + i) % period] as number;
        return mult ? base * s : base + s;
      }),
  };
}

const logistic = (z: number) => 1 / (1 + Math.exp(-z));

/** Chooses the smoothing parameters that minimise the one-step squared error. */
export function etsFit(
  x: readonly number[],
  kind: EtsKind,
  options: { period?: number; seasonal?: 'additive' | 'multiplicative' } = {},
): EtsFit {
  const count = kind === 'simple' ? 1 : kind === 'holt' ? 2 : 3;
  const build = (z: readonly number[]): EtsParams => ({
    alpha: logistic(z[0] as number),
    ...(count > 1 ? { beta: logistic(z[1] as number) * 0.5 } : {}),
    ...(count > 2 ? { gamma: logistic(z[2] as number) } : {}),
    ...(options.period !== undefined ? { period: options.period } : {}),
    ...(options.seasonal !== undefined ? { seasonal: options.seasonal } : {}),
  });
  const result = nelderMead(
    (z) => etsRun(x, kind, build(z)).sse,
    new Array<number>(count).fill(-0.5),
    {
      step: 1,
      maxIterations: 400,
    },
  );
  return etsRun(x, kind, build(result.x));
}

// ---- ARIMA by conditional sum of squares -------------------------------------------------------------

export interface ArimaOrder {
  p: number;
  d: number;
  q: number;
  /** Seasonal differences (1 - B^period) applied before the ordinary ones. */
  D?: number;
  /** One seasonal moving-average term (1 + Theta B^period): the second half of the classic airline model. */
  Q?: number;
  period?: number;
}

export interface ArimaFit {
  order: ArimaOrder;
  /** Autoregressive coefficients phi_1..phi_p. */
  phi: number[];
  /** Moving-average coefficients theta_1..theta_q. */
  theta: number[];
  /** Seasonal moving-average coefficient (0 when the model has none). */
  seasonalTheta: number;
  /** Mean of the differenced series. */
  mean: number;
  sigma2: number;
  /** In-sample one-step residuals of the differenced series (the first `p` are zero by construction). */
  residuals: number[];
  aic: number;
  /** The series the model was fitted to (differenced `d` times). */
  differenced: number[];
  /** Forecast `h` steps ahead with 95% prediction intervals on the original scale. */
  forecast: (h: number) => { mean: number[]; lower: number[]; upper: number[] };
}

function cssResiduals(
  w: readonly number[],
  mu: number,
  phi: readonly number[],
  theta: readonly number[],
): number[] {
  const n = w.length;
  const p = phi.length;
  const e = new Array<number>(n).fill(0);
  for (let t = p; t < n; t++) {
    let pred = mu;
    for (let i = 0; i < p; i++) pred += (phi[i] as number) * ((w[t - 1 - i] as number) - mu);
    for (let j = 0; j < theta.length; j++)
      if (t - 1 - j >= 0) pred += (theta[j] as number) * (e[t - 1 - j] as number);
    e[t] = (w[t] as number) - pred;
  }
  return e;
}

/** Weights psi_j of the MA(infinity) form of an ARMA process: how a shock at time 0 echoes j steps later. */
export function psiWeights(
  phi: readonly number[],
  theta: readonly number[],
  count: number,
): number[] {
  const psi = [1];
  for (let j = 1; j < count; j++) {
    let v = j <= theta.length ? (theta[j - 1] as number) : 0;
    for (let i = 0; i < Math.min(j, phi.length); i++)
      v += (phi[i] as number) * (psi[j - 1 - i] as number);
    psi.push(v);
  }
  return psi;
}

/** The MA polynomial (1 + sum theta_j B^j)(1 + Theta B^period) written as plain coefficients. */
function expandMovingAverage(theta: readonly number[], seasonal: number, period: number): number[] {
  if (seasonal === 0) return [...theta];
  const out = new Array<number>(theta.length + period).fill(0);
  theta.forEach((c, j) => {
    out[j] = (out[j] as number) + c;
    out[j + period] = (out[j + period] as number) + c * seasonal;
  });
  out[period - 1] = (out[period - 1] as number) + seasonal;
  return out;
}

/** Multiplies the AR polynomial 1 - sum(phi_j B^j) by (1 - B^lag); returns the new phi coefficients. */
function multiplyByDifference(phi: readonly number[], lag: number): number[] {
  const out = new Array<number>(Math.max(phi.length + lag, lag)).fill(0);
  phi.forEach((c, j) => {
    out[j] = (out[j] as number) + c;
    out[j + lag] = (out[j + lag] as number) - c;
  });
  out[lag - 1] = (out[lag - 1] as number) + 1;
  return out;
}

/**
 * ARIMA by conditional sum of squares, with optional seasonal differencing. The mean of the
 * differenced series is estimated when nothing is differenced (or when `drift` is set); with
 * differencing there is no drift by default, as in statsmodels.
 */
export function arimaFit(
  x: readonly number[],
  order: ArimaOrder,
  options: { drift?: boolean } = {},
): ArimaFit {
  const { p, d, q } = order;
  const seasonalD = order.D ?? 0;
  const s = order.period ?? 12;
  // the chain of differencing operations, outermost first: seasonal ones, then ordinary ones
  const lags: number[] = [...new Array<number>(seasonalD).fill(s), ...new Array<number>(d).fill(1)];
  const levels: number[][] = [[...x]];
  for (const lag of lags) levels.push(difference(levels[levels.length - 1] as number[], 1, lag));
  const w = levels[levels.length - 1] as number[];
  const n = w.length;
  const m0 = lags.length === 0 || options.drift ? mean(w) : 0;
  const seasonalQ = order.Q ?? 0;
  const nParams = p + q + seasonalQ;
  let phi: number[] = [];
  let theta: number[] = [];
  let seasonalTheta = 0;
  if (nParams > 0) {
    // start the search from the OLS fit of the AR part (a good guess for the whole model)
    const start = new Array<number>(nParams).fill(0);
    if (p > 0) {
      const ols = fitAutoRegression(w, p);
      for (let i = 0; i < p; i++) start[i] = Math.max(-0.95, Math.min(0.95, ols.phi[i] as number));
    }
    const result = nelderMead(
      (z) => {
        const e = cssResiduals(
          w,
          m0,
          z.slice(0, p),
          expandMovingAverage(z.slice(p, p + q), seasonalQ ? (z[p + q] as number) : 0, s),
        );
        let acc = 0;
        for (let t = p; t < n; t++) acc += (e[t] as number) ** 2;
        return acc;
      },
      start,
      { step: 0.2, maxIterations: 800, tolerance: 1e-12 },
    );
    phi = result.x.slice(0, p);
    theta = result.x.slice(p, p + q);
    seasonalTheta = seasonalQ ? (result.x[p + q] as number) : 0;
  }
  const mu = m0;
  const thetaFull = expandMovingAverage(theta, seasonalTheta, s);
  const residuals = cssResiduals(w, mu, phi, thetaFull);
  let sse = 0;
  for (let t = p; t < n; t++) sse += (residuals[t] as number) ** 2;
  const used = n - p;
  const sigma2 = sse / used;
  const aic = used * Math.log(sigma2) + 2 * (nParams + 1) + used * (Math.log(2 * Math.PI) + 1);

  const forecast = (h: number) => {
    // point forecasts of the differenced series
    const wExt = [...w];
    const eExt = [...residuals];
    for (let k = 0; k < h; k++) {
      const t = wExt.length;
      let pred = mu;
      for (let i = 0; i < p; i++) pred += (phi[i] as number) * ((wExt[t - 1 - i] as number) - mu);
      for (let j = 0; j < thetaFull.length; j++)
        if (t - 1 - j < eExt.length) pred += (thetaFull[j] as number) * (eExt[t - 1 - j] as number);
      wExt.push(pred);
      eExt.push(0);
    }
    let path = wExt.slice(n);
    // undo the differencing, innermost first: y_t = f_t + y_(t - lag)
    for (let k = lags.length - 1; k >= 0; k--) {
      const lag = lags[k] as number;
      const base = levels[k] as number[];
      const out: number[] = [];
      path.forEach((f, i) => {
        const back = i < lag ? (base[base.length - lag + i] as number) : (out[i - lag] as number);
        out.push(f + back);
      });
      path = out;
    }
    // forecast variance from the psi weights of the full operator (differences folded into the AR part)
    let poly: number[] = [...phi];
    for (const lag of lags) poly = multiplyByDifference(poly, lag);
    const psi = psiWeights(poly, thetaFull, h);
    const lower: number[] = [];
    const upper: number[] = [];
    let cum = 0;
    for (let k = 0; k < h; k++) {
      cum += (psi[k] as number) ** 2;
      const se = Math.sqrt(sigma2 * cum);
      lower.push((path[k] as number) - 1.96 * se);
      upper.push((path[k] as number) + 1.96 * se);
    }
    return { mean: path, lower, upper };
  };
  return {
    order,
    phi,
    theta,
    seasonalTheta,
    mean: mu,
    sigma2,
    residuals,
    aic,
    differenced: w,
    forecast,
  };
}

/** AR(p) by ordinary least squares with an intercept (the model of `statsmodels.tsa.ar_model.AutoReg`). */
export function fitAutoRegression(
  x: readonly number[],
  p: number,
): { intercept: number; phi: number[]; sigma2: number } {
  const n = x.length - p;
  const a = new Matrix(n, p + 1);
  const b = new Matrix(n, 1);
  for (let t = 0; t < n; t++) {
    a.set(t, 0, 1);
    for (let i = 0; i < p; i++) a.set(t, i + 1, x[p + t - 1 - i] as number);
    b.set(t, 0, x[p + t] as number);
  }
  // normal equations
  const ata = new Matrix(p + 1, p + 1);
  const atb = new Matrix(p + 1, 1);
  for (let i = 0; i <= p; i++) {
    for (let j = 0; j <= p; j++) {
      let s = 0;
      for (let t = 0; t < n; t++) s += a.get(t, i) * a.get(t, j);
      ata.set(i, j, s);
    }
    let s = 0;
    for (let t = 0; t < n; t++) s += a.get(t, i) * b.get(t, 0);
    atb.set(i, 0, s);
  }
  const beta = solve(ata, atb);
  let sse = 0;
  for (let t = 0; t < n; t++) {
    let pred = 0;
    for (let i = 0; i <= p; i++) pred += a.get(t, i) * beta.get(i, 0);
    sse += (b.get(t, 0) - pred) ** 2;
  }
  return {
    intercept: beta.get(0, 0),
    phi: Array.from({ length: p }, (_, i) => beta.get(i + 1, 0)),
    sigma2: sse / n,
  };
}

// ---- accuracy ---------------------------------------------------------------------------------------

export const mae = (actual: readonly number[], pred: readonly number[]): number =>
  actual.reduce((s, v, i) => s + Math.abs(v - (pred[i] as number)), 0) / actual.length;

export const rmse = (actual: readonly number[], pred: readonly number[]): number =>
  Math.sqrt(actual.reduce((s, v, i) => s + (v - (pred[i] as number)) ** 2, 0) / actual.length);

export const mape = (actual: readonly number[], pred: readonly number[]): number =>
  actual.reduce((s, v, i) => s + Math.abs((v - (pred[i] as number)) / v), 0) / actual.length;
