import { Matrix } from './matrix';

/** Descriptive statistics and the distribution functions behind p-values and confidence intervals. */

export const mean = (v: ArrayLike<number>): number => {
  let s = 0;
  for (let i = 0; i < v.length; i++) s += v[i] as number;
  return s / v.length;
};

/** Variance with `ddof` degrees-of-freedom correction (0 = population, 1 = sample). */
export const variance = (v: ArrayLike<number>, ddof = 0): number => {
  const m = mean(v);
  let s = 0;
  for (let i = 0; i < v.length; i++) s += ((v[i] as number) - m) ** 2;
  return s / (v.length - ddof);
};

export const std = (v: ArrayLike<number>, ddof = 0): number => Math.sqrt(variance(v, ddof));

/** Linear-interpolated quantile (q in [0, 1]), like NumPy's default. */
export function quantile(v: ArrayLike<number>, q: number): number {
  const sorted = Array.from(v).sort((a, b) => a - b);
  if (sorted.length === 0) return NaN;
  const pos = Math.min(1, Math.max(0, q)) * (sorted.length - 1);
  const lo = Math.floor(pos);
  const hi = Math.ceil(pos);
  return (sorted[lo] as number) + ((sorted[hi] as number) - (sorted[lo] as number)) * (pos - lo);
}

export const median = (v: ArrayLike<number>): number => quantile(v, 0.5);

/** Column means of an `n × d` matrix. */
export function columnMeans(x: Matrix): Float64Array {
  const out = new Float64Array(x.cols);
  for (let i = 0; i < x.rows; i++)
    for (let j = 0; j < x.cols; j++) out[j] = (out[j] as number) + x.get(i, j) / x.rows;
  return out;
}

/** Covariance matrix of the columns (`ddof` as in `variance`). */
export function covariance(x: Matrix, ddof = 1): Matrix {
  const mu = columnMeans(x);
  const c = new Matrix(x.cols, x.cols);
  for (let i = 0; i < x.rows; i++)
    for (let a = 0; a < x.cols; a++) {
      const da = x.get(i, a) - (mu[a] as number);
      for (let b = a; b < x.cols; b++)
        c.data[a * x.cols + b] =
          (c.data[a * x.cols + b] as number) + da * (x.get(i, b) - (mu[b] as number));
    }
  for (let a = 0; a < x.cols; a++)
    for (let b = a; b < x.cols; b++) {
      const v = c.get(a, b) / (x.rows - ddof);
      c.set(a, b, v);
      c.set(b, a, v);
    }
  return c;
}

export function correlation(a: ArrayLike<number>, b: ArrayLike<number>): number {
  const ma = mean(a);
  const mb = mean(b);
  let sab = 0;
  let saa = 0;
  let sbb = 0;
  for (let i = 0; i < a.length; i++) {
    const da = (a[i] as number) - ma;
    const db = (b[i] as number) - mb;
    sab += da * db;
    saa += da * da;
    sbb += db * db;
  }
  return saa > 0 && sbb > 0 ? sab / Math.sqrt(saa * sbb) : 0;
}

// ---------------------------------------------------------------------------------------------
// Special functions

/** Natural log of the gamma function (Lanczos approximation, accurate to ~1e-14). */
export function lgamma(x: number): number {
  if (x < 0.5) return Math.log(Math.PI / Math.abs(Math.sin(Math.PI * x))) - lgamma(1 - x);
  const g = 7;
  const c = [
    0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313,
    -176.61502916214059, 12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6,
    1.5056327351493116e-7,
  ];
  const y = x - 1;
  let a = c[0] as number;
  const t = y + g + 0.5;
  for (let i = 1; i < g + 2; i++) a += (c[i] as number) / (y + i);
  return 0.5 * Math.log(2 * Math.PI) + (y + 0.5) * Math.log(t) - t + Math.log(a);
}

/** Regularised incomplete beta function `I_x(a, b)` (continued fraction, Numerical Recipes). */
export function betaInc(x: number, a: number, b: number): number {
  if (x <= 0) return 0;
  if (x >= 1) return 1;
  const front = Math.exp(
    lgamma(a + b) - lgamma(a) - lgamma(b) + a * Math.log(x) + b * Math.log(1 - x),
  );
  const fraction = (xx: number, aa: number, bb: number): number => {
    const tiny = 1e-300;
    let c = 1;
    let d = 1 - ((aa + bb) * xx) / (aa + 1);
    if (Math.abs(d) < tiny) d = tiny;
    d = 1 / d;
    let h = d;
    for (let m = 1; m <= 300; m++) {
      const m2 = 2 * m;
      let num = (m * (bb - m) * xx) / ((aa + m2 - 1) * (aa + m2));
      d = 1 + num * d;
      if (Math.abs(d) < tiny) d = tiny;
      c = 1 + num / c;
      if (Math.abs(c) < tiny) c = tiny;
      d = 1 / d;
      h *= d * c;
      num = (-(aa + m) * (aa + bb + m) * xx) / ((aa + m2) * (aa + m2 + 1));
      d = 1 + num * d;
      if (Math.abs(d) < tiny) d = tiny;
      c = 1 + num / c;
      if (Math.abs(c) < tiny) c = tiny;
      d = 1 / d;
      const delta = d * c;
      h *= delta;
      if (Math.abs(delta - 1) < 1e-14) break;
    }
    return h;
  };
  return x < (a + 1) / (a + b + 2)
    ? (front * fraction(x, a, b)) / a
    : 1 - (front * fraction(1 - x, b, a)) / b;
}

/** Regularised lower incomplete gamma function `P(a, x)`. */
export function gammaInc(a: number, x: number): number {
  if (x <= 0) return 0;
  if (x < a + 1) {
    let sum = 1 / a;
    let term = sum;
    for (let n = 1; n < 500; n++) {
      term *= x / (a + n);
      sum += term;
      if (Math.abs(term) < Math.abs(sum) * 1e-15) break;
    }
    return sum * Math.exp(-x + a * Math.log(x) - lgamma(a));
  }
  const tiny = 1e-300;
  let b = x + 1 - a;
  let c = 1 / tiny;
  let d = 1 / b;
  let h = d;
  for (let i = 1; i < 500; i++) {
    const an = -i * (i - a);
    b += 2;
    d = an * d + b;
    if (Math.abs(d) < tiny) d = tiny;
    c = b + an / c;
    if (Math.abs(c) < tiny) c = tiny;
    d = 1 / d;
    const delta = d * c;
    h *= delta;
    if (Math.abs(delta - 1) < 1e-15) break;
  }
  return 1 - Math.exp(-x + a * Math.log(x) - lgamma(a)) * h;
}

export const normalPdf = (x: number, mu = 0, sigma = 1): number =>
  Math.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * Math.sqrt(2 * Math.PI));

/** Standard normal CDF through the error function (via the incomplete gamma function). */
export function normalCdf(x: number, mu = 0, sigma = 1): number {
  const z = (x - mu) / sigma;
  const p = gammaInc(0.5, (z * z) / 2) / 2;
  return z >= 0 ? 0.5 + p : 0.5 - p;
}

/** Inverse of the normal CDF (Acklam's algorithm refined by one Newton step). */
export function normalPpf(p: number, mu = 0, sigma = 1): number {
  if (p <= 0) return -Infinity;
  if (p >= 1) return Infinity;
  const a = [
    -39.69683028665376, 220.9460984245205, -275.9285104469687, 138.357751867269, -30.66479806614716,
    2.506628277459239,
  ];
  const b = [
    -54.47609879822406, 161.5858368580409, -155.6989798598866, 66.80131188771972,
    -13.28068155288572,
  ];
  const c = [
    -0.007784894002430293, -0.3223964580411365, -2.400758277161838, -2.549732539343734,
    4.374664141464968, 2.938163982698783,
  ];
  const d = [0.007784695709041462, 0.3224671290700398, 2.445134137142996, 3.754408661907416];
  const lo = 0.02425;
  let x: number;
  if (p < lo) {
    const q = Math.sqrt(-2 * Math.log(p));
    x =
      (((((c[0]! * q + c[1]!) * q + c[2]!) * q + c[3]!) * q + c[4]!) * q + c[5]!) /
      ((((d[0]! * q + d[1]!) * q + d[2]!) * q + d[3]!) * q + 1);
  } else if (p > 1 - lo) {
    const q = Math.sqrt(-2 * Math.log(1 - p));
    x =
      -(((((c[0]! * q + c[1]!) * q + c[2]!) * q + c[3]!) * q + c[4]!) * q + c[5]!) /
      ((((d[0]! * q + d[1]!) * q + d[2]!) * q + d[3]!) * q + 1);
  } else {
    const q = p - 0.5;
    const r = q * q;
    x =
      ((((((a[0]! * r + a[1]!) * r + a[2]!) * r + a[3]!) * r + a[4]!) * r + a[5]!) * q) /
      (((((b[0]! * r + b[1]!) * r + b[2]!) * r + b[3]!) * r + b[4]!) * r + 1);
  }
  const e = normalCdf(x) - p;
  const u = e * Math.sqrt(2 * Math.PI) * Math.exp((x * x) / 2);
  x -= u / (1 + (x * u) / 2);
  return mu + sigma * x;
}

/** CDF of Student's t distribution with `df` degrees of freedom. */
export function tCdf(t: number, df: number): number {
  const x = df / (df + t * t);
  const tail = 0.5 * betaInc(x, df / 2, 0.5);
  return t > 0 ? 1 - tail : tail;
}

/** Quantile of Student's t (bisection on the CDF; `p` in (0, 1)). */
export function tPpf(p: number, df: number): number {
  let lo = -1e3;
  let hi = 1e3;
  for (let i = 0; i < 200; i++) {
    const mid = (lo + hi) / 2;
    if (tCdf(mid, df) < p) lo = mid;
    else hi = mid;
  }
  return (lo + hi) / 2;
}

/** CDF of the F distribution with `d1` and `d2` degrees of freedom. */
export function fCdf(f: number, d1: number, d2: number): number {
  if (f <= 0) return 0;
  return betaInc((d1 * f) / (d1 * f + d2), d1 / 2, d2 / 2);
}

/** CDF of the chi-squared distribution with `k` degrees of freedom. */
export const chi2Cdf = (x: number, k: number): number => gammaInc(k / 2, x / 2);
