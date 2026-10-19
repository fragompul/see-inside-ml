import { Matrix, inverse, lgamma, normalCdf, solve, type Regressor } from '@sim/core';

/**
 * Generalised linear models fitted by iteratively reweighted least squares (IRLS):
 * `g(E[y]) = Xβ` with `y` from an exponential family. Each iteration solves a weighted least-squares
 * problem on a "working response", which is Newton's method on the log-likelihood for the canonical
 * links. Conventions follow statsmodels' `GLM`.
 */
export type GlmFamily = 'gaussian' | 'binomial' | 'poisson' | 'gamma';

interface FamilyDef {
  /** Inverse link `μ = g⁻¹(η)`. */
  mean: (eta: number) => number;
  /** Link `η = g(μ)`. */
  link: (mu: number) => number;
  /** Derivative of the link `g'(μ)`. */
  dLink: (mu: number) => number;
  variance: (mu: number) => number;
  /** Unit deviance of one observation. */
  unitDeviance: (y: number, mu: number) => number;
  start: (y: number, mean: number) => number;
  /** Does the dispersion φ have to be estimated (otherwise it is 1)? */
  estimatesScale: boolean;
}

const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v));
const xlogy = (a: number, b: number) => (a === 0 ? 0 : a * Math.log(b));

const FAMILIES: Record<GlmFamily, FamilyDef> = {
  gaussian: {
    mean: (e) => e,
    link: (m) => m,
    dLink: () => 1,
    variance: () => 1,
    unitDeviance: (y, m) => (y - m) ** 2,
    start: (y) => y,
    estimatesScale: true,
  },
  binomial: {
    mean: (e) => 1 / (1 + Math.exp(-clamp(e, -30, 30))),
    link: (m) => Math.log(m / (1 - m)),
    dLink: (m) => 1 / (m * (1 - m)),
    variance: (m) => m * (1 - m),
    unitDeviance: (y, m) => 2 * (xlogy(y, y / m) + xlogy(1 - y, (1 - y) / (1 - m))),
    start: (y) => (y + 0.5) / 2,
    estimatesScale: false,
  },
  poisson: {
    mean: (e) => Math.exp(clamp(e, -30, 30)),
    link: (m) => Math.log(m),
    dLink: (m) => 1 / m,
    variance: (m) => m,
    unitDeviance: (y, m) => 2 * (xlogy(y, y / m) - (y - m)),
    start: (y, mean) => (y + mean) / 2,
    estimatesScale: false,
  },
  gamma: {
    // log link (the canonical inverse link can give negative means)
    mean: (e) => Math.exp(clamp(e, -30, 30)),
    link: (m) => Math.log(m),
    dLink: (m) => 1 / m,
    variance: (m) => m * m,
    unitDeviance: (y, m) => 2 * ((y - m) / m - Math.log(y / m)),
    start: (y) => y,
    estimatesScale: true,
  },
};

export interface GlmOptions {
  family: GlmFamily;
  intercept?: boolean;
  /** Ridge penalty on the coefficients (never the intercept). */
  l2?: number;
  maxIterations?: number;
  tolerance?: number;
}

export interface GlmSummary {
  names: string[];
  estimates: number[];
  stdErrors: number[];
  z: number[];
  p: number[];
  deviance: number;
  nullDeviance: number;
  /** Pearson χ². */
  pearson: number;
  dispersion: number;
  logLikelihood: number | null;
  aic: number | null;
  n: number;
  dfResidual: number;
}

export class Glm implements Regressor {
  intercept = 0;
  coef: Float64Array = new Float64Array(0);
  iterations = 0;
  converged = false;
  private fitted: { x: Matrix; y: Float64Array; mu: Float64Array; w: Float64Array } | null = null;

  constructor(readonly options: GlmOptions) {}

  private get fam(): FamilyDef {
    return FAMILIES[this.options.family];
  }

  private design(x: Matrix): Matrix {
    if (this.options.intercept === false) return x;
    const d = new Matrix(x.rows, x.cols + 1);
    for (let i = 0; i < x.rows; i++) {
      d.set(i, 0, 1);
      for (let j = 0; j < x.cols; j++) d.set(i, j + 1, x.get(i, j));
    }
    return d;
  }

  fit(x: Matrix, y: ArrayLike<number>): this {
    const { maxIterations = 100, tolerance = 1e-12, l2 = 0 } = this.options;
    const fam = this.fam;
    const n = x.rows;
    const a = this.design(x);
    const p = a.cols;
    const yv = Float64Array.from(y);
    const ybar = yv.reduce((s, v) => s + v, 0) / n;
    let mu = Float64Array.from(yv, (v) => fam.start(v, ybar));
    let eta = Float64Array.from(mu, (m) => fam.link(m));
    let beta = new Float64Array(p);
    let w: Float64Array;
    let lastDeviance = Infinity;
    this.converged = false;
    let it = 0;
    for (; it < maxIterations; it++) {
      // working weights and response
      w = Float64Array.from(mu, (m) => 1 / (fam.variance(m) * fam.dLink(m) ** 2));
      const z = Float64Array.from(
        eta,
        (e, i) => e + ((yv[i] as number) - (mu[i] as number)) * fam.dLink(mu[i] as number),
      );
      const g = new Matrix(p, p);
      const b = new Matrix(p, 1);
      for (let i = 0; i < n; i++) {
        const wi = w[i] as number;
        for (let r = 0; r < p; r++) {
          const ar = a.get(i, r) * wi;
          b.data[r] = (b.data[r] as number) + ar * (z[i] as number);
          for (let c = 0; c <= r; c++)
            g.data[r * p + c] = (g.data[r * p + c] as number) + ar * a.get(i, c);
        }
      }
      for (let r = 0; r < p; r++) for (let c = r + 1; c < p; c++) g.set(r, c, g.get(c, r));
      if (l2 > 0)
        for (let j = this.options.intercept === false ? 0 : 1; j < p; j++)
          g.set(j, j, g.get(j, j) + l2);
      beta = Float64Array.from(solve(g, b).data);
      eta = Float64Array.from({ length: n }, (_, i) => {
        let s = 0;
        for (let j = 0; j < p; j++) s += a.get(i, j) * (beta[j] as number);
        return s;
      });
      mu = Float64Array.from(eta, (e) => fam.mean(e));
      const deviance = yv.reduce((s, v, i) => s + fam.unitDeviance(v, mu[i] as number), 0);
      if (Math.abs(deviance - lastDeviance) <= tolerance * (Math.abs(deviance) + 0.1)) {
        this.converged = true;
        it++;
        break;
      }
      lastDeviance = deviance;
    }
    this.iterations = it;
    w = Float64Array.from(mu, (m) => 1 / (fam.variance(m) * fam.dLink(m) ** 2));
    this.intercept = this.options.intercept === false ? 0 : (beta[0] as number);
    this.coef = Float64Array.from(beta.subarray(this.options.intercept === false ? 0 : 1));
    this.fitted = { x, y: yv, mu, w };
    return this;
  }

  /** Predicted mean `μ = g⁻¹(Xβ)`. */
  predict(x: Matrix): Float64Array {
    return Float64Array.from({ length: x.rows }, (_, i) => {
      let s = this.intercept;
      for (let j = 0; j < x.cols; j++) s += x.get(i, j) * (this.coef[j] as number);
      return this.fam.mean(s);
    });
  }

  /** Linear predictor `η = Xβ` (the log-odds for the binomial family). */
  linearPredictor(x: Matrix): Float64Array {
    return Float64Array.from({ length: x.rows }, (_, i) => {
      let s = this.intercept;
      for (let j = 0; j < x.cols; j++) s += x.get(i, j) * (this.coef[j] as number);
      return s;
    });
  }

  summary(names?: string[]): GlmSummary {
    if (!this.fitted) throw new Error('Call fit() first');
    const { x, y, mu, w } = this.fitted;
    const fam = this.fam;
    const n = x.rows;
    const a = this.design(x);
    const p = a.cols;
    const useIntercept = this.options.intercept !== false;
    const deviance = y.reduce((s, v, i) => s + fam.unitDeviance(v, mu[i] as number), 0);
    const ybar = y.reduce((s, v) => s + v, 0) / n;
    const nullMean = useIntercept ? ybar : fam.mean(0);
    const nullDeviance = y.reduce((s, v) => s + fam.unitDeviance(v, nullMean), 0);
    const pearson = y.reduce(
      (s, v, i) => s + (v - (mu[i] as number)) ** 2 / fam.variance(mu[i] as number),
      0,
    );
    const dfResidual = n - p;
    const dispersion = fam.estimatesScale ? pearson / dfResidual : 1;
    const g = new Matrix(p, p);
    for (let i = 0; i < n; i++)
      for (let r = 0; r < p; r++)
        for (let c = 0; c < p; c++)
          g.data[r * p + c] =
            (g.data[r * p + c] as number) + (w[i] as number) * a.get(i, r) * a.get(i, c);
    const cov = inverse(g);
    const estimates = useIntercept ? [this.intercept, ...this.coef] : Array.from(this.coef);
    const stdErrors = estimates.map((_, j) => Math.sqrt(dispersion * cov.get(j, j)));
    const z = estimates.map((b, j) => b / (stdErrors[j] as number));
    const labels = useIntercept
      ? ['intercept', ...(names ?? Array.from(this.coef, (_, j) => `x${j + 1}`))]
      : (names ?? Array.from(this.coef, (_, j) => `x${j + 1}`));
    const logLikelihood = this.logLikelihood(y, mu, deviance);
    return {
      names: labels,
      estimates,
      stdErrors,
      z,
      p: z.map((v) => 2 * (1 - normalCdf(Math.abs(v)))),
      deviance,
      nullDeviance,
      pearson,
      dispersion,
      logLikelihood,
      aic: logLikelihood === null ? null : -2 * logLikelihood + 2 * p,
      n,
      dfResidual,
    };
  }

  private logLikelihood(y: Float64Array, mu: Float64Array, deviance: number): number | null {
    const n = y.length;
    switch (this.options.family) {
      case 'gaussian': {
        const sigma2 = deviance / n;
        return -(n / 2) * (Math.log(2 * Math.PI * sigma2) + 1);
      }
      case 'binomial':
        return y.reduce(
          (s, v, i) => s + xlogy(v, mu[i] as number) + xlogy(1 - v, 1 - (mu[i] as number)),
          0,
        );
      case 'poisson':
        return y.reduce(
          (s, v, i) => s + xlogy(v, mu[i] as number) - (mu[i] as number) - lgamma(v + 1),
          0,
        );
      default:
        return null;
    }
  }
}

/** Convenience: the logistic-regression link and its inverse. */
export const logit = (p: number): number => Math.log(p / (1 - p));
export const inverseLogit = (z: number): number => 1 / (1 + Math.exp(-z));
