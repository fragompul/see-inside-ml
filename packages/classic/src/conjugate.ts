import { betaInc, gammaInc, lgamma, normalCdf } from '@sim/core';

/**
 * Conjugate Bayesian updating: for some likelihood/prior pairs the posterior is a distribution of
 * the same family as the prior, with parameters that simply add up the data. That makes the whole
 * of Bayesian learning visible in closed form.
 *   Beta prior   + Bernoulli/binomial data   -> Beta posterior
 *   Normal prior + normal data (known sigma) -> Normal posterior
 *   Gamma prior  + Poisson counts            -> Gamma posterior
 */
export interface Dist1D {
  name: string;
  pdf(x: number): number;
  cdf(x: number): number;
  quantile(p: number): number;
  mean: number;
  variance: number;
  mode: number;
  /** A range that contains essentially all the mass, for plotting. */
  support: [number, number];
}

/** Inverts a monotone CDF by bisection. */
function invert(cdf: (x: number) => number, p: number, lo: number, hi: number): number {
  let a = lo;
  let b = hi;
  for (let i = 0; i < 100; i++) {
    const m = (a + b) / 2;
    if (cdf(m) < p) a = m;
    else b = m;
  }
  return (a + b) / 2;
}

export function betaDist(a: number, b: number): Dist1D {
  const lbeta = lgamma(a) + lgamma(b) - lgamma(a + b);
  const mean = a / (a + b);
  const variance = (a * b) / ((a + b) ** 2 * (a + b + 1));
  return {
    name: `Beta(${a.toFixed(1)}, ${b.toFixed(1)})`,
    pdf: (x) =>
      x <= 0 || x >= 1
        ? a < 1 && x <= 0
          ? Infinity
          : 0
        : Math.exp((a - 1) * Math.log(x) + (b - 1) * Math.log(1 - x) - lbeta),
    cdf: (x) => (x <= 0 ? 0 : x >= 1 ? 1 : betaInc(x, a, b)),
    quantile: (p) => invert((x) => (x <= 0 ? 0 : x >= 1 ? 1 : betaInc(x, a, b)), p, 0, 1),
    mean,
    variance,
    mode: a > 1 && b > 1 ? (a - 1) / (a + b - 2) : a <= 1 && b > 1 ? 0 : b <= 1 && a > 1 ? 1 : mean,
    support: [0, 1],
  };
}

export function gammaDist(shape: number, rate: number): Dist1D {
  const lnorm = shape * Math.log(rate) - lgamma(shape);
  const mean = shape / rate;
  const sd = Math.sqrt(shape) / rate;
  return {
    name: `Gamma(${shape.toFixed(1)}, ${rate.toFixed(1)})`,
    pdf: (x) => (x <= 0 ? 0 : Math.exp(lnorm + (shape - 1) * Math.log(x) - rate * x)),
    cdf: (x) => (x <= 0 ? 0 : gammaInc(shape, rate * x)),
    quantile: (p) =>
      invert((x) => (x <= 0 ? 0 : gammaInc(shape, rate * x)), p, 0, mean + 40 * sd + 10),
    mean,
    variance: shape / (rate * rate),
    mode: shape >= 1 ? (shape - 1) / rate : 0,
    support: [0, mean + 5 * sd],
  };
}

export function normalDist(mu: number, sd: number): Dist1D {
  return {
    name: `Normal(${mu.toFixed(1)}, ${sd.toFixed(2)})`,
    pdf: (x) => Math.exp(-0.5 * ((x - mu) / sd) ** 2) / (sd * Math.sqrt(2 * Math.PI)),
    cdf: (x) => normalCdf(x, mu, sd),
    quantile: (p) => invert((x) => normalCdf(x, mu, sd), p, mu - 12 * sd, mu + 12 * sd),
    mean: mu,
    variance: sd * sd,
    mode: mu,
    support: [mu - 4.5 * sd, mu + 4.5 * sd],
  };
}

/** Beta(a, b) prior plus `heads` successes and `tails` failures. */
export const betaBinomialUpdate = (
  prior: { a: number; b: number },
  heads: number,
  tails: number,
): { a: number; b: number } => ({ a: prior.a + heads, b: prior.b + tails });

/** Normal(mu0, tau0) prior on the mean of normal data with known standard deviation `sigma`. */
export function normalMeanUpdate(
  prior: { mu: number; sd: number },
  n: number,
  sampleMean: number,
  sigma: number,
): { mu: number; sd: number } {
  const precision0 = 1 / prior.sd ** 2;
  const precisionData = n / sigma ** 2;
  const precision = precision0 + precisionData;
  return {
    mu: (precision0 * prior.mu + precisionData * sampleMean) / precision,
    sd: Math.sqrt(1 / precision),
  };
}

/** Gamma(shape, rate) prior on a Poisson rate, after observing `n` counts that add up to `total`. */
export const gammaPoissonUpdate = (
  prior: { shape: number; rate: number },
  n: number,
  total: number,
): { shape: number; rate: number } => ({ shape: prior.shape + total, rate: prior.rate + n });

/** Equal-tailed credible interval. */
export function credibleInterval(dist: Dist1D, level = 0.95): [number, number] {
  const tail = (1 - level) / 2;
  return [dist.quantile(tail), dist.quantile(1 - tail)];
}

/** Probability of exactly `k` successes in `n` future trials under a Beta(a, b) belief (beta-binomial). */
export function betaBinomialPmf(a: number, b: number, n: number, k: number): number {
  const lchoose = lgamma(n + 1) - lgamma(k + 1) - lgamma(n - k + 1);
  const lbeta = (x: number, y: number) => lgamma(x) + lgamma(y) - lgamma(x + y);
  return Math.exp(lchoose + lbeta(a + k, b + n - k) - lbeta(a, b));
}
