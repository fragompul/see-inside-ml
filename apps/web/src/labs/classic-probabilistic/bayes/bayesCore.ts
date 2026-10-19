import {
  betaBinomialPmf,
  betaBinomialUpdate,
  betaDist,
  credibleInterval,
  gammaDist,
  gammaPoissonUpdate,
  normalDist,
  normalMeanUpdate,
  type Dist1D,
} from '@sim/classic';

export type BayesModel = 'coin' | 'mean' | 'rate';

/** Known measurement spread for the "mean" model. */
export const SIGMA = 2;

export interface BayesScenario {
  model: BayesModel;
  /** Prior belief about the unknown (a fraction, a mean, or a rate, depending on the model). */
  priorCenter: Record<BayesModel, number>;
  /** How many observations the prior is worth. */
  priorStrength: number;
  /** Number of observations. */
  n: number;
  /** The data's own estimate: fraction of heads, sample mean, or mean count. */
  observed: Record<BayesModel, number>;
}

export const DEFAULT_BAYES: BayesScenario = {
  model: 'coin',
  priorCenter: { coin: 0.5, mean: 0, rate: 1 },
  priorStrength: 2,
  n: 10,
  observed: { coin: 0.7, mean: 6, rate: 3 },
};

export const MODEL_INFO: Record<
  BayesModel,
  { label: string; unknown: string; blurb: string; symbol: string }
> = {
  coin: {
    label: 'Is the coin fair?',
    unknown: 'chance of heads',
    symbol: 'θ',
    blurb: 'Unknown: the probability of heads. Data: flips. Prior: a Beta distribution.',
  },
  mean: {
    label: 'What is the true value?',
    unknown: 'true value',
    symbol: 'μ',
    blurb: `Unknown: a quantity measured with noise of known size (σ = ${SIGMA}). Data: repeated measurements. Prior: a normal distribution.`,
  },
  rate: {
    label: 'How often does it happen?',
    unknown: 'events per interval',
    symbol: 'λ',
    blurb:
      'Unknown: the average number of events per interval. Data: counts. Prior: a Gamma distribution.',
  },
};

export interface BayesRun {
  scenario: BayesScenario;
  prior: Dist1D;
  posterior: Dist1D;
  /** Best estimate from the data alone (maximum likelihood). */
  mle: number;
  /** Most probable value under the posterior. */
  map: number;
  interval: [number, number];
  /** Share of the final belief that comes from the prior: strength / (strength + n). */
  priorWeight: number;
  /** x grid and unit-area curves for plotting. */
  grid: number[];
  priorCurve: number[];
  likelihoodCurve: number[];
  posteriorCurve: number[];
  /** Coin only: probability of k heads in the next 10 flips. */
  predictive: number[] | null;
}

export function runBayes(s: BayesScenario): BayesRun {
  const m = s.model;
  const c = s.priorCenter[m];
  const obs = s.observed[m];
  const n = s.n;
  let prior: Dist1D;
  let posterior: Dist1D;
  let likelihood: (x: number) => number;
  let predictive: number[] | null = null;
  if (m === 'coin') {
    const heads = Math.round(obs * n);
    prior = betaDist(c * s.priorStrength, (1 - c) * s.priorStrength);
    const post = betaBinomialUpdate(
      { a: c * s.priorStrength, b: (1 - c) * s.priorStrength },
      heads,
      n - heads,
    );
    posterior = betaDist(post.a, post.b);
    likelihood = (x) =>
      x <= 0 || x >= 1 ? 0 : Math.exp(heads * Math.log(x) + (n - heads) * Math.log(1 - x));
    predictive = Array.from({ length: 11 }, (_, k) => betaBinomialPmf(post.a, post.b, 10, k));
  } else if (m === 'mean') {
    prior = normalDist(c, SIGMA / Math.sqrt(s.priorStrength));
    const post = normalMeanUpdate({ mu: c, sd: SIGMA / Math.sqrt(s.priorStrength) }, n, obs, SIGMA);
    posterior = normalDist(post.mu, post.sd);
    likelihood = (x) => Math.exp((-n * (x - obs) ** 2) / (2 * SIGMA ** 2));
  } else {
    prior = gammaDist(c * s.priorStrength, s.priorStrength);
    const total = Math.round(obs * n);
    const post = gammaPoissonUpdate(
      { shape: c * s.priorStrength, rate: s.priorStrength },
      n,
      total,
    );
    posterior = gammaDist(post.shape, post.rate);
    likelihood = (x) => (x <= 0 ? 0 : Math.exp(total * Math.log(x) - n * x));
  }
  // plotting range: wide enough for all three curves
  const lo = Math.min(prior.support[0], posterior.support[0], m === 'coin' ? 0 : Infinity);
  const hi = Math.max(prior.support[1], posterior.support[1]);
  const spanLo = m === 'mean' ? Math.min(lo, obs - 4 * (SIGMA / Math.sqrt(Math.max(n, 1)))) : lo;
  const spanHi = m === 'mean' ? Math.max(hi, obs + 4 * (SIGMA / Math.sqrt(Math.max(n, 1)))) : hi;
  const grid = Array.from({ length: 300 }, (_, i) => spanLo + ((spanHi - spanLo) * i) / 299);
  const dx = (spanHi - spanLo) / 299;
  const rawLik = grid.map(likelihood);
  const area = rawLik.reduce((a, b) => a + b, 0) * dx || 1;
  const finite = (v: number) => (Number.isFinite(v) ? v : 0);
  return {
    scenario: s,
    prior,
    posterior,
    mle: m === 'coin' ? Math.round(obs * n) / Math.max(n, 1) : obs,
    map: posterior.mode,
    interval: credibleInterval(posterior, 0.95),
    priorWeight: s.priorStrength / (s.priorStrength + n),
    grid,
    priorCurve: grid.map((x) => finite(prior.pdf(x))),
    likelihoodCurve: rawLik.map((v) => v / area),
    posteriorCurve: grid.map((x) => finite(posterior.pdf(x))),
    predictive,
  };
}

export const intervalWidth = (r: BayesRun): number => r.interval[1] - r.interval[0];
