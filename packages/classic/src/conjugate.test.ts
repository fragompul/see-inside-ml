import { describe, expect, it } from 'vitest';
import {
  betaBinomialPmf,
  betaBinomialUpdate,
  betaDist,
  credibleInterval,
  gammaDist,
  gammaPoissonUpdate,
  normalDist,
  normalMeanUpdate,
} from './conjugate';

/** Trapezoid integral of a density, to check that it is a proper density with the claimed mean. */
function integrate(f: (x: number) => number, lo: number, hi: number, n = 20000): number {
  const h = (hi - lo) / n;
  let s = 0.5 * (f(lo) + f(hi));
  for (let i = 1; i < n; i++) s += f(lo + i * h);
  return s * h;
}

describe('distributions', () => {
  it.each([
    ['beta', betaDist(3, 7)],
    ['gamma', gammaDist(4, 2)],
    ['normal', normalDist(1.5, 0.7)],
  ] as const)('%s: the density integrates to 1 and has the stated mean and variance', (_, d) => {
    const [lo, hi] = d.support;
    const a = d.name.startsWith('Beta') ? 1e-6 : lo;
    const b = d.name.startsWith('Beta') ? 1 - 1e-6 : hi * 3;
    expect(integrate(d.pdf, a, d.name.startsWith('Gamma') ? 12 : b)).toBeCloseTo(1, 3);
    expect(integrate((x) => x * d.pdf(x), a, d.name.startsWith('Gamma') ? 12 : b)).toBeCloseTo(
      d.mean,
      3,
    );
    expect(
      integrate((x) => (x - d.mean) ** 2 * d.pdf(x), a, d.name.startsWith('Gamma') ? 12 : b),
    ).toBeCloseTo(d.variance, 3);
  });

  it('quantile inverts the cdf', () => {
    for (const d of [betaDist(2, 5), gammaDist(3, 1.5), normalDist(0, 2)])
      for (const p of [0.025, 0.5, 0.975]) expect(d.cdf(d.quantile(p))).toBeCloseTo(p, 8);
  });

  it('the 95% credible interval contains 95% of the mass', () => {
    const d = betaDist(8, 4);
    const [lo, hi] = credibleInterval(d, 0.95);
    expect(d.cdf(hi) - d.cdf(lo)).toBeCloseTo(0.95, 8);
  });
});

describe('conjugate updates', () => {
  it('Beta-binomial posterior has the closed-form mean', () => {
    const post = betaBinomialUpdate({ a: 1, b: 1 }, 7, 3);
    expect(post).toEqual({ a: 8, b: 4 });
    expect(betaDist(post.a, post.b).mean).toBeCloseTo(8 / 12, 12);
  });

  it('updating in two batches equals updating once with all the data', () => {
    const once = betaBinomialUpdate({ a: 2, b: 3 }, 9, 6);
    const first = betaBinomialUpdate({ a: 2, b: 3 }, 4, 1);
    const twice = betaBinomialUpdate(first, 5, 5);
    expect(twice).toEqual(once);
  });

  it('the posterior is a compromise: the mean lies between prior mean and sample mean', () => {
    const prior = { a: 8, b: 2 }; // mean 0.8
    const post = betaBinomialUpdate(prior, 3, 7); // data mean 0.3
    const m = post.a / (post.a + post.b);
    expect(m).toBeGreaterThan(0.3);
    expect(m).toBeLessThan(0.8);
  });

  it('normal-normal: precisions add and the mean is the precision-weighted average', () => {
    const post = normalMeanUpdate({ mu: 0, sd: 2 }, 4, 3, 1);
    // prior precision 0.25, data precision 4
    expect(1 / post.sd ** 2).toBeCloseTo(4.25, 12);
    expect(post.mu).toBeCloseTo((0.25 * 0 + 4 * 3) / 4.25, 12);
  });

  it('gamma-Poisson: shape adds the counts, rate adds the observation periods', () => {
    expect(gammaPoissonUpdate({ shape: 2, rate: 1 }, 5, 12)).toEqual({ shape: 14, rate: 6 });
  });

  it('the beta-binomial predictive distribution sums to 1 and matches a hand calculation', () => {
    let total = 0;
    for (let k = 0; k <= 10; k++) total += betaBinomialPmf(3, 2, 10, k);
    expect(total).toBeCloseTo(1, 10);
    // one future trial: P(success) = a / (a + b)
    expect(betaBinomialPmf(3, 2, 1, 1)).toBeCloseTo(3 / 5, 12);
  });
});
