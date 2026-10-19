import { betaInc, createRng, normalCdf } from '@sim/core';
import { describe, expect, it } from 'vitest';
import {
  autocorrelation,
  chainColumn,
  chainEss,
  chainMean,
  gibbsBivariateNormal,
  hamiltonian,
  ksStatistic,
  metropolis,
  rHat,
} from './mcmc';

const stdNormal = (x: readonly number[]) => -0.5 * (x[0] as number) ** 2;

describe('Metropolis', () => {
  it('samples a standard normal (KS test, corrected for autocorrelation)', () => {
    const chain = metropolis(stdNormal, [0], { steps: 20000, proposalSd: 2.4, seed: 1 });
    const x = chainColumn(chain, 0).slice(1000);
    const d = ksStatistic(x, (v) => normalCdf(v));
    // 1% critical value 1.63/sqrt(n_eff)
    expect(d).toBeLessThan(1.63 / Math.sqrt(chainEss(x)));
  });

  it('samples a Beta(3, 5) posterior known only up to a constant', () => {
    const logTarget = (x: readonly number[]) => {
      const t = x[0] as number;
      return t <= 0 || t >= 1 ? -Infinity : 2 * Math.log(t) + 4 * Math.log(1 - t);
    };
    const chain = metropolis(logTarget, [0.5], { steps: 30000, proposalSd: 0.25, seed: 2 });
    const x = chainColumn(chain, 0).slice(1000);
    expect(ksStatistic(x, (v) => betaInc(v, 3, 5))).toBeLessThan(1.63 / Math.sqrt(chainEss(x)));
    expect(chainMean(x)).toBeCloseTo(3 / 8, 1);
  });

  it('the optimal random-walk step in one dimension accepts about 44% of proposals', () => {
    const chain = metropolis(stdNormal, [0], { steps: 20000, proposalSd: 2.4, seed: 3 });
    expect(chain.acceptance).toBeGreaterThan(0.35);
    expect(chain.acceptance).toBeLessThan(0.55);
  });

  it('tiny steps accept almost everything but barely move; huge steps rarely accept', () => {
    const tiny = metropolis(stdNormal, [0], { steps: 5000, proposalSd: 0.02, seed: 4 });
    const huge = metropolis(stdNormal, [0], { steps: 5000, proposalSd: 60, seed: 4 });
    expect(tiny.acceptance).toBeGreaterThan(0.95);
    expect(huge.acceptance).toBeLessThan(0.05);
    const ok = metropolis(stdNormal, [0], { steps: 5000, proposalSd: 2.4, seed: 4 });
    expect(chainEss(chainColumn(tiny, 0))).toBeLessThan(chainEss(chainColumn(ok, 0)) / 5);
    expect(chainEss(chainColumn(huge, 0))).toBeLessThan(chainEss(chainColumn(ok, 0)) / 5);
  });
});

describe('Gibbs and Hamiltonian', () => {
  it('Gibbs reproduces the correlation of a bivariate normal', () => {
    const chain = gibbsBivariateNormal(0.8, [3, -3], 20000, 5);
    const x = chainColumn(chain, 0).slice(500);
    const y = chainColumn(chain, 1).slice(500);
    const mx = chainMean(x);
    const my = chainMean(y);
    let sxy = 0;
    let sxx = 0;
    let syy = 0;
    x.forEach((v, i) => {
      sxy += (v - mx) * ((y[i] as number) - my);
      sxx += (v - mx) ** 2;
      syy += ((y[i] as number) - my) ** 2;
    });
    expect(sxy / Math.sqrt(sxx * syy)).toBeCloseTo(0.8, 1);
  });

  // a strongly correlated 2-D Gaussian: the case where random walks crawl
  const rho = 0.98;
  const logTarget = (x: readonly number[]) =>
    (-0.5 *
      ((x[0] as number) ** 2 -
        2 * rho * (x[0] as number) * (x[1] as number) +
        (x[1] as number) ** 2)) /
    (1 - rho * rho);
  const grad = (x: readonly number[]) => [
    (-(x[0] as number) + rho * (x[1] as number)) / (1 - rho * rho),
    (-(x[1] as number) + rho * (x[0] as number)) / (1 - rho * rho),
  ];

  it('HMC samples the correlated Gaussian and mixes far better than a random walk', () => {
    const hmc = hamiltonian(logTarget, grad, [0, 0], {
      steps: 4000,
      stepSize: 0.1,
      leapfrog: 20,
      seed: 6,
    });
    const rw = metropolis(logTarget, [0, 0], { steps: 4000, proposalSd: 0.15, seed: 6 });
    const hx = chainColumn(hmc, 0).slice(200);
    expect(chainMean(hx)).toBeCloseTo(0, 0);
    expect(ksStatistic(hx, (v) => normalCdf(v))).toBeLessThan(1.63 / Math.sqrt(chainEss(hx)));
    expect(chainEss(chainColumn(hmc, 0))).toBeGreaterThan(chainEss(chainColumn(rw, 0)) * 3);
  });

  it('leapfrog with a huge step size diverges and is rejected', () => {
    const bad = hamiltonian(logTarget, grad, [0, 0], {
      steps: 300,
      stepSize: 1.5,
      leapfrog: 20,
      seed: 7,
    });
    expect(bad.acceptance).toBeLessThan(0.2);
  });
});

describe('diagnostics', () => {
  it('autocorrelation and ESS of an AR(1) series match theory', () => {
    const rng = createRng(8);
    const phi = 0.9;
    const n = 40000;
    const x = new Array<number>(n);
    x[0] = 0;
    for (let i = 1; i < n; i++) x[i] = phi * (x[i - 1] as number) + rng.normal();
    const rho = autocorrelation(x, 3);
    expect(rho[1]).toBeCloseTo(phi, 1);
    expect(rho[2]).toBeCloseTo(phi * phi, 1);
    const theory = (n * (1 - phi)) / (1 + phi);
    expect(chainEss(x) / theory).toBeGreaterThan(0.7);
    expect(chainEss(x) / theory).toBeLessThan(1.4);
  });

  it('R-hat is close to 1 for chains that agree and large for chains stuck in different modes', () => {
    // two well separated modes at -8 and +8
    const bimodal = (x: readonly number[]) => {
      const t = x[0] as number;
      return Math.log(Math.exp(-0.5 * (t + 8) ** 2) + Math.exp(-0.5 * (t - 8) ** 2));
    };
    const stuck = [-8, 8].map((s, i) =>
      chainColumn(metropolis(bimodal, [s], { steps: 3000, proposalSd: 0.8, seed: 10 + i }), 0),
    );
    expect(rHat(stuck)).toBeGreaterThan(5);
    const agree = [1, 2, 3, 4].map((seed) =>
      chainColumn(metropolis(stdNormal, [0], { steps: 3000, proposalSd: 2.4, seed }), 0),
    );
    expect(rHat(agree)).toBeLessThan(1.02);
  });
});
