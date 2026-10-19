import { createRng, normalCdf } from '@sim/core';
import { describe, expect, it } from 'vitest';
import {
  LANDSCAPES,
  bayesianOptimization,
  expectedImprovement,
  gridSearch,
  halvingConfigurations,
  measure,
  randomSearch,
  successiveHalving,
} from './tuning';

describe('landscapes', () => {
  it.each(Object.values(LANDSCAPES))(
    '$id: the stated optimum is attained and never exceeded',
    (land) => {
      expect(land.value(land.argBest)).toBeCloseTo(land.best, 2);
      const rng = createRng(1);
      for (let i = 0; i < 2000; i++)
        expect(land.value([rng.next(), rng.next()])).toBeLessThanOrEqual(land.best + 1e-9);
    },
  );

  it('measurements are noisier when they use less of the experiment', () => {
    const land = LANDSCAPES.bowl as (typeof LANDSCAPES)[string];
    const rng = createRng(2);
    const sd = (fid: number) => {
      const v = Array.from(
        { length: 4000 },
        () => measure(land, [0.5, 0.5], 0.1, fid, rng) - land.value([0.5, 0.5]),
      );
      return Math.sqrt(v.reduce((a, b) => a + b * b, 0) / v.length);
    };
    expect(sd(0.25)).toBeCloseTo(0.2, 1);
    expect(sd(1)).toBeCloseTo(0.1, 1);
  });
});

describe('expected improvement', () => {
  it('is the improvement itself when there is no uncertainty', () => {
    expect(expectedImprovement(0.8, 0, 0.5)).toBeCloseTo(0.3, 12);
    expect(expectedImprovement(0.4, 0, 0.5)).toBe(0);
  });

  it('matches numerical integration of max(f - best, 0) under a normal belief', () => {
    const mean = 0.45;
    const std = 0.2;
    const best = 0.5;
    let sum = 0;
    const steps = 200000;
    const lo = mean - 8 * std;
    const hi = mean + 8 * std;
    const h = (hi - lo) / steps;
    for (let i = 0; i < steps; i++) {
      const f = lo + (i + 0.5) * h;
      const pdf = Math.exp(-0.5 * ((f - mean) / std) ** 2) / (std * Math.sqrt(2 * Math.PI));
      sum += Math.max(f - best, 0) * pdf * h;
    }
    expect(expectedImprovement(mean, std, best)).toBeCloseTo(sum, 6);
  });

  it('grows with the mean and with the uncertainty', () => {
    expect(expectedImprovement(0.6, 0.1, 0.5)).toBeGreaterThan(expectedImprovement(0.5, 0.1, 0.5));
    expect(expectedImprovement(0.4, 0.3, 0.5)).toBeGreaterThan(expectedImprovement(0.4, 0.05, 0.5));
    expect(normalCdf(0)).toBeCloseTo(0.5, 12);
  });
});

describe('search strategies', () => {
  const bowl = LANDSCAPES.bowl as (typeof LANDSCAPES)[string];
  it('grid search tries only a few distinct values per hyperparameter, random search one per trial', () => {
    const g = gridSearch(bowl, 25, 0.05);
    const r = randomSearch(bowl, 25, 0.05);
    expect(new Set(g.evaluations.map((e) => e.x[0].toFixed(6))).size).toBe(5);
    expect(new Set(r.evaluations.map((e) => e.x[0].toFixed(6))).size).toBe(25);
    expect(g.cost).toBe(25);
    expect(r.cost).toBe(25);
  });

  it('on a landscape where one setting matters, random search beats a grid of the same size on average', () => {
    const land = LANDSCAPES.oneThing as (typeof LANDSCAPES)[string];
    const avg = (f: (s: number) => number) =>
      Array.from({ length: 40 }, (_, s) => f(s + 1)).reduce((a, b) => a + b, 0) / 40;
    const grid = avg((s) => gridSearch(land, 16, 0.02, s).regret);
    const random = avg((s) => randomSearch(land, 16, 0.02, s).regret);
    expect(random).toBeLessThan(grid);
  });

  it('Bayesian optimisation beats random search on the narrow ridge on average', () => {
    const land = LANDSCAPES.ridge as (typeof LANDSCAPES)[string];
    const avg = (f: (s: number) => number) =>
      Array.from({ length: 12 }, (_, s) => f(s + 1)).reduce((a, b) => a + b, 0) / 12;
    const bo = avg((s) => bayesianOptimization(land, 20, 0.02, s).regret);
    const random = avg((s) => randomSearch(land, 20, 0.02, s).regret);
    expect(bo).toBeLessThan(random);
  });

  it('every method spends exactly its budget (in full experiments), halving at most', () => {
    expect(bayesianOptimization(bowl, 15, 0.05).cost).toBe(15);
    const budget = 20;
    const n = halvingConfigurations(budget);
    const sh = successiveHalving(bowl, n, 0.05);
    expect(sh.cost).toBeLessThanOrEqual(budget + 1e-9);
    expect(halvingConfigurations(budget + 5)).toBeGreaterThanOrEqual(n);
  });

  it('successive halving evaluates many more configurations than random search for the same cost', () => {
    const sh = successiveHalving(bowl, halvingConfigurations(20), 0.05);
    const distinct = new Set(sh.evaluations.map((e) => e.x.join(','))).size;
    expect(distinct).toBeGreaterThan(40);
  });

  it('is deterministic for a seed', () => {
    expect(randomSearch(bowl, 10, 0.1, 5).recommended).toEqual(
      randomSearch(bowl, 10, 0.1, 5).recommended,
    );
    expect(bayesianOptimization(bowl, 10, 0.1, 5).recommended).toEqual(
      bayesianOptimization(bowl, 10, 0.1, 5).recommended,
    );
  });
});
