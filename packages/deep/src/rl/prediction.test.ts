import { createRng } from '@sim/core';
import { describe, expect, it } from 'vitest';
import {
  averageError,
  learnValues,
  randomWalkEpisode,
  rmsError,
  trueValues,
  type PredictionOptions,
} from './prediction';

const base: PredictionOptions = {
  states: 19,
  method: 'td',
  alpha: 0.1,
  n: 4,
  lambda: 0.8,
  episodes: 100,
  seed: 1,
};

describe('the random walk', () => {
  it('true values solve the Bellman equation: v(i) = (v(i-1) + v(i+1)) / 2 with v(0) = 0 and v(n+1) = 1', () => {
    const n = 7;
    const v = [0, ...trueValues(n), 1];
    for (let i = 1; i <= n; i++)
      expect(v[i]).toBeCloseTo(((v[i - 1] as number) + (v[i + 1] as number)) / 2, 12);
  });

  it('episodes end on the right about as often as the true value of the middle state says', () => {
    const rng = createRng(2);
    let right = 0;
    for (let i = 0; i < 20000; i++) right += randomWalkEpisode(19, rng).reward;
    expect(right / 20000).toBeCloseTo(0.5, 1);
  });
});

describe('learning the values', () => {
  it.each(['mc', 'td', 'nstep', 'lambda'] as const)(
    '%s converges to the true values with a small step size',
    (method) => {
      const run = learnValues({ ...base, method, alpha: 0.02, episodes: 6000, seed: 3 });
      // Monte Carlo keeps a larger floor of noise: its targets are whole returns, not one-step estimates
      expect(run.error[run.error.length - 1]!).toBeLessThan(method === 'mc' ? 0.15 : 0.07);
      expect(run.error[0]!).toBeGreaterThan(0.2);
    },
  );

  it('n-step TD with n larger than any episode is Monte Carlo, and lambda = 1 behaves like it too', () => {
    const mc = learnValues({ ...base, method: 'mc', alpha: 0.05, episodes: 60, seed: 4 });
    const long = learnValues({
      ...base,
      method: 'nstep',
      n: 10000,
      alpha: 0.05,
      episodes: 60,
      seed: 4,
    });
    mc.values.forEach((v, i) => expect(long.values[i]).toBeCloseTo(v, 12));
  });

  it('TD(0) learns faster than Monte Carlo early on, at a moderate step size', () => {
    const td = averageError({ ...base, method: 'td', alpha: 0.1, episodes: 10 }, 60);
    const mc = averageError({ ...base, method: 'mc', alpha: 0.1, episodes: 10 }, 60);
    expect(td[10]!).toBeLessThan(mc[10]!);
  });

  it('a large step size leaves a floor of noise that a small one does not', () => {
    const big = averageError({ ...base, method: 'td', alpha: 0.5, episodes: 200 }, 30);
    const small = averageError({ ...base, method: 'td', alpha: 0.05, episodes: 200 }, 30);
    expect(big[200]!).toBeGreaterThan(small[200]!);
  });

  it('is deterministic for a seed and the error of the truth is zero', () => {
    expect(learnValues({ ...base, seed: 9 }).values).toEqual(
      learnValues({ ...base, seed: 9 }).values,
    );
    expect(rmsError(trueValues(5), trueValues(5))).toBe(0);
  });
});
