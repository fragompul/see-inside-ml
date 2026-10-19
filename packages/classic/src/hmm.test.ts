import { createRng } from '@sim/core';
import { describe, expect, it } from 'vitest';
import {
  baumWelchSteps,
  bruteForce,
  forward,
  randomParams,
  sample,
  smooth,
  viterbi,
  type HmmParams,
} from './hmm';

const weather: HmmParams = {
  initial: [0.6, 0.4],
  transition: [
    [0.7, 0.3],
    [0.4, 0.6],
  ],
  emission: [
    [0.1, 0.4, 0.5],
    [0.6, 0.3, 0.1],
  ],
};

describe('HMM inference against exhaustive enumeration', () => {
  const sequences = [[0, 1, 2], [2, 2, 0, 1], [1, 0, 0, 2, 1, 2], [0]];
  it.each(sequences)('forward log-likelihood is exact for %j', (...obs) => {
    const bf = bruteForce(weather, obs);
    expect(forward(weather, obs).logLikelihood).toBeCloseTo(bf.logLikelihood, 10);
  });
  it.each(sequences)('Viterbi finds the best path for %j', (...obs) => {
    const bf = bruteForce(weather, obs);
    const v = viterbi(weather, obs);
    expect(v.logProbability).toBeCloseTo(bf.bestLogProbability, 10);
    expect(v.path).toEqual(bf.bestPath);
  });
  it('smoothed marginals match the sum over all paths', () => {
    const obs = [2, 0, 1, 2];
    const { gamma } = smooth(weather, obs);
    let num = 0;
    let den = 0;
    for (let code = 0; code < 16; code++) {
      const path = [code & 1, (code >> 1) & 1, (code >> 2) & 1, (code >> 3) & 1];
      let pr = weather.initial[path[0]!]!;
      for (let t = 0; t < 4; t++) {
        if (t > 0) pr *= weather.transition[path[t - 1]!]![path[t]!]!;
        pr *= weather.emission[path[t]!]![obs[t]!]!;
      }
      den += pr;
      if (path[2] === 1) num += pr;
    }
    expect(gamma[2]![1]).toBeCloseTo(num / den, 10);
    for (const g of gamma) expect(g.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 10);
  });
  it('does not underflow on long sequences', () => {
    const { obs } = sample(weather, 5000, createRng(3));
    const ll = forward(weather, obs).logLikelihood;
    expect(Number.isFinite(ll)).toBe(true);
    expect(ll).toBeLessThan(0);
  });
});

describe('Baum-Welch', () => {
  // well separated emissions and sticky states, so the parameters are identifiable from the data
  const distinct: HmmParams = {
    initial: [0.5, 0.5],
    transition: [
      [0.9, 0.1],
      [0.15, 0.85],
    ],
    emission: [
      [0.7, 0.2, 0.1],
      [0.05, 0.25, 0.7],
    ],
  };
  const rng = createRng(11);
  const seqs = Array.from({ length: 6 }, () => sample(distinct, 200, rng).obs);
  it('never decreases the likelihood', () => {
    const states = [...baumWelchSteps(randomParams(2, 3, 5), seqs, 60)];
    for (let i = 1; i < states.length; i++)
      expect(states[i]!.logLikelihood).toBeGreaterThanOrEqual(states[i - 1]!.logLikelihood - 1e-9);
  });
  it('recovers the generating model up to relabelling of the states', () => {
    const states = [...baumWelchSteps(randomParams(2, 3, 5), seqs, 200)];
    const fit = states[states.length - 1]!;
    const trueTotal = seqs.reduce((s, o) => s + forward(distinct, o).logLikelihood, 0);
    // the fitted model explains the data at least about as well as the true one
    expect(fit.logLikelihood).toBeGreaterThan(trueTotal - 5);
    const e = fit.params.emission;
    const dist = (a: number[], b: number[]) => a.reduce((s, v, i) => s + Math.abs(v - b[i]!), 0);
    const direct = dist(e[0]!, distinct.emission[0]!) + dist(e[1]!, distinct.emission[1]!);
    const swapped = dist(e[0]!, distinct.emission[1]!) + dist(e[1]!, distinct.emission[0]!);
    expect(Math.min(direct, swapped)).toBeLessThan(0.35);
  });
});
