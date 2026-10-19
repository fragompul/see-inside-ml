import { describe, expect, it } from 'vitest';
import {
  convergenceRate,
  evolve,
  googleMatrix,
  isStochastic,
  pageRank,
  pageRankSteps,
  simulateVisits,
  stationary,
  totalVariation,
} from './markov';

describe('stationary distribution', () => {
  it('matches the closed form for a two-state chain', () => {
    const p = 0.2;
    const q = 0.5;
    const pi = stationary([
      [1 - p, p],
      [q, 1 - q],
    ]);
    expect(pi[0]).toBeCloseTo(q / (p + q), 12);
    expect(pi[1]).toBeCloseTo(p / (p + q), 12);
  });

  it('is a fixed point and agrees with the long-run distribution from any start', () => {
    const P = [
      [0.5, 0.3, 0.2],
      [0.1, 0.6, 0.3],
      [0.4, 0.2, 0.4],
    ];
    const pi = stationary(P);
    expect(pi.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 12);
    const after = evolve(pi, P, 1)[1]!;
    pi.forEach((v, i) => expect(after[i]).toBeCloseTo(v, 12));
    const far = evolve([1, 0, 0], P, 200)[200]!;
    expect(totalVariation(far, pi)).toBeLessThan(1e-10);
  });

  it('the distance to stationarity shrinks by the second eigenvalue per step', () => {
    // for a two-state chain the second eigenvalue is 1 - p - q
    const p = 0.2;
    const q = 0.3;
    const P = [
      [1 - p, p],
      [q, 1 - q],
    ];
    expect(convergenceRate(P)).toBeCloseTo(Math.abs(1 - p - q), 6);
    const pi = stationary(P);
    const path = evolve([1, 0], P, 12);
    const d = path.map((x) => totalVariation(x, pi));
    expect(d[10]! / d[9]!).toBeCloseTo(Math.abs(1 - p - q), 6);
  });

  it('a simulated walk spends time in proportion to the stationary distribution', () => {
    const P = [
      [0.7, 0.3],
      [0.4, 0.6],
    ];
    const visits = simulateVisits(P, 0, 200000, 5);
    const pi = stationary(P);
    expect(visits[0]).toBeCloseTo(pi[0]!, 2);
  });

  it('recognises matrices that are not stochastic', () => {
    expect(
      isStochastic([
        [0.5, 0.5],
        [1, 0],
      ]),
    ).toBe(true);
    expect(
      isStochastic([
        [0.5, 0.6],
        [1, 0],
      ]),
    ).toBe(false);
  });
});

describe('PageRank', () => {
  it('a symmetric cycle gives equal ranks', () => {
    const r = pageRank([[1], [2], [0]], 0.85);
    r.forEach((v) => expect(v).toBeCloseTo(1 / 3, 12));
  });

  it('matches the hand-solved three-page example', () => {
    // 0 -> 1, 1 -> 2, 2 -> 0 and 2 -> 1; damping d = 0.5, N = 3
    const links = [[1], [2], [0, 1]];
    const d = 0.5;
    const r = pageRank(links, d);
    // equations: r0 = (1-d)/3 + d r2/2 ; r1 = (1-d)/3 + d (r0 + r2/2) ; r2 = (1-d)/3 + d r1
    expect(r[0]).toBeCloseTo((1 - d) / 3 + (d * r[2]!) / 2, 12);
    expect(r[1]).toBeCloseTo((1 - d) / 3 + d * (r[0]! + r[2]! / 2), 12);
    expect(r[2]).toBeCloseTo((1 - d) / 3 + d * r[1]!, 12);
    expect(r.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 12);
  });

  it('a page with many incoming links ranks highest, and dangling pages do not leak rank', () => {
    const links = [[3], [3], [3], [], [3]]; // page 3 is a sink with no links
    const r = pageRank(links, 0.85);
    expect(r.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 12);
    expect(r.indexOf(Math.max(...r))).toBe(3);
  });

  it('power iteration converges to the exact answer faster with a smaller damping factor', () => {
    const links = [[1, 2], [2], [0], [0, 2]];
    const exact85 = pageRank(links, 0.85);
    const exact50 = pageRank(links, 0.5);
    const e85 = totalVariation(pageRankSteps(links, 0.85, 15)[15]!, exact85);
    const e50 = totalVariation(pageRankSteps(links, 0.5, 15)[15]!, exact50);
    expect(e50).toBeLessThan(e85);
  });

  it('every row of the Google matrix is a distribution', () => {
    expect(isStochastic(googleMatrix([[1, 2], [], [0]], 0.85))).toBe(true);
  });
});
