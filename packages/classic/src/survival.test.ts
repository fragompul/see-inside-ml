import { Matrix, createRng } from '@sim/core';
import { describe, expect, it } from 'vitest';
import fx from '../fixtures/survival.json';
import { coxRegression, kaplanMeier, logRank, survivalAt, survivalQuantile } from './survival';

describe('Kaplan-Meier', () => {
  it('matches statsmodels: times, risk sets, survival and Greenwood standard errors', () => {
    const km = kaplanMeier(fx.time, fx.event);
    expect(km.times).toEqual(fx.km.times);
    expect(km.atRisk).toEqual(fx.km.atRisk);
    expect(km.events).toEqual(fx.km.events);
    fx.km.survival.forEach((s, i) => expect(km.survival[i]).toBeCloseTo(s, 10));
    fx.km.se.forEach((s, i) => {
      if (s !== null) expect(km.se[i]).toBeCloseTo(s, 8);
    });
    expect(survivalQuantile(km, 0.5)).toBe(fx.km.median);
  });

  it('a hand-worked example: 5 subjects, one censored', () => {
    // times 2, 3+, 5, 7, 7: at 2: 5 at risk, 1 event -> 0.8; at 5: 3 at risk, 1 event -> 0.8 * 2/3; at 7: 2 at risk, 1 event (the other is censored)
    const km = kaplanMeier([2, 3, 5, 7, 7], [1, 0, 1, 1, 0]);
    expect(km.times).toEqual([2, 5, 7]);
    expect(km.atRisk).toEqual([5, 3, 2]);
    expect(km.survival[0]).toBeCloseTo(0.8, 12);
    expect(km.survival[1]).toBeCloseTo(0.8 * (2 / 3), 12);
    expect(km.survival[2]).toBeCloseTo(0.8 * (2 / 3) * (1 / 2), 12);
    expect(survivalAt(km, 4)).toBeCloseTo(0.8, 12);
    expect(survivalAt(km, 1)).toBe(1);
  });

  it('with no censoring it is the empirical survival function', () => {
    const t = [1, 2, 3, 4, 5, 6, 7, 8];
    const km = kaplanMeier(
      t,
      t.map(() => 1),
    );
    km.survival.forEach((s, i) => expect(s).toBeCloseTo(1 - (i + 1) / 8, 12));
  });
});

describe('log-rank test', () => {
  it('matches statsmodels', () => {
    const r = logRank(fx.time, fx.event, fx.group);
    expect(r.statistic).toBeCloseTo(fx.logrank.statistic, 8);
    expect(r.pValue).toBeCloseTo(fx.logrank.pValue, 6);
  });

  it('is large when the groups differ and small when they are the same', () => {
    const rng = createRng(3);
    const make = (rate: number[]) => {
      const group = Array.from({ length: 200 }, (_, i) => i % 2);
      const time = group.map((g) => -Math.log(1 - rng.next()) / (rate[g] as number));
      return { group, time, event: group.map(() => 1) };
    };
    const different = make([0.1, 0.3]);
    const same = make([0.2, 0.2]);
    expect(logRank(different.time, different.event, different.group).pValue).toBeLessThan(0.001);
    expect(logRank(same.time, same.event, same.group).pValue).toBeGreaterThan(0.01);
  });
});

describe('Cox regression', () => {
  it('matches statsmodels PHReg (Breslow ties): coefficients, standard errors and log-likelihood', () => {
    const x = new Matrix(
      fx.time.length,
      2,
      Float64Array.from(fx.group.flatMap((g, i) => [g, fx.age[i]!])),
    );
    const fit = coxRegression(fx.time, fx.event, x);
    fx.cox.coef.forEach((c, k) => expect(fit.coef[k]).toBeCloseTo(c, 6));
    fx.cox.se.forEach((s, k) => expect(fit.se[k]).toBeCloseTo(s, 6));
    expect(fit.logLikelihood).toBeCloseTo(fx.cox.llf, 6);
  });

  it('recovers a known hazard ratio from a large simulated sample', () => {
    const rng = createRng(11);
    const n = 3000;
    const x = new Matrix(
      n,
      1,
      Float64Array.from({ length: n }, (_, i) => i % 2),
    );
    const time = Array.from({ length: n }, (_, i) =>
      Math.min(-Math.log(1 - rng.next()) / (0.1 * Math.exp(Math.log(2.5) * x.data[i]!)), 30),
    );
    const event = time.map((t) => (t < 30 ? 1 : 0));
    const fit = coxRegression(time, event, x);
    expect(fit.hazardRatio[0]).toBeGreaterThan(2.3);
    expect(fit.hazardRatio[0]).toBeLessThan(2.7);
    expect(fit.low[0]).toBeLessThan(2.5);
    expect(fit.high[0]).toBeGreaterThan(2.5);
  });
});
