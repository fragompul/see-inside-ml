import { Matrix, createRng, makeRegression } from '@sim/core';
import { describe, expect, it } from 'vitest';
import fixture from '../fixtures/xgboost.json';
import { XgbClassifier, XgbRegressor, gainImportance, type XgbOptions } from './xgboost';

const optionsOf = (c: Record<string, number>): XgbOptions => ({
  rounds: c.n_estimators as number,
  eta: c.learning_rate as number,
  maxDepth: c.max_depth as number,
  lambda: c.reg_lambda as number,
  gamma: c.gamma as number,
  minChildWeight: c.min_child_weight as number,
  baseScore: 0.5,
});

describe('XGBoost-style boosting agrees with xgboost (exact greedy)', () => {
  const x = Matrix.from(fixture.x);
  const grid = Matrix.from(fixture.grid);
  for (const ref of fixture.classifier) {
    it(`classifier ${JSON.stringify(ref.config)}`, () => {
      const m = new XgbClassifier(optionsOf(ref.config)).fit(x, ref.y);
      const p = m.predictProba(grid);
      ref.proba.forEach((row, i) =>
        row.forEach((v, c) => expect(Math.abs(p.get(i, c) - v)).toBeLessThan(1e-4)),
      );
      const s = m.predictProba(grid, 3);
      ref.staged3.forEach((row, i) =>
        row.forEach((v, c) => expect(Math.abs(s.get(i, c) - v)).toBeLessThan(1e-4)),
      );
    });
  }
  for (const ref of fixture.regressor.fits) {
    it(`regressor ${JSON.stringify(ref.config)}`, () => {
      const m = new XgbRegressor(optionsOf(ref.config)).fit(
        Matrix.from(fixture.regressor.x),
        fixture.regressor.y,
      );
      const g = Matrix.from(fixture.regressor.grid);
      const pred = m.predict(g);
      ref.pred.forEach((v, i) => expect(Math.abs((pred[i] as number) - v)).toBeLessThan(1e-4));
      const s = m.predict(g, 3);
      ref.staged3.forEach((v, i) => expect(Math.abs((s[i] as number) - v)).toBeLessThan(1e-4));
    });
  }
});

describe('regularisation knobs', () => {
  const t = makeRegression('sine', 150, 0.3, createRng(2));
  const fitSize = (opts: XgbOptions) => {
    const m = new XgbRegressor({ rounds: 1, eta: 1, ...opts }).fit(t.x, t.y);
    const count = (n: XgbRegressor['trees'][number]): number =>
      n.left && n.right ? 1 + count(n.left) + count(n.right) : 0;
    return count(m.trees[0] as XgbRegressor['trees'][number]);
  };

  it('a larger γ prunes splits, a larger minChildWeight forbids small leaves', () => {
    expect(fitSize({ maxDepth: 6, gamma: 5 })).toBeLessThan(fitSize({ maxDepth: 6, gamma: 0 }));
    expect(fitSize({ maxDepth: 6, minChildWeight: 20 })).toBeLessThan(
      fitSize({ maxDepth: 6, minChildWeight: 1 }),
    );
  });

  it('a larger λ shrinks leaf values towards zero', () => {
    const big = (lambda: number) => {
      const m = new XgbRegressor({ rounds: 1, eta: 1, maxDepth: 1, lambda, baseScore: 0 }).fit(
        t.x,
        t.y,
      );
      return Math.max(...[m.trees[0]!.left!.value, m.trees[0]!.right!.value].map(Math.abs));
    };
    expect(big(50)).toBeLessThan(big(0));
  });

  it('training error falls with every round and importances sum to one', () => {
    const m = new XgbRegressor({ rounds: 30, eta: 0.3, maxDepth: 3, baseScore: 0 }).fit(t.x, t.y);
    for (let i = 1; i < m.trainRmse.length; i++)
      expect(m.trainRmse[i]).toBeLessThanOrEqual((m.trainRmse[i - 1] as number) + 1e-9);
    expect(gainImportance(m.trees, 1)[0]).toBeCloseTo(1, 12);
  });
});
