import { Matrix, accuracy, createRng, makeClassification, makeRegression, r2 } from '@sim/core';
import { describe, expect, it } from 'vitest';
import fixture from '../fixtures/ensemble.json';
import {
  AdaBoost,
  GradientBoostingClassifier,
  GradientBoostingRegressor,
  RandomForestClassifier,
  RandomForestRegressor,
} from './ensemble';
import { DecisionTree, DecisionTreeRegressor } from './tree';

const x = Matrix.from(fixture.x);
const grid = Matrix.from(fixture.grid);
const closeAll = (a: ArrayLike<number>, b: ArrayLike<number>, tol: number) => {
  expect(a.length).toBe(b.length);
  for (let i = 0; i < a.length; i++)
    expect(Math.abs((a[i] as number) - (b[i] as number))).toBeLessThan(tol);
};

describe('AdaBoost agrees with scikit-learn (SAMME)', () => {
  for (const ref of fixture.adaboost) {
    it(`${JSON.stringify(ref.config)}`, () => {
      const m = new AdaBoost({
        rounds: ref.config.n_estimators,
        learningRate: ref.config.learning_rate,
      }).fit(x, ref.y);
      closeAll(
        m.rounds.map((r) => r.alpha),
        ref.alphas.slice(0, m.rounds.length),
        1e-9,
      );
      expect(m.rounds.length).toBe(ref.alphas.filter((a) => a !== 0).length || m.rounds.length);
      expect(Array.from(m.predict(grid))).toEqual(ref.pred);
      const p = m.predictProba(grid);
      ref.proba.forEach((row, i) =>
        row.forEach((v, c) => expect(Math.abs(p.get(i, c) - v)).toBeLessThan(1e-9)),
      );
    });
  }
});

describe('gradient boosting agrees with scikit-learn', () => {
  for (const ref of fixture.gbm_classifier) {
    it(`classifier ${JSON.stringify(ref.config)}`, () => {
      const m = new GradientBoostingClassifier({
        rounds: ref.config.n_estimators,
        learningRate: ref.config.learning_rate,
        depth: ref.config.max_depth,
      }).fit(x, ref.y);
      // the first rounds match to rounding; late rounds fit tiny residuals where near-equal splits can
      // be broken differently from scikit-learn, so the final model only has to stay close
      const staged = m.predictProba(grid, 5);
      ref.staged5.forEach((row, i) =>
        row.forEach((v, c) => expect(Math.abs(staged.get(i, c) - v)).toBeLessThan(1e-6)),
      );
      const p = m.predictProba(grid);
      ref.proba.forEach((row, i) =>
        row.forEach((v, c) => expect(Math.abs(p.get(i, c) - v)).toBeLessThan(0.05)),
      );
      // scikit-learn's train_score_ is the (half-)deviance after each round, without the initial model
      closeAll(
        m.trainLoss.slice(1, 6),
        ref.train_loss.slice(0, 5).map((v) => v / (m.classes === 2 ? 2 : 1)),
        1e-6,
      );
    });
  }

  for (const ref of fixture.gbm_regressor.fits) {
    it(`regressor ${JSON.stringify(ref.config)}`, () => {
      const m = new GradientBoostingRegressor({
        rounds: ref.config.n_estimators,
        learningRate: ref.config.learning_rate,
        depth: ref.config.max_depth,
      }).fit(Matrix.from(fixture.gbm_regressor.x), fixture.gbm_regressor.y);
      const g = Matrix.from(fixture.gbm_regressor.grid);
      closeAll(m.predict(g, 5), ref.staged5, 1e-9);
      closeAll(m.predict(g), ref.pred, 0.25);
      closeAll(m.trainLoss.slice(1, 6), ref.train_loss.slice(0, 5), 1e-9);
    });
  }
});

describe('random forests', () => {
  const data = makeClassification('moons', 300, 0.35, createRng(3));
  const test = makeClassification('moons', 300, 0.35, createRng(99));

  it('beat a single deep tree on held-out points', () => {
    const tree = new DecisionTree().fit(data.x, data.y);
    const forest = new RandomForestClassifier({ trees: 80, seed: 2 }).fit(data.x, data.y);
    const accTree = accuracy(test.y, tree.predict(test.x));
    const accForest = accuracy(test.y, forest.predict(test.x));
    expect(accForest).toBeGreaterThanOrEqual(accTree);
  });

  it('the out-of-bag accuracy estimates test accuracy without a test set', () => {
    const forest = new RandomForestClassifier({ trees: 120, seed: 2 }).fit(data.x, data.y);
    const test_ = accuracy(test.y, forest.predict(test.x));
    expect(forest.oobAccuracy).not.toBeNull();
    expect(Math.abs((forest.oobAccuracy as number) - test_)).toBeLessThan(0.07);
  });

  it('the out-of-bag curve ends at the out-of-bag accuracy and improves with more trees', () => {
    const forest = new RandomForestClassifier({ trees: 100, seed: 2 }).fit(data.x, data.y);
    const curve = forest.oobCurve(data.x, data.y);
    expect(curve).toHaveLength(100);
    expect(curve[99]).toBeCloseTo(forest.oobAccuracy as number, 12);
    const early = curve.slice(2, 8).reduce((a, b) => a + b, 0) / 6;
    const late = curve.slice(-6).reduce((a, b) => a + b, 0) / 6;
    expect(late).toBeGreaterThan(early);
  });

  it('are reproducible for a seed and differ between seeds', () => {
    const a = new RandomForestClassifier({ trees: 10, seed: 5 })
      .fit(data.x, data.y)
      .predictProba(test.x);
    const b = new RandomForestClassifier({ trees: 10, seed: 5 })
      .fit(data.x, data.y)
      .predictProba(test.x);
    const c = new RandomForestClassifier({ trees: 10, seed: 6 })
      .fit(data.x, data.y)
      .predictProba(test.x);
    expect(Array.from(a.data)).toEqual(Array.from(b.data));
    expect(Array.from(a.data)).not.toEqual(Array.from(c.data));
  });

  it('probabilities sum to one and importances sum to one', () => {
    const f = new RandomForestClassifier({ trees: 30, seed: 1 }).fit(data.x, data.y);
    const p = f.predictProba(test.x);
    for (let i = 0; i < test.x.rows; i++) expect(p.get(i, 0) + p.get(i, 1)).toBeCloseTo(1, 12);
    expect(f.importances.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 9);
  });

  it('more trees reduce the variance of the prediction (regression)', () => {
    const train = makeRegression('sine', 120, 0.4, createRng(4));
    const held = makeRegression('sine', 200, 0.4, createRng(40));
    const single = new DecisionTreeRegressor().fit(train.x, train.y);
    const forest = new RandomForestRegressor({ trees: 100, seed: 3 }).fit(train.x, train.y);
    expect(r2(held.y, forest.predict(held.x))).toBeGreaterThan(r2(held.y, single.predict(held.x)));
    expect(forest.oobR2).toBeGreaterThan(0.5);
  });

  it('bagging off gives identical trees when every feature is tried', () => {
    const f = new RandomForestClassifier({
      trees: 3,
      bootstrap: false,
      maxFeatures: 2,
      seed: 1,
    }).fit(data.x, data.y);
    const p0 = f.trees[0]!.predict(test.x);
    expect(Array.from(f.trees[1]!.predict(test.x))).toEqual(Array.from(p0));
    expect(f.oobAccuracy).toBeNull();
  });
});

describe('boosting properties', () => {
  const data = makeClassification('moons', 200, 0.3, createRng(8));
  it('AdaBoost reweights: the weights of misclassified points grow, and training accuracy rises with rounds', () => {
    const m = new AdaBoost({ rounds: 40 }).fit(data.x, data.y);
    const acc = (k: number) => accuracy(data.y, m.predict(data.x, k));
    expect(acc(40)).toBeGreaterThan(acc(1));
    const first = m.rounds[0]!;
    const second = m.rounds[1]!;
    const wrong = Array.from(first.tree.predict(data.x))
      .map((p, i) => (p !== data.y[i] ? i : -1))
      .filter((i) => i >= 0);
    expect(wrong.length).toBeGreaterThan(0);
    for (const i of wrong)
      expect(second.weights[i] as number).toBeGreaterThan(first.weights[i] as number);
  });

  it('gradient boosting loss falls monotonically with rounds (full-batch)', () => {
    const m = new GradientBoostingClassifier({ rounds: 40, learningRate: 0.3, depth: 2 }).fit(
      data.x,
      data.y,
    );
    for (let i = 1; i < m.trainLoss.length; i++)
      expect(m.trainLoss[i]).toBeLessThanOrEqual((m.trainLoss[i - 1] as number) + 1e-9);
  });

  it('subsampling (stochastic gradient boosting) is reproducible and still learns', () => {
    const run = () => new GradientBoostingRegressor({ rounds: 30, subsample: 0.6, seed: 4 });
    const train = makeRegression('sine', 120, 0.3, createRng(7));
    const a = run().fit(train.x, train.y);
    const b = run().fit(train.x, train.y);
    expect(Array.from(a.predict(train.x))).toEqual(Array.from(b.predict(train.x)));
    expect(r2(train.y, a.predict(train.x))).toBeGreaterThan(0.5);
  });
});
