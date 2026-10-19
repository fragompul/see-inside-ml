import { Matrix } from '@sim/core';
import { describe, expect, it } from 'vitest';
import fixture from '../fixtures/knn.json';
import {
  KnnClassifier,
  KnnRegressor,
  distanceConcentration,
  looAccuracyByK,
  type Metric,
  type NeighborWeights,
} from './knn';

const x = Matrix.from(fixture.x);
const q = Matrix.from(fixture.queries);

describe('k-NN agrees with scikit-learn', () => {
  for (const ref of fixture.classifier) {
    it(`classifier: k = ${ref.k}, ${ref.weights} weights, ${ref.metric}`, () => {
      const m = new KnnClassifier({
        k: ref.k,
        weights: ref.weights as NeighborWeights,
        metric: ref.metric as Metric,
      }).fit(x, fixture.y);
      expect(Array.from(m.predict(q))).toEqual(ref.pred);
      const p = m.predictProba(q);
      ref.proba.forEach((row, i) => row.forEach((v, c) => expect(p.get(i, c)).toBeCloseTo(v, 10)));
    });
  }
  for (const ref of fixture.regressor.fits) {
    it(`regressor: k = ${ref.k}, ${ref.weights} weights`, () => {
      const m = new KnnRegressor({ k: ref.k, weights: ref.weights as NeighborWeights }).fit(
        Matrix.from(fixture.regressor.x),
        fixture.regressor.y,
      );
      const pred = m.predict(Matrix.from(fixture.regressor.grid));
      ref.pred.forEach((v, i) => expect(pred[i]).toBeCloseTo(v, 10));
    });
  }
});

describe('properties', () => {
  it('with k = 1 every training point is its own nearest neighbour: training accuracy is 100%', () => {
    const m = new KnnClassifier({ k: 1 }).fit(x, fixture.y);
    expect(Array.from(m.predict(x))).toEqual(fixture.y);
  });

  it('leave-one-out accuracy is lower than training accuracy at k = 1 and peaks at a moderate k', () => {
    const acc = looAccuracyByK(x, fixture.y, 40);
    expect(acc[0]).toBeLessThan(1);
    const best = Math.max(...acc);
    expect(best).toBeGreaterThan(acc[0] as number);
    expect(acc.indexOf(best) + 1).toBeGreaterThan(1);
  });

  it('with k = n every query gets the majority class of the training set', () => {
    const m = new KnnClassifier({ k: x.rows }).fit(x, fixture.y);
    const counts = [0, 0, 0];
    for (const c of fixture.y) counts[c] = (counts[c] as number) + 1;
    const majority = counts.indexOf(Math.max(...counts));
    for (const c of m.predict(q)) expect(c).toBe(majority);
  });

  it('distance weighting returns the training label for a query that sits on a training point', () => {
    const m = new KnnClassifier({ k: 9, weights: 'distance' }).fit(x, fixture.y);
    expect(m.predict(x)[17]).toBe(fixture.y[17]);
  });

  it('neighbours come back nearest first', () => {
    const m = new KnnClassifier({ k: 6 }).fit(x, fixture.y);
    const near = m.neighbors(q, 0);
    for (let i = 1; i < near.length; i++)
      expect(near[i]!.distance).toBeGreaterThanOrEqual(near[i - 1]!.distance);
  });
});

describe('curse of dimensionality', () => {
  it('the nearest point is relatively ever less close as the dimension grows', () => {
    const ratio = distanceConcentration([1, 2, 5, 10, 30, 100], { seed: 5 });
    for (let i = 1; i < ratio.length; i++) expect(ratio[i]).toBeGreaterThan(ratio[i - 1] as number);
    expect(ratio[0]).toBeLessThan(0.05);
    expect(ratio[ratio.length - 1]).toBeGreaterThan(0.5);
  });

  it('is reproducible for a seed', () => {
    expect(distanceConcentration([3, 8], { seed: 9 })).toEqual(
      distanceConcentration([3, 8], { seed: 9 }),
    );
  });
});
