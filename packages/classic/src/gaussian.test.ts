import { Matrix } from '@sim/core';
import { describe, expect, it } from 'vitest';
import fixture from '../fixtures/gaussian.json';
import { GaussianNB, LDA, QDA } from './gaussian';

const x = Matrix.from(fixture.x);
const q = Matrix.from(fixture.queries);

const agrees = (
  model: { predict: (m: Matrix) => Int32Array; predictProba: (m: Matrix) => Matrix },
  ref: { pred: number[]; proba: number[][] },
  tol = 1e-8,
) => {
  expect(Array.from(model.predict(q))).toEqual(ref.pred);
  const p = model.predictProba(q);
  ref.proba.forEach((row, i) =>
    row.forEach((v, c) => expect(Math.abs(p.get(i, c) - v)).toBeLessThan(tol)),
  );
};

describe('Gaussian classifiers agree with scikit-learn', () => {
  it('Gaussian naive Bayes: parameters and probabilities', () => {
    const ref = fixture.models.nb;
    const m = new GaussianNB().fit(x, fixture.y);
    ref.theta.forEach((row, c) => row.forEach((v, j) => expect(m.means[c]![j]).toBeCloseTo(v, 10)));
    ref.var.forEach((row, c) =>
      row.forEach((v, j) => expect(m.variances[c]![j]).toBeCloseTo(v, 10)),
    );
    ref.prior.forEach((v, c) => expect(m.priors[c]).toBeCloseTo(v, 12));
    agrees(m, ref);
  });

  it('LDA (lsqr): shared covariance and probabilities', () => {
    const ref = fixture.models.lda;
    const m = new LDA().fit(x, fixture.y);
    ref.covariance.forEach((row, a) =>
      row.forEach((v, b) => expect(m.pooled.get(a, b)).toBeCloseTo(v, 10)),
    );
    ref.means.forEach((row, c) => row.forEach((v, j) => expect(m.means[c]![j]).toBeCloseTo(v, 10)));
    agrees(m, ref);
  });

  it('LDA with shrinkage', () => {
    const ref = fixture.models.lda_shrink;
    const m = new LDA({ shrinkage: 0.4 }).fit(x, fixture.y);
    ref.covariance.forEach((row, a) =>
      row.forEach((v, b) => expect(m.pooled.get(a, b)).toBeCloseTo(v, 9)),
    );
    agrees(m, ref, 1e-7);
  });

  it('QDA', () => agrees(new QDA().fit(x, fixture.y), fixture.models.qda, 1e-7));
  it('QDA with regularisation', () =>
    agrees(new QDA({ regParam: 0.3 }).fit(x, fixture.y), fixture.models.qda_reg, 1e-7));
});

describe('properties', () => {
  it('probabilities sum to one', () => {
    for (const m of [new GaussianNB(), new LDA(), new QDA()]) {
      const p = m.fit(x, fixture.y).predictProba(q);
      for (let i = 0; i < q.rows; i++)
        expect(p.get(i, 0) + p.get(i, 1) + p.get(i, 2)).toBeCloseTo(1, 12);
    }
  });

  it('LDA boundaries are straight: the log-odds are an affine function of x', () => {
    const m = new LDA().fit(x, fixture.y);
    const logOdds = (i: number) => {
      const p = m.predictProba(Matrix.from([fixture.queries[i] as number[]]));
      return Math.log(p.get(0, 1) / p.get(0, 0));
    };
    const a = fixture.queries[0] as number[];
    const b = fixture.queries[1] as number[];
    const mid = a.map((v, j) => (v + (b[j] as number)) / 2);
    const pm = m.predictProba(Matrix.from([mid]));
    expect(Math.log(pm.get(0, 1) / pm.get(0, 0))).toBeCloseTo((logOdds(0) + logOdds(1)) / 2, 9);
  });

  it('QDA fits the training data at least as well as LDA when covariances differ', () => {
    const acc = (m: { predict: (a: Matrix) => Int32Array }) =>
      Array.from(m.predict(x)).filter((p, i) => p === fixture.y[i]).length / x.rows;
    expect(acc(new QDA().fit(x, fixture.y))).toBeGreaterThanOrEqual(
      acc(new LDA().fit(x, fixture.y)),
    );
  });

  it('priors shift the decision towards the common class', () => {
    const base = new LDA().fit(x, fixture.y);
    const tilted = new LDA({ priors: [0.8, 0.1, 0.1] }).fit(x, fixture.y);
    const count = (m: LDA) => Array.from(m.predict(q)).filter((c) => c === 0).length;
    expect(count(tilted)).toBeGreaterThanOrEqual(count(base));
  });
});
