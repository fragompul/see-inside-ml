import type { Matrix } from '@sim/core';
import { accuracy, createRng, makeClassification } from '@sim/core';
import { describe, expect, it } from 'vitest';
import { DecisionTree } from './tree';
import { GaussianNB } from './gaussian';
import { KnnClassifier } from './knn';
import { LogisticRegression } from './logistic';
import { StackingClassifier, VotingClassifier } from './voting';

const train = makeClassification('moons', 240, 0.4, createRng(3));
const test = makeClassification('moons', 400, 0.4, createRng(30));
const makers = [
  () => new LogisticRegression(),
  () => new KnnClassifier({ k: 9 }),
  () => new DecisionTree({ maxDepth: 4, minSamplesLeaf: 4 }),
  () => new GaussianNB(),
];
const acc = (m: { predict: (x: Matrix) => Int32Array }) => accuracy(test.y, m.predict(test.x));

describe('voting', () => {
  it('soft voting is exactly the average of the members’ probabilities', () => {
    const v = new VotingClassifier(makers, { voting: 'soft' }).fit(train.x, train.y);
    const p = v.predictProba(test.x);
    for (let i = 0; i < 20; i++) {
      for (let c = 0; c < 2; c++) {
        const mean =
          v.members.reduce(
            (s, m) => s + (m.predictProba as (x: Matrix) => Matrix)(test.x).get(i, c),
            0,
          ) / v.members.length;
        expect(p.get(i, c)).toBeCloseTo(mean, 12);
      }
    }
  });

  it('hard voting returns the majority class and its vote share', () => {
    const v = new VotingClassifier(makers, { voting: 'hard' }).fit(train.x, train.y);
    const preds = v.members.map((m) => m.predict(test.x));
    const p = v.predictProba(test.x);
    for (let i = 0; i < 30; i++) {
      const ones = preds.filter((q) => q[i] === 1).length;
      expect(p.get(i, 1)).toBeCloseTo(ones / preds.length, 12);
    }
  });

  it('a diverse ensemble is at least as good as the average member and not far from the best', () => {
    const v = new VotingClassifier(makers, { voting: 'soft' }).fit(train.x, train.y);
    const member = v.members.map(acc);
    const mean = member.reduce((a, b) => a + b, 0) / member.length;
    expect(acc(v)).toBeGreaterThanOrEqual(mean);
    expect(acc(v)).toBeGreaterThan(Math.max(...member) - 0.07);
  });

  it('weights tilt the vote: all weight on one member reproduces it', () => {
    const v = new VotingClassifier(makers, { voting: 'hard', weights: [0, 1, 0, 0] }).fit(
      train.x,
      train.y,
    );
    expect(Array.from(v.predict(test.x))).toEqual(Array.from(v.members[1]!.predict(test.x)));
  });
});

describe('stacking', () => {
  it('is competitive with its best base model and trusts the useful ones', () => {
    const s = new StackingClassifier(makers, { folds: 5, seed: 2 }).fit(train.x, train.y);
    const base = s.bases.map(acc);
    expect(acc(s)).toBeGreaterThan(Math.max(...base) - 0.04);
    expect(acc(s)).toBeGreaterThan(Math.min(...base));
    const trust = s.trust();
    expect(trust).toHaveLength(makers.length);
    // the model with the lowest accuracy is not the one the meta-learner leans on most
    const worst = base.indexOf(Math.min(...base));
    expect(trust.indexOf(Math.max(...trust))).not.toBe(worst);
  });

  it('builds its meta-features only from rows the base models did not train on', () => {
    // a memorising base model (1-NN) would be perfect on its own training rows; out-of-fold it is not
    const s = new StackingClassifier(
      [() => new KnnClassifier({ k: 1 }), () => new LogisticRegression()],
      { folds: 5, seed: 1 },
    ).fit(train.x, train.y);
    // if the meta-learner had seen in-sample 1-NN outputs it would trust it overwhelmingly
    const [knn, lr] = s.trust() as [number, number];
    expect(knn).toBeLessThan(5 * lr);
  });

  it('predicts valid probabilities', () => {
    const s = new StackingClassifier(makers, { seed: 4 }).fit(train.x, train.y);
    const p = s.predictProba(test.x);
    for (let i = 0; i < 25; i++) expect(p.get(i, 0) + p.get(i, 1)).toBeCloseTo(1, 12);
  });
});
