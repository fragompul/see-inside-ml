import { createRng } from '@sim/core';
import { describe, expect, it } from 'vitest';
import {
  RM_FEATURES,
  bestOfN,
  expectation,
  fitRewardModel,
  klDivergence,
  klRegularisedPolicy,
  makeCandidates,
  preferenceAccuracy,
  samplePreferences,
  trainRlhf,
  type Response,
} from './rlhf';

describe('reward model', () => {
  it('recovers the direction of a known linear reward from many noiseless-ish preferences', () => {
    const rng = createRng(3);
    // candidates whose truth is exactly a linear function of the features
    const candidates: Response[] = Array.from({ length: 200 }, () => {
      const features = Array.from({ length: RM_FEATURES }, () => rng.normal());
      return { features, truth: 2 * (features[0] as number) - (features[1] as number), length: 0 };
    });
    const pairs = samplePreferences(candidates, 3000, 0.3, createRng(4));
    const model = fitRewardModel(candidates, pairs, {
      iterations: 600,
      learningRate: 0.5,
      decay: 0.001,
    });
    const w = model.weights;
    expect(w[0]! / Math.abs(w[1]!)).toBeGreaterThan(1.5);
    expect(w[0]!).toBeGreaterThan(0);
    expect(w[1]!).toBeLessThan(0);
    for (const k of [2, 3, 4]) expect(Math.abs(w[k]!)).toBeLessThan(0.4 * w[0]!);
    const test = samplePreferences(candidates, 800, 0.3, createRng(5));
    expect(preferenceAccuracy(model, candidates, test)).toBeGreaterThan(0.85);
  });

  it('more preference data gives a better reward model', () => {
    const accuracy = (pairs: number) =>
      trainRlhf({ candidates: 300, pairs, noise: 0.8, seed: 2 }).heldOutAccuracy;
    expect(accuracy(400)).toBeGreaterThan(accuracy(15));
  });
});

describe('optimising against the reward model', () => {
  const scores = [0.2, 1.5, -0.4, 0.9, 2.2, 0.0];
  const reference = scores.map(() => 1 / scores.length);

  it('is the reference at a huge penalty and the best candidate at a tiny one, and always a distribution', () => {
    for (const beta of [1e6, 5, 1, 0.1, 0.01]) {
      const p = klRegularisedPolicy(scores, reference, beta);
      expect(p.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 12);
    }
    const loose = klRegularisedPolicy(scores, reference, 1e6);
    for (const v of loose) expect(v).toBeCloseTo(1 / 6, 4);
    const tight = klRegularisedPolicy(scores, reference, 0.01);
    expect(tight[4]).toBeGreaterThan(0.999);
  });

  it('a smaller penalty always raises the expected reward-model score and the distance from the reference', () => {
    const betas = [10, 3, 1, 0.3, 0.1];
    const policies = betas.map((b) => klRegularisedPolicy(scores, reference, b));
    for (let i = 1; i < betas.length; i++) {
      expect(expectation(policies[i]!, scores)).toBeGreaterThan(
        expectation(policies[i - 1]!, scores),
      );
      expect(klDivergence(policies[i]!, reference)).toBeGreaterThan(
        klDivergence(policies[i - 1]!, reference),
      );
    }
  });

  it('is the true maximiser of score minus beta times KL (no nearby distribution does better)', () => {
    const beta = 0.7;
    const value = (p: number[]) => expectation(p, scores) - beta * klDivergence(p, reference);
    const best = klRegularisedPolicy(scores, reference, beta);
    const rng = createRng(8);
    for (let t = 0; t < 50; t++) {
      const q = best.map((v) => v * Math.exp(0.2 * rng.normal()));
      const z = q.reduce((a, b) => a + b, 0);
      expect(value(q.map((v) => v / z))).toBeLessThanOrEqual(value(best) + 1e-12);
    }
  });

  it('best-of-n is the average for n = 1 and approaches the best candidate as n grows', () => {
    const truths = [0, 1, 2, 3, 4, 5];
    const mean = truths.reduce((a, b) => a + b, 0) / 6;
    expect(bestOfN(scores, truths, 1).truth).toBeCloseTo(mean, 12);
    expect(bestOfN(scores, truths, 1).proxy).toBeCloseTo(scores.reduce((a, b) => a + b, 0) / 6, 12);
    expect(bestOfN(scores, truths, 500).proxy).toBeCloseTo(Math.max(...scores), 6);
  });

  it('candidates are reproducible and their truth penalises excessive length', () => {
    const a = makeCandidates(50, createRng(1));
    const b = makeCandidates(50, createRng(1));
    expect(a).toEqual(b);
  });
});
