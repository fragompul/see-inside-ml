import { describe, expect, it } from 'vitest';
import {
  runRegressor,
  buildRegression,
  regressionSetup,
  runRegressorExperiment,
} from '@/shared/lab-kit';
import { BayesianPolynomial, bayesSpec, meanStd, plainPolynomial } from './bayesSpec';

describe('Bayesian regression lab', () => {
  for (const exp of bayesSpec.experiments) {
    it(`experiment "${exp.title}" holds`, () => {
      expect(exp.claim(runRegressorExperiment(bayesSpec, exp))).toBe(true);
    });
  }

  it('more data narrows the posterior', () => {
    const run = (id: string) =>
      runRegressorExperiment(
        bayesSpec,
        bayesSpec.experiments.find((e) => e.id === id)!,
      );
    expect(meanStd(run('many').model as BayesianPolynomial, -3, 3)).toBeLessThan(
      meanStd(run('few').model as BayesianPolynomial, -3, 3),
    );
  });

  it('with a flexible model and little data, evidence-chosen shrinkage beats unregularised least squares', () => {
    const data = buildRegression(regressionSetup(bayesSpec, { n: 20 }));
    const bayes = runRegressor(new BayesianPolynomial(12), data);
    const plain = runRegressor(plainPolynomial(12), data);
    expect(bayes.scores.testRmse as number).toBeLessThan(plain.scores.testRmse as number);
  });
});
