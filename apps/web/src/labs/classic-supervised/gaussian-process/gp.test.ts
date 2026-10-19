import { describe, expect, it } from 'vitest';
import { runRegressorExperiment } from '@/shared/lab-kit';
import { gpOf, gpSpec } from './gpSpec';

const run = (id: string) =>
  runRegressorExperiment(
    gpSpec,
    gpSpec.experiments.find((e) => e.id === id)!,
  );

describe('Gaussian process lab', () => {
  for (const exp of gpSpec.experiments) {
    it(`experiment "${exp.title}" holds`, () => {
      expect(exp.claim(runRegressorExperiment(gpSpec, exp))).toBe(true);
    });
  }

  it('optimising the evidence beats both extreme length scales on the marginal likelihood', () => {
    const tuned = gpOf(run('tuned').model)!.likelihood;
    expect(tuned).toBeGreaterThan(gpOf(run('short').model)!.likelihood);
    expect(tuned).toBeGreaterThan(gpOf(run('long').model)!.likelihood);
  });
});
