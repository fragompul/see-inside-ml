import { describe, expect, it } from 'vitest';
import { DEFAULT_METRICS, runMetrics } from './metricsCore';
import { metricsExperiments } from './metricsSpec';

describe('metrics lab', () => {
  for (const exp of metricsExperiments) {
    it(`experiment "${exp.title}" holds`, () => {
      expect(exp.claim(runMetrics({ ...DEFAULT_METRICS, ...exp.scenario }))).toBe(true);
    });
  }

  it('the confusion counts add up to the test set and the curves end where they should', () => {
    const r = runMetrics(DEFAULT_METRICS);
    expect(r.at.tp + r.at.fp + r.at.tn + r.at.fn).toBe(r.scores.length);
    expect(r.at.tp + r.at.fn).toBe(r.positives);
    expect(r.roc.fpr[0]).toBe(0);
    expect(r.roc.tpr[r.roc.tpr.length - 1]).toBe(1);
    expect(r.pr.recall[r.pr.recall.length - 1]).toBe(1);
  });

  it('the best threshold is never worse than the current one', () => {
    const r = runMetrics({ ...DEFAULT_METRICS, costRatio: 10 });
    expect(r.best.cost).toBeLessThanOrEqual(r.cost);
  });
});
