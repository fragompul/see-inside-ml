import { describe, expect, it } from 'vitest';
import { SHOWDOWN_MODELS, experimentSetup, runShowdown, showdownExperiments } from './showdown';

describe('model showdown', () => {
  for (const exp of showdownExperiments) {
    it(`experiment "${exp.title}" holds`, () => {
      expect(exp.claim(runShowdown(experimentSetup(exp)))).toBe(true);
    });
  }

  it('every model runs on every dataset and reports sane numbers', () => {
    const r = runShowdown(experimentSetup(showdownExperiments[0]!));
    expect(r.rows).toHaveLength(SHOWDOWN_MODELS.length);
    for (const row of r.rows) {
      expect(row.trainAccuracy).toBeGreaterThan(0.4);
      expect(row.testAccuracy).not.toBeNull();
      expect(row.regions.labels.length).toBe(row.regions.n * row.regions.n);
    }
  });

  it('no single model wins on every dataset', () => {
    const winners = new Set(showdownExperiments.map((e) => runShowdown(experimentSetup(e)).winner));
    expect(winners.size).toBeGreaterThan(1);
  });

  it('is deterministic for a seed', () => {
    const exp = showdownExperiments[1]!;
    const a = runShowdown(experimentSetup(exp));
    const b = runShowdown(experimentSetup(exp));
    expect(a.rows.map((r) => r.testAccuracy)).toEqual(b.rows.map((r) => r.testAccuracy));
  });
});
