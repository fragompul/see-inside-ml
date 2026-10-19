import { describe, expect, it } from 'vitest';
import { DEFAULT_PAGERANK, WEBS, runPageRank, type WebId } from './pagerankCore';
import { pagerankExperiments } from './pagerankSpec';

describe('PageRank lab', () => {
  for (const exp of pagerankExperiments) {
    it(`experiment "${exp.title}" holds`, () => {
      expect(exp.claim(runPageRank({ ...DEFAULT_PAGERANK, ...exp.scenario }))).toBe(true);
    });
  }

  it.each(Object.keys(WEBS) as WebId[])('%s: ranks add to one and are positive', (web) => {
    const r = runPageRank({ ...DEFAULT_PAGERANK, web });
    expect(r.exact.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 12);
    for (const v of r.exact) expect(v).toBeGreaterThan(0);
    expect(r.shown).toHaveLength(8);
  });
});
