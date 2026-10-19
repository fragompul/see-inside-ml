import { describe, expect, it } from 'vitest';
import { DEFAULT_ASSOCIATION, ITEMS, makeBaskets, runAssociation } from './associationCore';
import { associationExperiments } from './associationSpec';

describe('association lab', () => {
  for (const exp of associationExperiments) {
    it(`experiment "${exp.title}" holds`, () => {
      expect(exp.claim(runAssociation({ ...DEFAULT_ASSOCIATION, ...exp.scenario }))).toBe(true);
    });
  }

  it('baskets are deterministic, non-empty, sorted and use known items', () => {
    const a = makeBaskets(100, 3);
    expect(a).toEqual(makeBaskets(100, 3));
    for (const b of a) {
      expect(b.length).toBeGreaterThan(0);
      expect([...b].sort((x, y) => x - y)).toEqual(b);
      for (const i of b) expect(i).toBeLessThan(ITEMS.length);
    }
  });

  it('with a lot of data the planted lifts approach their true values', () => {
    const r = runAssociation({ ...DEFAULT_ASSOCIATION, baskets: 20000, minSupport: 0.02 });
    const lift = (a: number, c: number) =>
      r.rules.find(
        (x) => x.antecedent[0] === a && x.consequent[0] === c && x.antecedent.length === 1,
      )!.lift;
    // P(butter | bread) = 0.75 and P(butter) = 0.45 * 0.75 + 0.55 * 0.1 = 0.3925
    expect(lift(0, 1)).toBeCloseTo(0.75 / 0.3925, 1);
  });
});
