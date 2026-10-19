import { describe, expect, it } from 'vitest';
import { DEFAULT_RECOMMENDER, ITEMS, USERS, makeWorld, runRecommender } from './recommenderCore';
import { recommenderExperiments } from './recommenderSpec';

describe('recommender lab', () => {
  for (const exp of recommenderExperiments) {
    it(`experiment "${exp.title}" holds`, () => {
      expect(exp.claim(runRecommender({ ...DEFAULT_RECOMMENDER, ...exp.scenario }))).toBe(true);
    });
  }

  it('the world has the advertised size, star values in 1..5 and a genre per film', () => {
    const w = makeWorld(0.3, 5);
    expect(w.stars).toHaveLength(USERS);
    expect(w.stars[0]).toHaveLength(ITEMS);
    for (const row of w.stars)
      for (const v of row) {
        expect(v).toBeGreaterThanOrEqual(1);
        expect(v).toBeLessThanOrEqual(5);
      }
    expect(new Set(w.itemGenre).size).toBe(3);
  });

  it('known and unknown ratings partition the table exactly', () => {
    const r = runRecommender(DEFAULT_RECOMMENDER);
    expect(r.train.length + r.test.length).toBe(USERS * ITEMS);
    const seen = new Set(r.train.map((x) => `${x.user},${x.item}`));
    for (const t of r.test) expect(seen.has(`${t.user},${t.item}`)).toBe(false);
  });

  it('the regularised objective never rises over the iterations (ALS)', () => {
    const r = runRecommender(DEFAULT_RECOMMENDER);
    for (let i = 1; i < r.history.length; i++)
      expect(r.history[i]!.objective).toBeLessThanOrEqual(r.history[i - 1]!.objective + 1e-9);
  });
});
