import { describe, expect, it } from 'vitest';
import { DEFAULT_MCTS, POSITIONS, runMcts } from './mctsCore';
import { mctsExperiments } from './mctsSpec';

describe('MCTS lab', () => {
  for (const exp of mctsExperiments) {
    it(`experiment "${exp.title}" holds`, () => {
      expect(exp.claim(runMcts({ ...DEFAULT_MCTS, ...exp.scenario }))).toBe(true);
    }, 300_000);
  }

  it('each preset has at least one optimal move and the tournaments add up to 40 or 20 games', () => {
    for (const id of Object.keys(POSITIONS) as (keyof typeof POSITIONS)[]) {
      const r = runMcts({ ...DEFAULT_MCTS, position: id, simulations: 20 });
      expect(r.optimal.length).toBeGreaterThan(0);
      expect(r.vsRandom.wins + r.vsRandom.draws + r.vsRandom.losses).toBe(40);
      expect(r.vsPerfect.wins + r.vsPerfect.draws + r.vsPerfect.losses).toBe(20);
    }
  });
});
