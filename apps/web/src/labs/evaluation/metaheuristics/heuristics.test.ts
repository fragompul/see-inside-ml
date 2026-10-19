import { describe, expect, it } from 'vitest';
import { DEFAULT_HEURISTICS, RACE_REPEATS, runHeuristics } from './heuristicsCore';
import { heuristicsExperiments } from './heuristicsSpec';

describe('metaheuristics lab', () => {
  for (const exp of heuristicsExperiments) {
    it(`experiment "${exp.title}" holds`, () => {
      expect(exp.claim(runHeuristics({ ...DEFAULT_HEURISTICS, ...exp.scenario }))).toBe(true);
    }, 60_000);
  }

  it('the race covers the four methods with whole-number success counts', () => {
    const r = runHeuristics({ ...DEFAULT_HEURISTICS, objective: 'sphere', generations: 10 });
    expect(r.race.map((e) => e.method)).toEqual(['genetic', 'annealing', 'swarm', 'cmaes']);
    for (const e of r.race)
      expect(e.success * RACE_REPEATS).toBeCloseTo(Math.round(e.success * RACE_REPEATS), 9);
    expect(r.final).toBe(r.history[r.history.length - 1]);
  });
});
