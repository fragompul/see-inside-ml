import {
  LANDSCAPES,
  runSearch,
  type Landscape,
  type SearchMethod,
  type SearchResult,
} from '@sim/classic';

export type LandscapeId = keyof typeof LANDSCAPES;

export interface TuningScenario {
  landscape: LandscapeId;
  method: SearchMethod;
  /** Budget in full experiments. */
  budget: number;
  noise: number;
  seed: number;
}

export const DEFAULT_TUNING: TuningScenario = {
  landscape: 'bowl',
  method: 'bayesian',
  budget: 20,
  noise: 0.05,
  seed: 3,
};

export const METHODS: SearchMethod[] = ['grid', 'random', 'halving', 'bayesian'];

export const METHOD_LABELS: Record<SearchMethod, string> = {
  grid: 'Grid search',
  random: 'Random search',
  halving: 'Successive halving',
  bayesian: 'Bayesian optimisation',
};

export interface RaceEntry {
  method: SearchMethod;
  /** Average regret over many independent repetitions (lower is better). */
  meanRegret: number;
  /** Share of repetitions that ended within 0.05 of the best achievable score. */
  success: number;
}

export interface TuningRun {
  scenario: TuningScenario;
  landscape: Landscape;
  single: SearchResult;
  race: RaceEntry[];
}

export const RACE_REPEATS = 16;

export function runTuning(s: TuningScenario): TuningRun {
  const land = LANDSCAPES[s.landscape] as Landscape;
  const single = runSearch(s.method, land, s.budget, s.noise, s.seed);
  const race = METHODS.map((method) => {
    const regrets = Array.from(
      { length: RACE_REPEATS },
      (_, r) => runSearch(method, land, s.budget, s.noise, 100 + r).regret,
    );
    return {
      method,
      meanRegret: regrets.reduce((a, b) => a + b, 0) / regrets.length,
      success: regrets.filter((v) => v < 0.05).length / regrets.length,
    };
  });
  return { scenario: s, landscape: land, single, race };
}

export const raceOf = (run: TuningRun, method: SearchMethod): RaceEntry =>
  run.race.find((r) => r.method === method) as RaceEntry;
