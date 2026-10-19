import {
  HEURISTICS,
  OBJECTIVES,
  runHeuristic,
  type Generation,
  type Heuristic,
  type Objective,
} from '@sim/core';

export type ObjectiveId = 'sphere' | 'ellipsoid' | 'rastrigin' | 'ackley' | 'deceptive';

export interface HeuristicsScenario {
  objective: ObjectiveId;
  method: Heuristic;
  population: number;
  generations: number;
  seed: number;
}

export const DEFAULT_HEURISTICS: HeuristicsScenario = {
  objective: 'rastrigin',
  method: 'genetic',
  population: 20,
  generations: 60,
  seed: 4,
};

export const METHOD_LABELS: Record<Heuristic, string> = {
  genetic: 'Genetic algorithm',
  annealing: 'Simulated annealing',
  swarm: 'Particle swarm',
  cmaes: 'CMA-ES',
};

export const METHOD_NOTES: Record<Heuristic, string> = {
  genetic:
    'A population evolves: the fitter parents are picked more often, their genes are blended and mutated, and the best always survives.',
  annealing:
    'One walker takes random steps and accepts a worse position with probability exp(-loss / temperature); as it cools it becomes greedy.',
  swarm:
    'Particles fly with momentum, pulled towards the best place each has seen and the best the whole swarm has seen.',
  cmaes:
    'Samples from a Gaussian and moves it towards the best samples while it learns the shape (covariance) and the size (step) of the valley.',
};

export const RACE_REPEATS = 16;

/** A run counts as a success when it ends within this distance (in function value) of the global minimum. */
export const SUCCESS_TOLERANCE = 0.1;

export interface RaceEntry {
  method: Heuristic;
  /** Share of repetitions that found the global minimum. */
  success: number;
  /** Median best value reached. */
  medianBest: number;
  /** Median number of evaluations used. */
  evaluations: number;
}

export interface HeuristicsRun {
  scenario: HeuristicsScenario;
  objective: Objective;
  history: Generation[];
  final: Generation;
  race: RaceEntry[];
}

const median = (v: number[]): number =>
  [...v].sort((a, b) => a - b)[Math.floor(v.length / 2)] as number;

export function runHeuristics(s: HeuristicsScenario): HeuristicsRun {
  const objective = OBJECTIVES[s.objective] as Objective;
  const opts = { population: s.population, generations: s.generations };
  const history = runHeuristic(s.method, objective, { ...opts, seed: s.seed });
  const race = HEURISTICS.map((method) => {
    const runs = Array.from({ length: RACE_REPEATS }, (_, r) =>
      runHeuristic(method, objective, { ...opts, seed: 100 + r }),
    );
    const bests = runs.map((g) => (g[g.length - 1] as Generation).best.f);
    return {
      method,
      success:
        bests.filter((b) => b - objective.optimum.f < SUCCESS_TOLERANCE).length / RACE_REPEATS,
      medianBest: median(bests),
      evaluations: median(runs.map((g) => (g[g.length - 1] as Generation).evaluations)),
    };
  });
  return {
    scenario: s,
    objective,
    history,
    final: history[history.length - 1] as Generation,
    race,
  };
}

export const raceOf = (run: HeuristicsRun, m: Heuristic): RaceEntry =>
  run.race.find((r) => r.method === m) as RaceEntry;
