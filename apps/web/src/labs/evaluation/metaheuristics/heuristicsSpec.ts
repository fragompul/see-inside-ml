import { HEURISTICS, OBJECTIVES } from '@sim/core';
import type { ParamSpec } from '@/shared/lab-kit';
import {
  METHOD_LABELS,
  raceOf,
  type HeuristicsRun,
  type HeuristicsScenario,
  type ObjectiveId,
} from './heuristicsCore';

export const heuristicsParams: ParamSpec[] = [
  {
    key: 'objective',
    label: 'Terrain',
    kind: 'select',
    options: (['sphere', 'ellipsoid', 'rastrigin', 'ackley', 'deceptive'] as ObjectiveId[]).map(
      (id) => ({
        value: id,
        label: OBJECTIVES[id]!.label,
      }),
    ),
  },
  {
    key: 'method',
    label: 'Method shown',
    kind: 'segmented',
    options: HEURISTICS.map((m) => ({ value: m, label: METHOD_LABELS[m] })),
  },
  {
    key: 'population',
    label: 'Population (points per generation)',
    kind: 'ladder',
    values: [6, 12, 20, 60],
    format: (v) => String(v),
  },
  {
    key: 'generations',
    label: 'Generations',
    kind: 'ladder',
    values: [20, 40, 60, 100],
    format: (v) => String(v),
  },
];

export interface HeuristicsExperiment {
  id: string;
  title: string;
  text: string;
  scenario: Partial<HeuristicsScenario>;
  claim: (run: HeuristicsRun) => boolean;
}

export const heuristicsExperiments: HeuristicsExperiment[] = [
  {
    id: 'easy',
    title: 'The easy bowl',
    text: 'On a smooth bowl every method finds the minimum in every run. Population-based search is overkill here; a gradient would do.',
    scenario: { objective: 'sphere' },
    claim: (r) => r.race.every((e) => e.success === 1),
  },
  {
    id: 'tilted',
    title: 'A tilted, elongated valley',
    text: "The useful moves are along a diagonal valley a hundred times narrower than it is long. CMA-ES learns the valley's orientation and scale from its own samples and reaches essentially exact answers; the others get close but not nearly as precisely.",
    scenario: { objective: 'ellipsoid' },
    claim: (r) =>
      raceOf(r, 'cmaes').medianBest < 1e-12 &&
      raceOf(r, 'swarm').medianBest > 1e-8 &&
      raceOf(r, 'genetic').medianBest > 1e-8 &&
      raceOf(r, 'annealing').success < 0.8,
  },
  {
    id: 'pits',
    title: 'A field of pits',
    text: 'Rastrigin has about a hundred local minima. The genetic algorithm and the swarm keep a diverse population and mostly find the global one; annealing and a small CMA-ES population settle into a pit in most runs.',
    scenario: { objective: 'rastrigin' },
    claim: (r) =>
      raceOf(r, 'genetic').success >= 0.8 &&
      raceOf(r, 'swarm').success >= 0.7 &&
      raceOf(r, 'annealing').success <= 0.4 &&
      raceOf(r, 'cmaes').success <= 0.6,
  },
  {
    id: 'big-population',
    title: 'A bigger population smooths out the pits',
    text: 'CMA-ES samples a Gaussian, and with 60 samples per generation that Gaussian averages over many pits and follows the overall bowl. With the same method and a bigger population the success rate climbs sharply.',
    scenario: { objective: 'rastrigin', population: 60, generations: 100 },
    claim: (r) => raceOf(r, 'cmaes').success >= 0.8 && raceOf(r, 'annealing').success >= 0.7,
  },
  {
    id: 'funnel',
    title: 'A funnel on a plain',
    text: 'Ackley is almost flat far from the centre with a single deep funnel. All methods find it; CMA-ES gets by far the most precise answer.',
    scenario: { objective: 'ackley' },
    claim: (r) => r.race.every((e) => e.success >= 0.9) && raceOf(r, 'cmaes').medianBest < 1e-8,
  },
  {
    id: 'decoy',
    title: 'A decoy basin',
    text: 'A wide, shallow basin and, far away, a narrow and deeper one. With 1,200 evaluations nobody does well: almost every run settles in the wide basin, whose fall is easy to see from afar.',
    scenario: { objective: 'deceptive' },
    claim: (r) => r.race.every((e) => e.success <= 0.25),
  },
  {
    id: 'patient-annealing',
    title: 'Patience pays for annealing',
    text: 'Give the same decoy terrain five times the budget: slow cooling lets simulated annealing wander across both basins before it commits, and most of its runs find the narrow, deeper one. The population methods have already converged on the decoy.',
    scenario: { objective: 'deceptive', population: 60, generations: 100 },
    claim: (r) =>
      raceOf(r, 'annealing').success >= 0.7 &&
      raceOf(r, 'cmaes').success <= 0.1 &&
      raceOf(r, 'swarm').success <= 0.3,
  },
  {
    id: 'tiny-budget',
    title: 'A starved budget',
    text: 'With six points for twenty generations (about 120 evaluations) none of the methods can cover the field of pits: almost every run fails.',
    scenario: { objective: 'rastrigin', population: 6, generations: 20 },
    claim: (r) => r.race.every((e) => e.success <= 0.1),
  },
];
