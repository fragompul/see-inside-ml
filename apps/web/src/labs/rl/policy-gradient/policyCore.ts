import {
  DEFAULT_POLICY,
  GRID_PRESETS,
  GridWorld,
  parseGrid,
  trainReinforce,
  valueIteration,
  type Baseline,
  type PolicyTraining,
} from '@sim/deep';

export type LayoutId = 'open' | 'cliff' | 'maze' | 'detour';
export const BASELINES: Baseline[] = ['none', 'average', 'value'];

export const BASELINE_LABELS: Record<Baseline, string> = {
  none: 'No baseline',
  average: 'Average-return baseline',
  value: 'Learned value baseline',
};

export interface PolicyScenario {
  layout: LayoutId;
  slip: number;
  baseline: Baseline;
  learningRate: number;
  entropy: number;
  batch: number;
  episodes: number;
  seed: number;
}

export const DEFAULT_POLICY_LAB: PolicyScenario = {
  layout: 'open',
  slip: 0,
  baseline: 'value',
  learningRate: 0.1,
  entropy: 0,
  batch: 1,
  episodes: 400,
  seed: 3,
};

export const SEEDS = 6;

/** Rewards scaled so that every return lies between -1 and 1: policy gradients are happier with small numbers. */
export const PG_REWARDS = { step: -0.02, goal: 1, pit: -1 };

export interface PolicyRun {
  scenario: PolicyScenario;
  world: GridWorld;
  training: PolicyTraining;
  /** Return of an optimal policy from the start (undiscounted, no slip noise accounted for). */
  optimal: number;
  /** Mean return and goal rate over the last 100 episodes of the shown run. */
  finalReturn: number;
  goalRate: number;
  /** Average learning curves (moving average of 20 episodes) over several seeds, per baseline. */
  curves: Record<Baseline, number[]>;
  /** Mean and spread of the last-100 return across seeds, per baseline. */
  summary: Record<Baseline, { mean: number; sd: number }>;
}

export const smooth = (v: number[], w = 20): number[] =>
  v.map((_, i) => {
    const from = Math.max(0, i - w + 1);
    const slice = v.slice(from, i + 1);
    return slice.reduce((a, b) => a + b, 0) / slice.length;
  });

export function makeWorld(layout: LayoutId, slip: number): GridWorld {
  return new GridWorld(
    parseGrid((GRID_PRESETS[layout] as { rows: string[] }).rows),
    PG_REWARDS,
    slip,
  );
}

const optionsOf = (s: PolicyScenario, baseline: Baseline, seed: number) => ({
  ...DEFAULT_POLICY,
  baseline,
  learningRate: s.learningRate,
  entropy: s.entropy,
  batch: s.batch,
  maxSteps: 80,
  seed,
});

const tail = (v: number[], n = 100): number =>
  v.slice(-n).reduce((a, b) => a + b, 0) / Math.min(n, v.length);

export function runPolicy(s: PolicyScenario): PolicyRun {
  const world = makeWorld(s.layout, s.slip);
  const training = trainReinforce(world, optionsOf(s, s.baseline, s.seed), s.episodes);
  const optimal = valueIteration(world, 1).values[world.start] as number;
  const curves = {} as Record<Baseline, number[]>;
  const summary = {} as Record<Baseline, { mean: number; sd: number }>;
  for (const b of BASELINES) {
    const runs = Array.from(
      { length: SEEDS },
      (_, r) => trainReinforce(world, optionsOf(s, b, 100 + r), s.episodes).returns,
    );
    curves[b] = smooth(
      Array.from(
        { length: s.episodes },
        (_, i) => runs.reduce((a, rr) => a + (rr[i] as number), 0) / SEEDS,
      ),
    );
    const finals = runs.map((rr) => tail(rr));
    const m = finals.reduce((a, c) => a + c, 0) / SEEDS;
    summary[b] = { mean: m, sd: Math.sqrt(finals.reduce((a, c) => a + (c - m) ** 2, 0) / SEEDS) };
  }
  const outcomes = training.outcomes.slice(-100);
  return {
    scenario: s,
    world,
    training,
    optimal,
    finalReturn: tail(training.returns),
    goalRate: outcomes.filter((o) => o === 'goal').length / outcomes.length,
    curves,
    summary,
  };
}
