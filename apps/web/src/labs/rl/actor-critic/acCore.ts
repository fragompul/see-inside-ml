import {
  DEFAULT_ACTOR_CRITIC,
  GRID_PRESETS,
  GridWorld,
  gradientSpread,
  parseGrid,
  trainActorCritic,
  valueIteration,
  type ActorCriticTraining,
} from '@sim/deep';

export type AcLayout = 'open' | 'cliff' | 'maze';
export const LAMBDAS = [0, 0.5, 0.9, 1];

export interface AcScenario {
  layout: AcLayout;
  slip: number;
  /** Trace decay of the shown run (the race always compares all of LAMBDAS). */
  lambda: number;
  actorRate: number;
  episodes: number;
  seed: number;
}

export const DEFAULT_AC: AcScenario = {
  layout: 'open',
  slip: 0,
  lambda: 0.9,
  actorRate: 0.1,
  episodes: 300,
  seed: 3,
};

export const SEEDS = 6;
export const PG_REWARDS = { step: -0.02, goal: 1, pit: -1 };

export interface AcRun {
  scenario: AcScenario;
  world: GridWorld;
  single: ActorCriticTraining;
  optimal: number;
  /** Average learning curves (moving average of 20 episodes over seeds) for every lambda. */
  curves: Record<string, number[]>;
  /** Mean return over the last 50 episodes, averaged over seeds, per lambda. */
  final: Record<string, number>;
  /** Episodes until the smoothed average return first passes 70% of the optimal return, per lambda (Infinity if never). */
  speed: Record<string, number>;
  /** Variance of the gradient estimate and distance of its mean from the lambda = 1 mean, per lambda, at one fixed mid-training agent. */
  variance: Record<string, number>;
  bias: Record<string, number>;
}

export const makeWorld = (layout: AcLayout, slip: number): GridWorld =>
  new GridWorld(parseGrid((GRID_PRESETS[layout] as { rows: string[] }).rows), PG_REWARDS, slip);

const smooth = (v: number[], w = 20): number[] =>
  v.map((_, i) => {
    const part = v.slice(Math.max(0, i - w + 1), i + 1);
    return part.reduce((a, b) => a + b, 0) / part.length;
  });

const options = (s: AcScenario, lambda: number, seed: number) => ({
  ...DEFAULT_ACTOR_CRITIC,
  lambda,
  actorRate: s.actorRate,
  seed,
});

export function runAc(s: AcScenario): AcRun {
  const world = makeWorld(s.layout, s.slip);
  const single = trainActorCritic(world, options(s, s.lambda, s.seed), s.episodes);
  const optimal = valueIteration(world, 1).values[world.start] as number;
  const curves: Record<string, number[]> = {};
  const final: Record<string, number> = {};
  const speed: Record<string, number> = {};
  for (const lambda of LAMBDAS) {
    const runs = Array.from(
      { length: SEEDS },
      (_, r) => trainActorCritic(world, options(s, lambda, 100 + r), s.episodes).returns,
    );
    const mean = Array.from(
      { length: s.episodes },
      (_, i) => runs.reduce((a, rr) => a + (rr[i] as number), 0) / SEEDS,
    );
    const curve = smooth(mean);
    curves[String(lambda)] = curve;
    final[String(lambda)] = mean.slice(-50).reduce((a, b) => a + b, 0) / 50;
    const first = curve.findIndex((v) => v >= 0.7 * optimal);
    speed[String(lambda)] = first < 0 ? Infinity : first + 1;
  }
  // gradient quality at a fixed agent that has trained for a while, with a critic that is still imperfect
  const probe = trainActorCritic(world, options(s, 0.9, 7), 60).agent;
  const reference = gradientSpread(probe, 1, 800).mean;
  const variance: Record<string, number> = {};
  const bias: Record<string, number> = {};
  for (const lambda of LAMBDAS) {
    const spread = gradientSpread(probe, lambda, 800);
    variance[String(lambda)] = spread.variance;
    bias[String(lambda)] = Math.sqrt(
      spread.mean.reduce((a, v, i) => a + (v - (reference[i] as number)) ** 2, 0),
    );
  }
  return { scenario: s, world, single, optimal, curves, final, speed, variance, bias };
}
