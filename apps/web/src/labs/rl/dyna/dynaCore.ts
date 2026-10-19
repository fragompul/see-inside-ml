import {
  DEFAULT_DYNA,
  DEFAULT_REWARDS,
  GRID_PRESETS,
  GridWorld,
  averageSteps,
  parseGrid,
  trainDyna,
  valueIteration,
  type DynaRun,
} from '@sim/deep';

export type DynaLayout = 'open' | 'maze' | 'detour';

export interface DynaScenario {
  layout: DynaLayout;
  planning: number;
  slip: number;
  episodes: number;
  epsilon: number;
  seed: number;
}

export const DEFAULT_DYNA_LAB: DynaScenario = {
  layout: 'maze',
  planning: 0,
  slip: 0,
  episodes: 30,
  epsilon: 0.1,
  seed: 2,
};

export const RUNS = 20;

export interface DynaLabRun {
  scenario: DynaScenario;
  world: GridWorld;
  single: DynaRun;
  /** Average steps per episode with the chosen amount of planning, and with none, for comparison. */
  curve: number[];
  baseline: number[];
  /** Real steps used over all episodes (average), for the chosen planning and for none. */
  realSteps: number;
  baselineSteps: number;
  /** Shortest path length from the start. */
  shortest: number;
}

export const makeWorld = (layout: DynaLayout, slip: number): GridWorld =>
  new GridWorld(
    parseGrid((GRID_PRESETS[layout] as { rows: string[] }).rows),
    DEFAULT_REWARDS,
    slip,
  );

export function runDyna(s: DynaScenario): DynaLabRun {
  const world = makeWorld(s.layout, s.slip);
  const options = {
    ...DEFAULT_DYNA,
    planning: s.planning,
    episodes: s.episodes,
    epsilon: s.epsilon,
    seed: s.seed,
  };
  const single = trainDyna(world, options);
  const curve = averageSteps(world, options, RUNS);
  const baseline =
    s.planning === 0 ? curve : averageSteps(world, { ...options, planning: 0 }, RUNS);
  const sum = (v: number[]) => v.reduce((a, b) => a + b, 0);
  // shortest path by following the optimal actions
  const opt = valueIteration(makeWorld(s.layout, 0), DEFAULT_DYNA.gamma);
  const plain = makeWorld(s.layout, 0);
  let t = plain.start;
  let shortest = 0;
  while (!plain.isTerminal(t) && shortest < 200) {
    const q = Array.from({ length: 4 }, (_, a) => opt.q[t * 4 + a] as number);
    t = plain.move(t, q.indexOf(Math.max(...q)));
    shortest++;
  }
  return {
    scenario: s,
    world,
    single,
    curve,
    baseline,
    realSteps: sum(curve),
    baselineSteps: sum(baseline),
    shortest,
  };
}

/** First episode (1-based) at which the average steps fall within 2 of the shortest path and stay there for 3 episodes. */
export function episodesToLearn(curve: number[], shortest: number): number {
  for (let i = 0; i + 2 < curve.length; i++)
    if (curve.slice(i, i + 3).every((v) => v <= shortest + 2 + 0.5)) return i + 1;
  return Infinity;
}
