import { DEFAULT_DQN, GRID_PRESETS, GridWorld, parseGrid, trainDqn, type DqnRun } from '@sim/deep';

export type DqnLayout = 'open' | 'maze';

export interface DqnScenario {
  layout: DqnLayout;
  episodes: number;
  /** Steps between target-network updates, for the configurations that have one. */
  targetEvery: number;
}

export const DEFAULT_DQN_LAB: DqnScenario = { layout: 'open', episodes: 150, targetEvery: 100 };

export const SEEDS = 4;
export const DQN_REWARDS = { step: -0.04, goal: 1, pit: -1 };

export interface DqnConfig {
  id: string;
  label: string;
  replay: boolean;
  target: boolean;
}

export const CONFIGS: DqnConfig[] = [
  { id: 'full', label: 'Replay and target network', replay: true, target: true },
  { id: 'no-target', label: 'Replay only', replay: true, target: false },
  { id: 'no-replay', label: 'Target network only', replay: false, target: true },
  { id: 'plain', label: 'Neither', replay: false, target: false },
];

export interface ConfigResult {
  config: DqnConfig;
  /** Average return per episode over the seeds, smoothed over 10 episodes. */
  curve: number[];
  /** Largest |Q| seen in each seed's whole run: the true best value is below 1. */
  maxQ: number;
  solved: number;
  /** Mean return over the last 30 episodes, averaged over seeds. */
  tail: number;
}

export const makeWorld = (layout: DqnLayout): GridWorld =>
  new GridWorld(parseGrid((GRID_PRESETS[layout] as { rows: string[] }).rows), DQN_REWARDS, 0);

const smooth = (v: number[], w = 10): number[] =>
  v.map((_, i) => {
    const part = v.slice(Math.max(0, i - w + 1), i + 1);
    return part.reduce((a, b) => a + b, 0) / part.length;
  });

const cache = new Map<string, ConfigResult>();

export function runConfig(s: DqnScenario, config: DqnConfig): ConfigResult {
  const key = `${s.layout}|${s.episodes}|${s.targetEvery}|${config.id}`;
  const hit = cache.get(key);
  if (hit) return hit;
  const world = makeWorld(s.layout);
  const runs: DqnRun[] = Array.from({ length: SEEDS }, (_, i) =>
    trainDqn(world, {
      ...DEFAULT_DQN,
      replay: config.replay,
      targetEvery: config.target ? s.targetEvery : 0,
      episodes: s.episodes,
      seed: 10 + i,
    }),
  );
  const mean = Array.from(
    { length: s.episodes },
    (_, e) => runs.reduce((a, r) => a + (r.returns[e] as number), 0) / SEEDS,
  );
  const out: ConfigResult = {
    config,
    curve: smooth(mean),
    maxQ: Math.max(...runs.map((r) => Math.max(...r.maxQ))),
    solved: runs.filter((r) => r.solved).length,
    tail:
      runs.reduce((a, r) => a + r.returns.slice(-30).reduce((x, y) => x + y, 0) / 30, 0) / SEEDS,
  };
  cache.set(key, out);
  return out;
}

export interface DqnRunAll {
  scenario: DqnScenario;
  results: Record<string, ConfigResult>;
}

export function runDqnLab(s: DqnScenario): DqnRunAll {
  return {
    scenario: s,
    results: Object.fromEntries(CONFIGS.map((c) => [c.id, runConfig(s, c)])),
  };
}
