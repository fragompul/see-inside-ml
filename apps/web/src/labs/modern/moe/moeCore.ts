import { createRng, type Rng } from '@sim/core';
import type { Moe } from '@sim/deep';
import { DEFAULT_MOE_TRAINING, trainMoe } from '@sim/deep';

/**
 * Four regions of the square [-1, 1]^2, each with its own function to learn: the kind of problem where
 * specialised experts should help.
 */
export const REGIONS = 4;
export const regionOf = (x1: number, x2: number): number => (x1 < 0 ? 0 : 1) + (x2 < 0 ? 0 : 2);

const target = (x1: number, x2: number): number => {
  switch (regionOf(x1, x2)) {
    case 0:
      return Math.sin(4 * x1) + 0.5 * x2;
    case 1:
      return 2 * x1 * x2 - 0.5;
    case 2:
      return Math.cos(5 * x2) - x1;
    default:
      return 1.5 - 3 * Math.abs(x1) + x2;
  }
};

const sample = (rng: Rng) => {
  const x1 = rng.next() * 2 - 1;
  const x2 = rng.next() * 2 - 1;
  return { x: [x1, x2] as [number, number], t: target(x1, x2) };
};

export type Design = 'dense-small' | 'dense-big' | 'moe-1' | 'moe-2' | 'moe-unbalanced';

export const DESIGN_LABEL: Record<Design, string> = {
  'dense-small': 'One network, 6 hidden units',
  'dense-big': 'One network, 24 hidden units',
  'moe-1': 'Four experts, one runs',
  'moe-2': 'Four experts, two run',
  'moe-unbalanced': 'Four experts, one runs, no balance term',
};

export interface MoeScenario {
  design: Design;
  steps: number;
  seed: number;
}

export const DEFAULT_MOE_LAB: MoeScenario = { design: 'moe-1', steps: 1500, seed: 1 };

export interface MoeResult {
  scenario: MoeScenario;
  /** Test error per design (the chosen one first used for the page, all of them for the comparison). */
  mse: Record<Design, number>;
  /** Parameters in total and parameters used for one input, per design. */
  total: Record<Design, number>;
  active: Record<Design, number>;
  /** For the chosen design: share of test inputs sent to each expert, and for each region the share going to its favourite expert. */
  share: number[];
  purity: number[];
  /** Which expert is the favourite of each region. */
  favourite: number[];
  entropy: number;
  /** A grid of which expert handles each point (index -1 for dense designs), GRID x GRID, row 0 at the top. */
  map: number[];
  history: { step: number; mse: number; share: number[] }[];
}

export const GRID = 24;

const testSet = (() => {
  const rng = createRng(777);
  const n = 400;
  const xs = new Float64Array(2 * n);
  const ts = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const s = sample(rng);
    xs[2 * i] = s.x[0];
    xs[2 * i + 1] = s.x[1];
    ts[i] = s.t;
  }
  return { xs, ts };
})();

const CONFIG: Record<Design, { experts: number; hidden: number; top: number; balance: number }> = {
  'dense-small': { experts: 1, hidden: 6, top: 1, balance: 0 },
  'dense-big': { experts: 1, hidden: 24, top: 1, balance: 0 },
  'moe-1': { experts: 4, hidden: 6, top: 1, balance: 0.1 },
  'moe-2': { experts: 4, hidden: 6, top: 2, balance: 0.1 },
  'moe-unbalanced': { experts: 4, hidden: 6, top: 1, balance: 0 },
};

const runs = new Map<string, { model: Moe; history: MoeResult['history'] }>();

function train(design: Design, steps: number, seed: number) {
  const key = `${design}|${steps}|${seed}`;
  let hit = runs.get(key);
  if (!hit) {
    const c = CONFIG[design];
    hit = trainMoe(
      { experts: c.experts, hidden: c.hidden, top: c.top },
      { ...DEFAULT_MOE_TRAINING, steps, seed, balance: c.balance },
      sample,
      testSet,
    );
    runs.set(key, hit);
  }
  return hit;
}

const cache = new Map<string, MoeResult>();

export function runMoeLab(s: MoeScenario): MoeResult {
  const key = JSON.stringify(s);
  const hit = cache.get(key);
  if (hit) return hit;
  const designs = Object.keys(CONFIG) as Design[];
  const mse = {} as Record<Design, number>;
  const total = {} as Record<Design, number>;
  const active = {} as Record<Design, number>;
  for (const d of designs) {
    const r = train(d, s.steps, s.seed);
    mse[d] = (r.history[r.history.length - 1] as { mse: number }).mse;
    total[d] = r.model.parameterCount;
    active[d] = r.model.activeParameterCount;
  }
  const { model, history } = train(s.design, s.steps, s.seed);
  const E = model.config.experts;
  const share = new Array<number>(E).fill(0);
  const byRegion = Array.from({ length: REGIONS }, () => new Array<number>(E).fill(0));
  const n = testSet.ts.length;
  for (let i = 0; i < n; i++) {
    const x1 = testSet.xs[2 * i] as number;
    const x2 = testSet.xs[2 * i + 1] as number;
    const e = model.route(x1, x2)[0] as number;
    share[e] = (share[e] as number) + 1 / n;
    const row = byRegion[regionOf(x1, x2)] as number[];
    row[e] = (row[e] as number) + 1;
  }
  const favourite = byRegion.map((row) => row.indexOf(Math.max(...row)));
  const purity = byRegion.map((row) => Math.max(...row) / row.reduce((a, b) => a + b, 0));
  const entropy = -share.reduce((a, p) => a + (p > 0 ? p * Math.log(p) : 0), 0);
  const map: number[] = [];
  for (let r = 0; r < GRID; r++)
    for (let c = 0; c < GRID; c++)
      map.push(
        E === 1
          ? -1
          : (model.route(-1 + ((c + 0.5) * 2) / GRID, 1 - ((r + 0.5) * 2) / GRID)[0] as number),
      );
  const out: MoeResult = {
    scenario: s,
    mse,
    total,
    active,
    share,
    purity,
    favourite,
    entropy,
    map,
    history,
  };
  cache.set(key, out);
  return out;
}
