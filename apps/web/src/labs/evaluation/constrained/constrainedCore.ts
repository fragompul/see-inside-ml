import {
  CONSTRAINED_PROBLEMS,
  distanceToSolution,
  kktResidual,
  solveConstrained,
  type ConstrainedMethod,
  type ConstrainedProblem,
  type OuterStep,
} from '@sim/core';

export interface ConstrainedScenario {
  problem: string;
  method: ConstrainedMethod;
  outer: number;
  /** First penalty weight (penalty, augmented) or barrier weight. */
  weight: number;
  /** Growth of the penalty weight or shrink of the barrier weight per stage. */
  factor: number;
}

export const DEFAULT_CONSTRAINED: ConstrainedScenario = {
  problem: 'disk',
  method: 'augmented',
  outer: 6,
  weight: 10,
  factor: 10,
};

export const GRID = 56;

export interface ConstrainedResult {
  scenario: ConstrainedScenario;
  problem: { id: string; label: string; note: string; constraints: string[] };
  /** The chosen method's stages and, for each stage, the distance from the true solution. */
  steps: OuterStep[];
  error: number[];
  /** The same measure for each method on this problem (null where the method does not apply). */
  compare: Record<ConstrainedMethod, number[] | null>;
  /** Values of f on a GRID x GRID window (row 0 at the top), and 1 where every constraint holds. */
  surface: number[];
  feasible: number[];
  solution: { x: [number, number]; multipliers: number[] };
  /** KKT residual at the last stage with its multiplier estimates. */
  kkt: number;
  start: [number, number];
  bounds: ConstrainedProblem['bounds'];
}

const barrierApplies = (p: ConstrainedProblem) => p.constraints.every((c) => c.kind === 'ineq');

const cache = new Map<string, ConstrainedResult>();

export function runConstrained(s: ConstrainedScenario): ConstrainedResult {
  const key = JSON.stringify(s);
  const hit = cache.get(key);
  if (hit) return hit;
  const p = CONSTRAINED_PROBLEMS[s.problem] as ConstrainedProblem;
  const applicable = s.method !== 'barrier' || barrierApplies(p);
  const steps = applicable ? solveConstrained(p, s) : [];
  const defaults = (m: ConstrainedMethod) =>
    m === 'penalty'
      ? { method: m, outer: 8, weight: 1, factor: 10 }
      : m === 'augmented'
        ? { method: m, outer: 8, weight: 10, factor: 1 }
        : { method: m, outer: 8, weight: 1, factor: 10 };
  const compare = {} as Record<ConstrainedMethod, number[] | null>;
  for (const m of ['penalty', 'augmented', 'barrier'] as ConstrainedMethod[]) {
    compare[m] =
      m === 'barrier' && !barrierApplies(p)
        ? null
        : solveConstrained(p, defaults(m)).map((st) => distanceToSolution(p, st.x));
  }
  const { x0, x1, y0, y1 } = p.bounds;
  const surface: number[] = [];
  const feasible: number[] = [];
  for (let r = 0; r < GRID; r++)
    for (let c = 0; c < GRID; c++) {
      const x: [number, number] = [
        x0 + ((c + 0.5) / GRID) * (x1 - x0),
        y1 - ((r + 0.5) / GRID) * (y1 - y0),
      ];
      surface.push(p.f(x));
      feasible.push(
        p.constraints.every((k) =>
          k.kind === 'eq' ? Math.abs(k.value(x)) < 0.04 : k.value(x) <= 0,
        )
          ? 1
          : 0,
      );
    }
  const last = steps[steps.length - 1];
  const out: ConstrainedResult = {
    scenario: s,
    problem: {
      id: p.id,
      label: p.label,
      note: p.note,
      constraints: p.constraints.map((c) => c.label),
    },
    steps,
    error: steps.map((st) => distanceToSolution(p, st.x)),
    compare,
    surface,
    feasible,
    solution: p.solution,
    kkt: last ? kktResidual(p, last.x, last.multipliers) : Number.NaN,
    start: p.start,
    bounds: p.bounds,
  };
  cache.set(key, out);
  return out;
}
