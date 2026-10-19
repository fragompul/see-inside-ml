import { ACTION_COUNT, type GridWorld } from './grid';

export interface OptimalSolution {
  /** Optimal state values V*(s) (0 on terminal states and walls). */
  values: Float64Array;
  /** Optimal action values Q*(s, a), `states × 4`. */
  q: Float64Array;
  /** For each state, which actions are optimal (within a small tolerance). */
  best: boolean[][];
  iterations: number;
}

/**
 * Value iteration: with the environment's exact dynamics known, the best possible values can be
 * computed directly. It is the yardstick a learning agent (which does not know the dynamics) is
 * measured against.
 */
export function valueIteration(world: GridWorld, gamma: number, tolerance = 1e-9): OptimalSolution {
  const n = world.states;
  const v = new Float64Array(n);
  const q = new Float64Array(n * ACTION_COUNT);
  let iterations = 0;
  for (; iterations < 5000; iterations++) {
    let change = 0;
    for (let s = 0; s < n; s++) {
      if (world.isTerminal(s) || world.kind(s) === 'wall') continue;
      let best = -Infinity;
      for (let a = 0; a < ACTION_COUNT; a++) {
        let value = 0;
        for (const { probability, result } of world.transitions(s, a))
          value +=
            probability * (result.reward + (result.done ? 0 : gamma * (v[result.next] as number)));
        q[s * ACTION_COUNT + a] = value;
        best = Math.max(best, value);
      }
      change = Math.max(change, Math.abs(best - (v[s] as number)));
      v[s] = best;
    }
    if (change < tolerance) break;
  }
  const best = Array.from({ length: n }, (_, s) => {
    const top = Math.max(
      ...Array.from({ length: ACTION_COUNT }, (_, a) => q[s * ACTION_COUNT + a] as number),
    );
    return Array.from(
      { length: ACTION_COUNT },
      (_, a) => (q[s * ACTION_COUNT + a] as number) >= top - 1e-6,
    );
  });
  return { values: v, q, best, iterations };
}
