import { createRng } from '@sim/core';
import { ACTION_COUNT, type GridWorld } from './grid';

/**
 * Dyna-Q: Q-learning that also learns a model of the world (what happened the last time it did this here)
 * and uses it to rehearse: after each real step it replays `planning` remembered steps. Real experience
 * is expensive; imagined experience is free.
 */
export interface DynaOptions {
  gamma: number;
  alpha: number;
  epsilon: number;
  /** Imagined updates after every real step. */
  planning: number;
  episodes: number;
  maxSteps: number;
  seed: number;
}

export const DEFAULT_DYNA: DynaOptions = {
  gamma: 0.95,
  alpha: 0.5,
  epsilon: 0.1,
  planning: 0,
  episodes: 40,
  maxSteps: 400,
  seed: 1,
};

export interface DynaRun {
  /** Real steps taken in each episode. */
  steps: number[];
  /** Total return of each episode. */
  returns: number[];
  /** Greedy action in each state at the end. */
  policy: number[];
  q: Float64Array;
  /** Real steps taken in total. */
  realSteps: number;
  /** Imagined updates performed in total. */
  imaginedSteps: number;
}

export function trainDyna(world: GridWorld, o: DynaOptions): DynaRun {
  const rng = createRng(o.seed);
  const q = new Float64Array(world.states * ACTION_COUNT);
  const model = new Map<number, { reward: number; next: number; done: boolean }>();
  const seen: number[] = [];
  const steps: number[] = [];
  const returns: number[] = [];
  let real = 0;
  let imagined = 0;
  const best = (s: number) => {
    let top = -Infinity;
    for (let a = 0; a < ACTION_COUNT; a++) top = Math.max(top, q[s * ACTION_COUNT + a] as number);
    return top;
  };
  const update = (s: number, a: number, r: number, next: number, done: boolean) => {
    const target = r + (done ? 0 : o.gamma * best(next));
    const i = s * ACTION_COUNT + a;
    q[i] = (q[i] as number) + o.alpha * (target - (q[i] as number));
  };
  const greedy = (s: number) => {
    const top = best(s);
    const ties: number[] = [];
    for (let a = 0; a < ACTION_COUNT; a++)
      if ((q[s * ACTION_COUNT + a] as number) >= top - 1e-12) ties.push(a);
    return ties[Math.floor(rng.next() * ties.length)] as number;
  };
  for (let e = 0; e < o.episodes; e++) {
    let s = world.start;
    let total = 0;
    let n = 0;
    while (n < o.maxSteps) {
      const a = rng.next() < o.epsilon ? Math.floor(rng.next() * ACTION_COUNT) : greedy(s);
      const step = world.step(s, a, rng);
      update(s, a, step.reward, step.next, step.done);
      const key = s * ACTION_COUNT + a;
      if (!model.has(key)) seen.push(key);
      model.set(key, { reward: step.reward, next: step.next, done: step.done });
      for (let p = 0; p < o.planning; p++) {
        const k = seen[Math.floor(rng.next() * seen.length)] as number;
        const m = model.get(k) as { reward: number; next: number; done: boolean };
        update(Math.floor(k / ACTION_COUNT), k % ACTION_COUNT, m.reward, m.next, m.done);
        imagined += 1;
      }
      total += step.reward;
      real += 1;
      n += 1;
      s = step.next;
      if (step.done) break;
    }
    steps.push(n);
    returns.push(total);
  }
  const policy = Array.from({ length: world.states }, (_, s) => {
    let a0 = 0;
    for (let a = 1; a < ACTION_COUNT; a++)
      if ((q[s * ACTION_COUNT + a] as number) > (q[s * ACTION_COUNT + a0] as number)) a0 = a;
    return a0;
  });
  return { steps, returns, policy, q, realSteps: real, imaginedSteps: imagined };
}

/** Average steps-per-episode curve over several runs. */
export function averageSteps(world: GridWorld, o: DynaOptions, runs: number): number[] {
  const acc = new Array<number>(o.episodes).fill(0);
  for (let r = 0; r < runs; r++)
    trainDyna(world, { ...o, seed: o.seed * 1000 + r }).steps.forEach(
      (v, i) => (acc[i] = (acc[i] as number) + v / runs),
    );
  return acc;
}
