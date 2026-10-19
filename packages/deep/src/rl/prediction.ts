import { createRng, type Rng } from '@sim/core';

/**
 * Prediction (policy evaluation) on Sutton and Barto's random walk: a row of states, an agent that steps
 * left or right with equal probability, reward 1 for leaving on the right and 0 for leaving on the left.
 * The true value of state i is i / (n + 1), so every method can be scored exactly.
 */
export type PredictionMethod = 'mc' | 'td' | 'nstep' | 'lambda';

export const PREDICTION_METHODS: PredictionMethod[] = ['mc', 'td', 'nstep', 'lambda'];

export interface PredictionOptions {
  /** Number of non-terminal states. */
  states: number;
  method: PredictionMethod;
  alpha: number;
  /** Look-ahead of n-step TD. */
  n: number;
  /** Trace decay of TD(lambda). */
  lambda: number;
  episodes: number;
  seed: number;
}

export const trueValues = (states: number): number[] =>
  Array.from({ length: states }, (_, i) => (i + 1) / (states + 1));

export interface Episode {
  /** Visited non-terminal states (0-based) in order. */
  states: number[];
  /** Reward on leaving the row: 1 at the right end, 0 at the left. */
  reward: number;
}

export function randomWalkEpisode(states: number, rng: Rng): Episode {
  let s = Math.floor(states / 2);
  const visited: number[] = [];
  for (;;) {
    visited.push(s);
    s += rng.next() < 0.5 ? -1 : 1;
    if (s < 0) return { states: visited, reward: 0 };
    if (s >= states) return { states: visited, reward: 1 };
  }
}

export const rmsError = (v: readonly number[], truth: readonly number[]): number =>
  Math.sqrt(v.reduce((a, x, i) => a + (x - (truth[i] as number)) ** 2, 0) / v.length);

export interface PredictionRun {
  values: number[];
  /** Root-mean-square error against the true values after every episode (entry 0 is the start). */
  error: number[];
}

export function learnValues(o: PredictionOptions): PredictionRun {
  const rng = createRng(o.seed);
  const truth = trueValues(o.states);
  const v = new Array<number>(o.states).fill(0.5);
  const error = [rmsError(v, truth)];
  for (let e = 0; e < o.episodes; e++) {
    const ep = randomWalkEpisode(o.states, rng);
    const T = ep.states.length;
    // value of the terminal step is 0; the only reward arrives on the last transition
    const value = (t: number) => (t >= T ? 0 : (v[ep.states[t] as number] as number));
    const reward = (t: number) => (t === T - 1 ? ep.reward : 0);
    if (o.method === 'mc') {
      // every-visit Monte Carlo: the return from any visit is the final reward
      ep.states.forEach(
        (s) => (v[s] = (v[s] as number) + o.alpha * (ep.reward - (v[s] as number))),
      );
    } else if (o.method === 'td') {
      for (let t = 0; t < T; t++) {
        const s = ep.states[t] as number;
        v[s] = (v[s] as number) + o.alpha * (reward(t) + value(t + 1) - (v[s] as number));
      }
    } else if (o.method === 'nstep') {
      for (let t = 0; t < T; t++) {
        const end = Math.min(t + o.n, T);
        let g = 0;
        for (let k = t; k < end; k++) g += reward(k);
        if (t + o.n < T) g += value(t + o.n);
        const s = ep.states[t] as number;
        v[s] = (v[s] as number) + o.alpha * (g - (v[s] as number));
      }
    } else {
      // backward view with accumulating eligibility traces
      const trace = new Array<number>(o.states).fill(0);
      for (let t = 0; t < T; t++) {
        const s = ep.states[t] as number;
        const delta = reward(t) + value(t + 1) - (v[s] as number);
        for (let i = 0; i < o.states; i++) trace[i] = (trace[i] as number) * o.lambda;
        trace[s] = (trace[s] as number) + 1;
        for (let i = 0; i < o.states; i++)
          v[i] = (v[i] as number) + o.alpha * delta * (trace[i] as number);
      }
    }
    error.push(rmsError(v, truth));
  }
  return { values: v, error };
}

/** Average error curve over many independent runs. */
export function averageError(o: PredictionOptions, runs: number): number[] {
  const acc = new Array<number>(o.episodes + 1).fill(0);
  for (let r = 0; r < runs; r++)
    learnValues({ ...o, seed: o.seed * 1000 + r }).error.forEach(
      (e, i) => (acc[i] = (acc[i] as number) + e / runs),
    );
  return acc;
}
