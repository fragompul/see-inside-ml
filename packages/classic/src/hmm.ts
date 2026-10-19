import { createRng, type Rng } from '@sim/core';

/**
 * Hidden Markov model with discrete observations. A hidden state evolves as a Markov chain
 * (`transition`), and each state emits a symbol (`emission`). Only the symbols are seen. The three
 * classic questions are answered by dynamic programming over the chain of states:
 *   - how likely is the sequence?  -> forward algorithm,
 *   - what was each state?         -> forward-backward (smoothing) or Viterbi (best whole path),
 *   - what are the parameters?     -> Baum-Welch (EM).
 * Everything is scaled per step, so long sequences do not underflow.
 */
export interface HmmParams {
  /** Initial state distribution, length K. */
  initial: number[];
  /** `K x K`, row i = distribution of the next state given state i. */
  transition: number[][];
  /** `K x M`, row i = distribution of the symbol given state i. */
  emission: number[][];
}

export const cloneParams = (p: HmmParams): HmmParams => ({
  initial: [...p.initial],
  transition: p.transition.map((r) => [...r]),
  emission: p.emission.map((r) => [...r]),
});

export interface ForwardResult {
  /** Scaled forward messages: `alpha[t][i] = P(state_t = i | obs_1..t)`. */
  alpha: number[][];
  /** Scaling factors: `scale[t] = P(obs_t | obs_1..t-1)`. */
  scale: number[];
  logLikelihood: number;
}

export function forward(p: HmmParams, obs: readonly number[]): ForwardResult {
  const k = p.initial.length;
  const alpha: number[][] = [];
  const scale: number[] = [];
  let ll = 0;
  for (let t = 0; t < obs.length; t++) {
    const a = new Array<number>(k).fill(0);
    for (let j = 0; j < k; j++) {
      let prior = 0;
      if (t === 0) prior = p.initial[j] as number;
      else
        for (let i = 0; i < k; i++)
          prior += (alpha[t - 1]![i] as number) * (p.transition[i]![j] as number);
      a[j] = prior * (p.emission[j]![obs[t] as number] as number);
    }
    const c = a.reduce((s, v) => s + v, 0);
    scale.push(c);
    ll += Math.log(c);
    alpha.push(a.map((v) => v / (c || 1)));
  }
  return { alpha, scale, logLikelihood: ll };
}

export function backward(
  p: HmmParams,
  obs: readonly number[],
  scale: readonly number[],
): number[][] {
  const k = p.initial.length;
  const n = obs.length;
  const beta: number[][] = Array.from({ length: n }, () => new Array<number>(k).fill(1));
  for (let t = n - 2; t >= 0; t--)
    for (let i = 0; i < k; i++) {
      let s = 0;
      for (let j = 0; j < k; j++)
        s +=
          (p.transition[i]![j] as number) *
          (p.emission[j]![obs[t + 1] as number] as number) *
          (beta[t + 1]![j] as number);
      beta[t]![i] = s / (scale[t + 1] as number);
    }
  return beta;
}

export interface Smoothed {
  /** `gamma[t][i] = P(state_t = i | all observations)`. */
  gamma: number[][];
  /** `xi[t][i][j] = P(state_t = i, state_{t+1} = j | all observations)` for t < n - 1. */
  xi: number[][][];
  logLikelihood: number;
}

export function smooth(p: HmmParams, obs: readonly number[]): Smoothed {
  const k = p.initial.length;
  const { alpha, scale, logLikelihood } = forward(p, obs);
  const beta = backward(p, obs, scale);
  const gamma = alpha.map((a, t) => a.map((v, i) => v * (beta[t]![i] as number)));
  const xi: number[][][] = [];
  for (let t = 0; t < obs.length - 1; t++) {
    const m: number[][] = [];
    for (let i = 0; i < k; i++) {
      const row: number[] = [];
      for (let j = 0; j < k; j++)
        row.push(
          ((alpha[t]![i] as number) *
            (p.transition[i]![j] as number) *
            (p.emission[j]![obs[t + 1] as number] as number) *
            (beta[t + 1]![j] as number)) /
            (scale[t + 1] as number),
        );
      m.push(row);
    }
    xi.push(m);
  }
  return { gamma, xi, logLikelihood };
}

/** The single most probable state path (Viterbi), computed in log space. */
export function viterbi(
  p: HmmParams,
  obs: readonly number[],
): { path: number[]; logProbability: number } {
  const k = p.initial.length;
  const n = obs.length;
  const delta: number[][] = [];
  const from: number[][] = [];
  for (let t = 0; t < n; t++) {
    const d = new Array<number>(k).fill(-Infinity);
    const f = new Array<number>(k).fill(0);
    for (let j = 0; j < k; j++) {
      const emit = Math.log(p.emission[j]![obs[t] as number] as number);
      if (t === 0) d[j] = Math.log(p.initial[j] as number) + emit;
      else {
        let best = -Infinity;
        let arg = 0;
        for (let i = 0; i < k; i++) {
          const v = (delta[t - 1]![i] as number) + Math.log(p.transition[i]![j] as number);
          if (v > best) {
            best = v;
            arg = i;
          }
        }
        d[j] = best + emit;
        f[j] = arg;
      }
    }
    delta.push(d);
    from.push(f);
  }
  const last = delta[n - 1] as number[];
  let state = last.indexOf(Math.max(...last));
  const logProbability = last[state] as number;
  const path = new Array<number>(n).fill(0);
  for (let t = n - 1; t >= 0; t--) {
    path[t] = state;
    state = from[t]![state] as number;
  }
  return { path, logProbability };
}

/** Draws a hidden path and the symbols it emitted. */
export function sample(
  p: HmmParams,
  length: number,
  rng: Rng,
): { states: number[]; obs: number[] } {
  const draw = (dist: readonly number[]) => {
    let u = rng.next();
    for (let i = 0; i < dist.length; i++) {
      u -= dist[i] as number;
      if (u <= 0) return i;
    }
    return dist.length - 1;
  };
  const states: number[] = [];
  const obs: number[] = [];
  let s = draw(p.initial);
  for (let t = 0; t < length; t++) {
    if (t > 0) s = draw(p.transition[s] as number[]);
    states.push(s);
    obs.push(draw(p.emission[s] as number[]));
  }
  return { states, obs };
}

export interface BaumWelchState {
  iteration: number;
  params: HmmParams;
  logLikelihood: number;
}

/** Baum-Welch (EM) as a generator over one or more sequences; state 0 is the starting point. */
export function* baumWelchSteps(
  start: HmmParams,
  sequences: readonly (readonly number[])[],
  iterations = 50,
  tolerance = 1e-8,
): Generator<BaumWelchState, void, undefined> {
  let p = cloneParams(start);
  const k = p.initial.length;
  const m = (p.emission[0] as number[]).length;
  const total = (q: HmmParams) => sequences.reduce((s, o) => s + forward(q, o).logLikelihood, 0);
  let ll = total(p);
  yield { iteration: 0, params: cloneParams(p), logLikelihood: ll };
  for (let it = 1; it <= iterations; it++) {
    const initial = new Array<number>(k).fill(0);
    const trans = Array.from({ length: k }, () => new Array<number>(k).fill(0));
    const emit = Array.from({ length: k }, () => new Array<number>(m).fill(0));
    for (const obs of sequences) {
      const { gamma, xi } = smooth(p, obs);
      for (let i = 0; i < k; i++) initial[i] = (initial[i] as number) + (gamma[0]![i] as number);
      xi.forEach((mat) =>
        mat.forEach((row, i) =>
          row.forEach((v, j) => (trans[i]![j] = (trans[i]![j] as number) + v)),
        ),
      );
      gamma.forEach((g, t) =>
        g.forEach(
          (v, i) => (emit[i]![obs[t] as number] = (emit[i]![obs[t] as number] as number) + v),
        ),
      );
    }
    const norm = (v: number[]) => {
      const s = v.reduce((a, b) => a + b, 0);
      return v.map((x) => (s > 0 ? x / s : 1 / v.length));
    };
    p = { initial: norm(initial), transition: trans.map(norm), emission: emit.map(norm) };
    const next = total(p);
    const done = Math.abs(next - ll) < tolerance;
    ll = next;
    yield { iteration: it, params: cloneParams(p), logLikelihood: ll };
    if (done) break;
  }
}

/** The log-likelihood by brute force over every state path (for tiny models: the tests' ground truth). */
export function bruteForce(
  p: HmmParams,
  obs: readonly number[],
): { logLikelihood: number; bestPath: number[]; bestLogProbability: number } {
  const k = p.initial.length;
  const n = obs.length;
  let sum = 0;
  let best = -Infinity;
  let bestPath: number[] = [];
  const path = new Array<number>(n).fill(0);
  const total = k ** n;
  for (let code = 0; code < total; code++) {
    let c = code;
    for (let t = 0; t < n; t++) {
      path[t] = c % k;
      c = Math.floor(c / k);
    }
    let pr = p.initial[path[0] as number] as number;
    for (let t = 0; t < n; t++) {
      if (t > 0) pr *= p.transition[path[t - 1] as number]![path[t] as number] as number;
      pr *= p.emission[path[t] as number]![obs[t] as number] as number;
    }
    sum += pr;
    if (pr > 0 && Math.log(pr) > best) {
      best = Math.log(pr);
      bestPath = [...path];
    }
  }
  return { logLikelihood: Math.log(sum), bestPath, bestLogProbability: best };
}

export const randomParams = (k: number, m: number, seed: number): HmmParams => {
  const rng = createRng(seed);
  const dist = (n: number) => {
    const v = Array.from({ length: n }, () => 0.5 + rng.next());
    const s = v.reduce((a, b) => a + b, 0);
    return v.map((x) => x / s);
  };
  return {
    initial: dist(k),
    transition: Array.from({ length: k }, () => dist(k)),
    emission: Array.from({ length: k }, () => dist(m)),
  };
};
