import { createRng, type Rng } from '@sim/core';
import { ACTION_COUNT, type GridWorld } from './grid';

/**
 * Deep Q-learning on a gridworld. A small network maps a cell to the value of each move and is trained to
 * satisfy the Bellman equation, Q(s, a) = r + gamma max_a' Q'(s', a'), on mini-batches of past transitions.
 * Two devices keep this from falling apart: a replay buffer (so a batch is a mix of old and new experience
 * instead of a run of nearly identical steps) and a target network Q' that is a slowly updated copy of Q (so
 * the target does not move with every update). Either can be switched off to see why they are there.
 */
export type CellFeatures = 'coordinates' | 'one-hot';

export interface DqnOptions {
  gamma: number;
  learningRate: number;
  hidden: number;
  features: CellFeatures;
  replay: boolean;
  replayCapacity: number;
  batch: number;
  /** Steps between copies of the online network into the target network; 0 uses the online network as its own target. */
  targetEvery: number;
  epsilonStart: number;
  epsilonEnd: number;
  /** Steps over which epsilon falls linearly. */
  epsilonSteps: number;
  episodes: number;
  maxSteps: number;
  seed: number;
}

export const DEFAULT_DQN: DqnOptions = {
  gamma: 0.95,
  learningRate: 0.01,
  hidden: 32,
  features: 'coordinates',
  replay: true,
  replayCapacity: 2000,
  batch: 32,
  targetEvery: 100,
  epsilonStart: 1,
  epsilonEnd: 0.05,
  epsilonSteps: 1500,
  episodes: 150,
  maxSteps: 60,
  seed: 1,
};

/** One hidden layer of tanh units; backpropagation written by hand and checked against finite differences. */
export class QNet {
  readonly w1: Float64Array;
  readonly b1: Float64Array;
  readonly w2: Float64Array;
  readonly b2: Float64Array;
  private readonly m: Float64Array[] = [];
  private readonly v: Float64Array[] = [];
  private t = 0;

  constructor(
    readonly inputs: number,
    readonly hidden: number,
    rng: Rng,
  ) {
    this.w1 = Float64Array.from(
      { length: inputs * hidden },
      () => (rng.normal() * Math.sqrt(2)) / Math.sqrt(inputs),
    );
    this.b1 = new Float64Array(hidden);
    this.w2 = Float64Array.from(
      { length: hidden * ACTION_COUNT },
      () => (rng.normal() * 0.5) / Math.sqrt(hidden),
    );
    this.b2 = new Float64Array(ACTION_COUNT);
    for (const p of this.parameters) {
      this.m.push(new Float64Array(p.length));
      this.v.push(new Float64Array(p.length));
    }
  }

  get parameters(): Float64Array[] {
    return [this.w1, this.b1, this.w2, this.b2];
  }

  copyFrom(other: QNet): void {
    this.parameters.forEach((p, i) => p.set(other.parameters[i] as Float64Array));
  }

  forward(x: ArrayLike<number>): { hidden: Float64Array; q: Float64Array } {
    const h = new Float64Array(this.hidden);
    for (let j = 0; j < this.hidden; j++) {
      let s = this.b1[j] as number;
      for (let i = 0; i < this.inputs; i++)
        s += (x[i] as number) * (this.w1[i * this.hidden + j] as number);
      h[j] = Math.tanh(s);
    }
    const q = new Float64Array(ACTION_COUNT);
    for (let a = 0; a < ACTION_COUNT; a++) {
      let s = this.b2[a] as number;
      for (let j = 0; j < this.hidden; j++)
        s += (h[j] as number) * (this.w2[j * ACTION_COUNT + a] as number);
      q[a] = s;
    }
    return { hidden: h, q };
  }

  /** Gradients of 0.5 * mean over the batch of (Q(s, a) - target)^2, accumulated into fresh arrays. */
  gradients(
    xs: ArrayLike<number>[],
    actions: number[],
    targets: number[],
  ): { grads: Float64Array[]; loss: number } {
    const grads = this.parameters.map((p) => new Float64Array(p.length));
    const [g1, gb1, g2, gb2] = grads as [Float64Array, Float64Array, Float64Array, Float64Array];
    let loss = 0;
    const n = xs.length;
    xs.forEach((x, k) => {
      const { hidden, q } = this.forward(x);
      const a = actions[k] as number;
      const err = (q[a] as number) - (targets[k] as number);
      loss += (0.5 * err * err) / n;
      const dq = err / n;
      gb2[a] = (gb2[a] as number) + dq;
      for (let j = 0; j < this.hidden; j++) {
        g2[j * ACTION_COUNT + a] =
          (g2[j * ACTION_COUNT + a] as number) + dq * (hidden[j] as number);
        const dh =
          dq * (this.w2[j * ACTION_COUNT + a] as number) * (1 - (hidden[j] as number) ** 2);
        gb1[j] = (gb1[j] as number) + dh;
        for (let i = 0; i < this.inputs; i++)
          g1[i * this.hidden + j] = (g1[i * this.hidden + j] as number) + dh * (x[i] as number);
      }
    });
    return { grads, loss };
  }

  adam(grads: Float64Array[], lr: number): void {
    this.t += 1;
    const b1 = 0.9;
    const b2 = 0.999;
    this.parameters.forEach((p, k) => {
      const g = grads[k] as Float64Array;
      const m = this.m[k] as Float64Array;
      const v = this.v[k] as Float64Array;
      for (let i = 0; i < p.length; i++) {
        m[i] = b1 * (m[i] as number) + (1 - b1) * (g[i] as number);
        v[i] = b2 * (v[i] as number) + (1 - b2) * (g[i] as number) ** 2;
        const mh = (m[i] as number) / (1 - b1 ** this.t);
        const vh = (v[i] as number) / (1 - b2 ** this.t);
        p[i] = (p[i] as number) - (lr * mh) / (Math.sqrt(vh) + 1e-8);
      }
    });
  }
}

export function featuresOf(world: GridWorld, state: number, kind: CellFeatures): Float64Array {
  const { width, height } = world.layout;
  if (kind === 'one-hot') {
    const x = new Float64Array(world.states);
    x[state] = 1;
    return x;
  }
  return Float64Array.of(
    ((state % width) / (width - 1)) * 2 - 1,
    (Math.floor(state / width) / (height - 1)) * 2 - 1,
  );
}

/** The Bellman targets for a batch: r + gamma max Q'(s'), or just r when the episode ended. */
export function bellmanTargets(
  target: QNet,
  gamma: number,
  rewards: number[],
  nextFeatures: ArrayLike<number>[],
  done: boolean[],
): number[] {
  return rewards.map((r, k) =>
    done[k] ? r : r + gamma * Math.max(...target.forward(nextFeatures[k] as ArrayLike<number>).q),
  );
}

interface Transition {
  x: Float64Array;
  a: number;
  r: number;
  next: Float64Array;
  done: boolean;
}

export interface DqnRun {
  /** Undiscounted return and length of every episode. */
  returns: number[];
  steps: number[];
  /** Mean training loss in each episode (NaN before learning starts). */
  losses: number[];
  /** Largest |Q| over all cells and moves at the end of each episode: a runaway value signals divergence. */
  maxQ: number[];
  net: QNet;
  /** Whether following the greedy policy from the start reaches the goal at the end of training. */
  solved: boolean;
}

export function trainDqn(world: GridWorld, o: DqnOptions): DqnRun {
  const rng = createRng(o.seed);
  const inputs = featuresOf(world, world.start, o.features).length;
  const net = new QNet(inputs, o.hidden, createRng(o.seed + 1));
  const target = new QNet(inputs, o.hidden, createRng(o.seed + 2));
  target.copyFrom(net);
  const buffer: Transition[] = [];
  let write = 0;
  let totalSteps = 0;
  const returns: number[] = [];
  const steps: number[] = [];
  const losses: number[] = [];
  const maxQ: number[] = [];
  const cells = Array.from({ length: world.states }, (_, s) => s).filter(
    (s) => world.kind(s) !== 'wall',
  );
  const everyFeature = cells.map((s) => featuresOf(world, s, o.features));
  const greedy = (x: Float64Array) => {
    const { q } = net.forward(x);
    return q.indexOf(Math.max(...q));
  };

  for (let e = 0; e < o.episodes; e++) {
    let s = world.start;
    let total = 0;
    let n = 0;
    let lossSum = 0;
    let lossCount = 0;
    while (n < o.maxSteps) {
      const epsilon =
        o.epsilonStart +
        (o.epsilonEnd - o.epsilonStart) * Math.min(1, totalSteps / Math.max(1, o.epsilonSteps));
      const x = featuresOf(world, s, o.features);
      const a = rng.next() < epsilon ? Math.floor(rng.next() * ACTION_COUNT) : greedy(x);
      const step = world.step(s, a, rng);
      const tr: Transition = {
        x,
        a,
        r: step.reward,
        next: featuresOf(world, step.next, o.features),
        done: step.done,
      };
      if (o.replay) {
        if (buffer.length < o.replayCapacity) buffer.push(tr);
        else buffer[write % o.replayCapacity] = tr;
        write += 1;
      }
      const pool = o.replay ? buffer : [tr];
      if (pool.length >= (o.replay ? o.batch : 1)) {
        const batch = o.replay
          ? Array.from(
              { length: o.batch },
              () => pool[Math.floor(rng.next() * pool.length)] as Transition,
            )
          : pool;
        const targets = bellmanTargets(
          o.targetEvery > 0 ? target : net,
          o.gamma,
          batch.map((b) => b.r),
          batch.map((b) => b.next),
          batch.map((b) => b.done),
        );
        const { grads, loss } = net.gradients(
          batch.map((b) => b.x),
          batch.map((b) => b.a),
          targets,
        );
        net.adam(grads, o.learningRate);
        lossSum += loss;
        lossCount += 1;
      }
      totalSteps += 1;
      if (o.targetEvery > 0 && totalSteps % o.targetEvery === 0) target.copyFrom(net);
      total += step.reward;
      n += 1;
      s = step.next;
      if (step.done) break;
    }
    returns.push(total);
    steps.push(n);
    losses.push(lossCount ? lossSum / lossCount : Number.NaN);
    maxQ.push(Math.max(...everyFeature.flatMap((x) => Array.from(net.forward(x).q).map(Math.abs))));
  }

  // does the greedy policy get to the goal?
  let state = world.start;
  let solved = false;
  for (let t = 0; t < o.maxSteps; t++) {
    const next = world.move(state, greedy(featuresOf(world, state, o.features)));
    if (next === state && t > 0) break;
    state = next;
    if (world.isTerminal(state)) {
      solved = world.kind(state) === 'goal';
      break;
    }
  }
  return { returns, steps, losses, maxQ, net, solved };
}
