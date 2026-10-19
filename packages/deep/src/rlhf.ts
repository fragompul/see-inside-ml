import { createRng, type Rng } from '@sim/core';

/**
 * A toy of learning from human feedback. There is a pool of candidate responses; each has features, a hidden true
 * quality and a length. People prefer the better of two responses (with noise). A reward model is fitted to those
 * preferences (Bradley-Terry: P(a preferred) = sigmoid(r(a) - r(b))) from features that only partly reveal quality, and a
 * policy is then improved against the reward model, with a penalty on how far it moves from the original policy.
 * Because the reward model is imperfect, pushing hard on it eventually makes the policy worse for the people it was
 * meant to please: the model of the reward is exploited, not the reward.
 */
export interface Response {
  /** The features the reward model sees: noisy views of three quality factors, the length, and a pure-noise feature. */
  features: number[];
  /** The true reward people would give. */
  truth: number;
  length: number;
}

export const RM_FEATURES = 5;

export function makeCandidates(count: number, rng: Rng, viewNoise = 0.7): Response[] {
  return Array.from({ length: count }, () => {
    const quality = [rng.normal(), rng.normal(), rng.normal()];
    // long responses tend to be good ones in the data, which is where the reward model learns to like length
    const length = 0.55 * (quality[0] as number) + Math.sqrt(1 - 0.55 ** 2) * rng.normal();
    // but being too long is a real fault
    const penalty = 1.6 * Math.max(0, length - 0.4) ** 2;
    const truth =
      (quality[0] as number) + (quality[1] as number) + (quality[2] as number) - penalty;
    const view = quality.map((q) => q + viewNoise * rng.normal());
    return { features: [...view, length, rng.normal()], truth, length };
  });
}

export interface PreferencePair {
  better: number;
  worse: number;
}

/** Human preferences: the response with the higher true reward plus Gumbel-like noise (a Bradley-Terry choice). */
export function samplePreferences(
  candidates: readonly Response[],
  pairs: number,
  noise: number,
  rng: Rng,
): PreferencePair[] {
  const out: PreferencePair[] = [];
  while (out.length < pairs) {
    const a = Math.floor(rng.next() * candidates.length);
    const b = Math.floor(rng.next() * candidates.length);
    if (a === b) continue;
    const gap = ((candidates[a] as Response).truth - (candidates[b] as Response).truth) / noise;
    const pA = 1 / (1 + Math.exp(-gap));
    out.push(rng.next() < pA ? { better: a, worse: b } : { better: b, worse: a });
  }
  return out;
}

export interface RewardModel {
  weights: number[];
  score: (features: readonly number[]) => number;
}

/** Fits a linear reward model to the preferences by gradient descent on the Bradley-Terry loss (with a small L2 penalty). */
export function fitRewardModel(
  candidates: readonly Response[],
  pairs: readonly PreferencePair[],
  options: { iterations: number; learningRate: number; decay: number } = {
    iterations: 400,
    learningRate: 0.5,
    decay: 0.01,
  },
): RewardModel {
  const w = new Array<number>(RM_FEATURES).fill(0);
  for (let it = 0; it < options.iterations; it++) {
    const grad = new Array<number>(RM_FEATURES).fill(0);
    for (const { better, worse } of pairs) {
      const a = (candidates[better] as Response).features;
      const b = (candidates[worse] as Response).features;
      let diff = 0;
      for (let k = 0; k < RM_FEATURES; k++)
        diff += (w[k] as number) * ((a[k] as number) - (b[k] as number));
      const p = 1 / (1 + Math.exp(-diff));
      for (let k = 0; k < RM_FEATURES; k++)
        grad[k] =
          (grad[k] as number) - ((1 - p) * ((a[k] as number) - (b[k] as number))) / pairs.length;
    }
    for (let k = 0; k < RM_FEATURES; k++)
      w[k] =
        (w[k] as number) -
        options.learningRate * ((grad[k] as number) + options.decay * (w[k] as number));
  }
  return { weights: w, score: (f) => f.reduce((s, v, k) => s + v * (w[k] as number), 0) };
}

/** Share of held-out pairs on which the reward model agrees with the people. */
export function preferenceAccuracy(
  model: RewardModel,
  candidates: readonly Response[],
  pairs: readonly PreferencePair[],
): number {
  let right = 0;
  for (const { better, worse } of pairs)
    if (
      model.score((candidates[better] as Response).features) >
      model.score((candidates[worse] as Response).features)
    )
      right++;
  return right / pairs.length;
}

/**
 * The policy that maximises (expected reward model score) - beta * KL(policy || reference) over distributions on the
 * candidates is proportional to reference * exp(score / beta). Returns its probabilities.
 */
export function klRegularisedPolicy(
  scores: readonly number[],
  reference: readonly number[],
  beta: number,
): number[] {
  const logits = scores.map((s, i) => s / beta + Math.log(reference[i] as number));
  const top = Math.max(...logits);
  const e = logits.map((l) => Math.exp(l - top));
  const z = e.reduce((a, b) => a + b, 0);
  return e.map((v) => v / z);
}

export const klDivergence = (p: readonly number[], q: readonly number[]): number =>
  p.reduce((s, pi, i) => (pi > 0 ? s + pi * Math.log(pi / (q[i] as number)) : s), 0);

export const expectation = (p: readonly number[], values: readonly number[]): number =>
  p.reduce((s, pi, i) => s + pi * (values[i] as number), 0);

/** Mean true reward of picking, among `n` random candidates, the one the reward model likes best (exact, over the pool). */
export function bestOfN(
  scores: readonly number[],
  truths: readonly number[],
  n: number,
): { proxy: number; truth: number } {
  const order = Array.from(scores.keys()).sort(
    (a, b) => (scores[a] as number) - (scores[b] as number),
  );
  const m = scores.length;
  // probability that the candidate at rank r (0 = worst) is the best of n draws with replacement
  let proxy = 0;
  let truth = 0;
  for (let r = 0; r < m; r++) {
    const p = ((r + 1) / m) ** n - (r / m) ** n;
    proxy += p * (scores[order[r] as number] as number);
    truth += p * (truths[order[r] as number] as number);
  }
  return { proxy, truth };
}

export interface RlhfRun {
  candidates: Response[];
  model: RewardModel;
  /** Accuracy of the reward model on new preference pairs. */
  heldOutAccuracy: number;
}

export function trainRlhf(options: {
  candidates: number;
  pairs: number;
  noise: number;
  seed: number;
  viewNoise?: number;
}): RlhfRun {
  const rng = createRng(options.seed);
  const candidates = makeCandidates(options.candidates, rng, options.viewNoise);
  const train = samplePreferences(candidates, options.pairs, options.noise, rng);
  const model = fitRewardModel(candidates, train);
  const test = samplePreferences(candidates, 1500, options.noise, createRng(options.seed + 99));
  return { candidates, model, heldOutAccuracy: preferenceAccuracy(model, candidates, test) };
}
