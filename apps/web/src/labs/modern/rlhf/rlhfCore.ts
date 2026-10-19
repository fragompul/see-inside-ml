import {
  bestOfN,
  expectation,
  klDivergence,
  klRegularisedPolicy,
  trainRlhf,
  type Response,
} from '@sim/deep';

export interface RlhfScenario {
  /** Preference pairs the reward model is trained on. */
  pairs: number;
  /** How noisy the human choices are (larger means less reliable). */
  noise: number;
}

export const DEFAULT_RLHF: RlhfScenario = { pairs: 20, noise: 0.8 };

const SEEDS = [1, 2, 3, 4, 5];
const POOL = 3000;
const VIEW_NOISE = 1.0;
/** Penalty weights from very loose to very tight. */
export const BETAS = Array.from({ length: 24 }, (_, i) => 30 * 0.72 ** i);
export const BEST_OF = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 3000];

export interface RlhfResult {
  scenario: RlhfScenario;
  /** Accuracy of the reward model on new pairs, mean over seeds. */
  accuracy: number;
  /** Weights of the reward model (mean over seeds) on: three quality views, length, noise. */
  weights: number[];
  /** Along the sweep from a loose to a tight penalty (means over seeds): distance from the original policy, reward-model score and true reward. */
  kl: number[];
  proxy: number[];
  truth: number[];
  /** Best-of-n: reward-model score and true reward. */
  bestOfProxy: number[];
  bestOfTruth: number[];
  /** True reward of the original policy (the average response). */
  baseline: number;
  /** The KL at which the true reward is highest, and the highest true reward. */
  peakKl: number;
  peakTruth: number;
  /** How far below its peak the true reward is at the end of the sweep. */
  fall: number;
}

const cache = new Map<string, RlhfResult>();

export function runRlhf(s: RlhfScenario): RlhfResult {
  const key = JSON.stringify(s);
  const hit = cache.get(key);
  if (hit) return hit;
  const n = SEEDS.length;
  const kl = new Array<number>(BETAS.length).fill(0);
  const proxy = new Array<number>(BETAS.length).fill(0);
  const truth = new Array<number>(BETAS.length).fill(0);
  const bProxy = new Array<number>(BEST_OF.length).fill(0);
  const bTruth = new Array<number>(BEST_OF.length).fill(0);
  const weights = new Array<number>(5).fill(0);
  let accuracy = 0;
  let baseline = 0;
  for (const seed of SEEDS) {
    const run = trainRlhf({
      candidates: POOL,
      pairs: s.pairs,
      noise: s.noise,
      seed,
      viewNoise: VIEW_NOISE,
    });
    accuracy += run.heldOutAccuracy / n;
    run.model.weights.forEach((w, k) => (weights[k] = (weights[k] as number) + w / n));
    const scores = run.candidates.map((c: Response) => run.model.score(c.features));
    const truths = run.candidates.map((c: Response) => c.truth);
    const reference = scores.map(() => 1 / POOL);
    baseline += expectation(reference, truths) / n;
    BETAS.forEach((beta, i) => {
      const p = klRegularisedPolicy(scores, reference, beta);
      kl[i] = (kl[i] as number) + klDivergence(p, reference) / n;
      proxy[i] = (proxy[i] as number) + expectation(p, scores) / n;
      truth[i] = (truth[i] as number) + expectation(p, truths) / n;
    });
    BEST_OF.forEach((m, i) => {
      const r = bestOfN(scores, truths, m);
      bProxy[i] = (bProxy[i] as number) + r.proxy / n;
      bTruth[i] = (bTruth[i] as number) + r.truth / n;
    });
  }
  const peak = truth.indexOf(Math.max(...truth));
  const out: RlhfResult = {
    scenario: s,
    accuracy,
    weights,
    kl,
    proxy,
    truth,
    bestOfProxy: bProxy,
    bestOfTruth: bTruth,
    baseline,
    peakKl: kl[peak] as number,
    peakTruth: truth[peak] as number,
    fall: (truth[peak] as number) - (truth[truth.length - 1] as number),
  };
  cache.set(key, out);
  return out;
}
