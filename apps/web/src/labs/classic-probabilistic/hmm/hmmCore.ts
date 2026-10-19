import {
  baumWelchSteps,
  forward,
  sample,
  smooth,
  viterbi,
  type BaumWelchState,
  type HmmParams,
} from '@sim/classic';
import { createRng } from '@sim/core';

/**
 * The dishonest casino: a croupier swaps between a fair die and a loaded one (secretly, now and
 * then). You only see the rolls. The hidden-Markov machinery answers: which die was in use when,
 * how likely is this sequence, and can the model be learned from the rolls alone?
 */
export interface HmmScenario {
  /** Probability that the croupier keeps the same die for the next roll. */
  stay: number;
  /** Chance that the loaded die shows a six (the other faces share the rest equally). */
  sixes: number;
  length: number;
  /** Which model the filter uses: the one that made the data, a naive guess, or one learned by Baum-Welch. */
  inference: 'truth' | 'naive' | 'learned';
  iterations: number;
  seed: number;
}

export const DEFAULT_HMM: HmmScenario = {
  stay: 0.95,
  sixes: 0.5,
  length: 200,
  inference: 'truth',
  iterations: 40,
  seed: 4,
};

export const FAIR = 0;
export const LOADED = 1;

export function casinoModel(stay: number, sixes: number): HmmParams {
  const other = (1 - sixes) / 5;
  return {
    initial: [0.5, 0.5],
    transition: [
      [stay, 1 - stay],
      [1 - stay, stay],
    ],
    emission: [new Array<number>(6).fill(1 / 6), [other, other, other, other, other, sixes]],
  };
}

/** A deliberately naive model: states switch half the time and the loaded die is only mildly loaded. */
export const naiveModel = (): HmmParams => casinoModel(0.6, 0.25);

export interface HmmRun {
  scenario: HmmScenario;
  truth: HmmParams;
  states: number[];
  rolls: number[];
  /** The model used for inference. */
  model: HmmParams;
  /** P(loaded | all rolls) at every step. */
  posterior: number[];
  viterbi: number[];
  viterbiAccuracy: number;
  posteriorAccuracy: number;
  logLikelihood: number;
  /** log-likelihood under the generating model (the best a perfectly informed observer can do). */
  trueLogLikelihood: number;
  /** Baum-Welch trace when `inference === 'learned'`. */
  learning: BaumWelchState[] | null;
}

const agreement = (a: readonly number[], b: readonly number[]): number =>
  a.reduce((s, v, i) => s + (v === b[i] ? 1 : 0), 0) / a.length;

/** States found by an unsupervised fit carry arbitrary labels: use whichever labelling agrees more with the truth. */
export function relabelled(path: readonly number[], truth: readonly number[]): number[] {
  const flip = path.map((v) => 1 - v);
  return agreement(path, truth) >= agreement(flip, truth) ? [...path] : flip;
}

export function runHmm(s: HmmScenario): HmmRun {
  const truth = casinoModel(s.stay, s.sixes);
  const { states, obs } = sample(truth, s.length, createRng(s.seed));
  let learning: BaumWelchState[] | null = null;
  let model = truth;
  if (s.inference === 'naive') model = naiveModel();
  if (s.inference === 'learned') {
    // start from the naive guess: Baum-Welch only finds a local optimum, so where it starts matters
    learning = [...baumWelchSteps(naiveModel(), [obs], s.iterations)];
    model = learning[learning.length - 1]!.params;
  }
  const { gamma } = smooth(model, obs);
  const path = viterbi(model, obs).path;
  let pLoaded = gamma.map((g) => g[LOADED] as number);
  let vit = path;
  let post: number[] = gamma.map((g) => ((g[LOADED] as number) > 0.5 ? 1 : 0));
  if (s.inference === 'learned') {
    // the learned state called "1" may be the fair die: flip both views to match the truth best
    const flip =
      agreement(post, states) <
      agreement(
        post.map((v) => 1 - v),
        states,
      );
    if (flip) {
      pLoaded = pLoaded.map((v) => 1 - v);
      vit = path.map((v) => 1 - v);
      post = post.map((v) => 1 - v);
    }
  }
  return {
    scenario: s,
    truth,
    states,
    rolls: obs.map((o) => o + 1),
    model,
    posterior: pLoaded,
    viterbi: vit,
    viterbiAccuracy: agreement(vit, states),
    posteriorAccuracy: agreement(post, states),
    logLikelihood: forward(model, obs).logLikelihood,
    trueLogLikelihood: forward(truth, obs).logLikelihood,
    learning,
  };
}
