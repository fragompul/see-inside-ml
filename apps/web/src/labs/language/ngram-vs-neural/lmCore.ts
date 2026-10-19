import { createRng } from '@sim/core';
import {
  CharTrainer,
  DEFAULT_CHAR_TRAIN,
  crossEntropy,
  makeCorpus,
  trainNgram,
  type Smoothing,
} from '@sim/deep';

/** Sentences of the toy grammar "the (cat|dog|bird) (sat|ran|slept) on the (mat|hill|log)." */
const ANIMALS = ['cat', 'dog', 'bird'];
const VERBS = ['sat', 'ran', 'slept'];
const PLACES = ['mat', 'hill', 'log'];

function sentences(count: number, seed: number): string {
  const rng = createRng(seed);
  const pick = (v: string[]) => v[Math.floor(rng.next() * v.length)] as string;
  return Array.from(
    { length: count },
    () => `the ${pick(ANIMALS)} ${pick(VERBS)} on the ${pick(PLACES)}. `,
  ).join('');
}

export const TEST_SENTENCES = 200;
/** Training texts are nested: the first 10 sentences are the start of the first 30, and so on. */
export const trainText = (count: number): string => sentences(count, 21);
export const TEST_TEXT = sentences(TEST_SENTENCES, 99);
const ALPHABET = makeCorpus(trainText(70)).alphabet;

/**
 * The lowest cross-entropy any model can reach on this grammar: each sentence is one of 27 equally likely
 * choices, so it carries ln 27 nats, spread over the average sentence length in characters.
 */
export const FLOOR_NATS = (() => {
  const mean = (v: string[]) => v.reduce((a, w) => a + w.length, 0) / v.length;
  const length =
    'the '.length +
    mean(ANIMALS) +
    1 +
    mean(VERBS) +
    ' on the '.length +
    mean(PLACES) +
    '. '.length;
  return Math.log(ANIMALS.length * VERBS.length * PLACES.length) / length;
})();

export interface LmScenario {
  /** Sentences of training text. */
  trainSentences: number;
  smoothing: Smoothing;
  /** Training steps of the neural model (0 skips it). */
  neuralSteps: number;
}

export const DEFAULT_LM: LmScenario = {
  trainSentences: 70,
  smoothing: 'interpolated',
  neuralSteps: 400,
};

export const ORDERS = [1, 2, 3, 4, 5, 6, 8, 10];

export interface LmRun {
  scenario: LmScenario;
  alphabet: number;
  /** Cross-entropy in nats per character of each order on the training text and on the new text. */
  train: number[];
  test: number[];
  /** Neural model on the new text (NaN when not trained) and its training text loss. */
  neuralTest: number;
  neuralTrain: number;
  uniform: number;
  floor: number;
}

const neuralCache = new Map<string, { test: number; train: number }>();

function neuralLoss(sentenceCount: number, steps: number, test: Int32Array) {
  const key = `${sentenceCount}|${steps}`;
  const hit = neuralCache.get(key);
  if (hit) return hit;
  const corpus = corpusOf(sentenceCount);
  const trainer = new CharTrainer(corpus, {
    ...DEFAULT_CHAR_TRAIN,
    hidden: 32,
    learningRate: 0.02,
  });
  for (let i = 0; i < steps; i++) trainer.step();
  const out = { test: trainer.model.evaluate(test), train: trainer.model.evaluate(corpus.tokens) };
  neuralCache.set(key, out);
  return out;
}

/** The text as numbers, with one shared alphabet so that every training size sees the same vocabulary. */
function corpusOf(count: number) {
  const text = trainText(count);
  return {
    text,
    alphabet: ALPHABET,
    tokens: Int32Array.from([...text].map((c) => ALPHABET.indexOf(c))),
  };
}

export function runLm(s: LmScenario): LmRun {
  const corpus = corpusOf(s.trainSentences);
  const test = Int32Array.from([...TEST_TEXT].map((c) => ALPHABET.indexOf(c)));
  const vocab = ALPHABET.length;
  const models = ORDERS.map((n) => trainNgram(corpus.tokens, vocab, n, s.smoothing));
  const neural =
    s.neuralSteps > 0
      ? neuralLoss(s.trainSentences, s.neuralSteps, test)
      : { test: Number.NaN, train: Number.NaN };
  return {
    scenario: s,
    alphabet: vocab,
    train: models.map((m) => crossEntropy(m, corpus.tokens)),
    test: models.map((m) => crossEntropy(m, test)),
    neuralTest: neural.test,
    neuralTrain: neural.train,
    uniform: Math.log(vocab),
    floor: FLOOR_NATS,
  };
}
