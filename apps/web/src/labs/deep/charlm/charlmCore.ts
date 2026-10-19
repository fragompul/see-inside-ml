import { CharTrainer, DEFAULT_CHAR_TRAIN, type CellName } from '@sim/deep';
import { CORPORA, corpusOf, type CorpusId } from './charlmData';

export interface CharLmScenario {
  corpus: CorpusId;
  cell: CellName;
  hidden: number;
  learningRate: number;
  /** Optimiser steps trained for (each step is a batch of windows). */
  steps: number;
  temperature: number;
  seed: number;
}

export const DEFAULT_CHARLM: CharLmScenario = {
  corpus: 'grammar',
  cell: 'gru',
  hidden: 32,
  learningRate: 0.02,
  steps: 400,
  temperature: 0.7,
  seed: 1,
};

/** Characters at the end of the corpus that are never trained on (Hamlet only: it is long enough to split). */
export const holdoutOf = (corpus: CorpusId): number => (corpus === 'hamlet' ? 80 : 0);

export const makeTrainer = (s: CharLmScenario): CharTrainer =>
  new CharTrainer(
    corpusOf(s.corpus),
    {
      ...DEFAULT_CHAR_TRAIN,
      cell: s.cell,
      hidden: s.hidden,
      learningRate: s.learningRate,
      seed: s.seed,
    },
    holdoutOf(s.corpus),
  );

const SENTENCE = /^the (cat|dog|bird) (sat|ran|slept) on the (mat|hill|log)$/;

/** Share of the complete sentences of a sample that follow the toy grammar. */
export function grammarValidity(sample: string): number {
  const sentences = sample
    .split('.')
    .slice(1, -1)
    .map((x) => x.trim());
  return sentences.length ? sentences.filter((x) => SENTENCE.test(x)).length / sentences.length : 0;
}

/** Share of the words of a sample that occur in the training text. */
export function wordValidity(sample: string, corpus: CorpusId): number {
  const clean = (w: string) => w.replace(/[^a-z']/g, '');
  const known = new Set(CORPORA[corpus].text.split(/\s+/).map(clean));
  const words = sample.split(/\s+/).map(clean).filter(Boolean);
  return words.length ? words.filter((w) => known.has(w)).length / words.length : 0;
}

export interface CharLmRun {
  scenario: CharLmScenario;
  trainer: CharTrainer;
  /** Mean training loss over the last 20 steps, in nats per character. */
  trainLoss: number;
  /** Loss on the held-out tail (or on the training text when nothing is held out). */
  validationLoss: number;
  /** Loss of a model that guesses uniformly among the characters. */
  uniformLoss: number;
  sample: string;
}

export function runCharLm(s: CharLmScenario): CharLmRun {
  const trainer = makeTrainer(s);
  for (let i = 0; i < s.steps; i++) trainer.step();
  const tail = trainer.history.slice(-20);
  return {
    scenario: s,
    trainer,
    trainLoss: tail.length
      ? tail.reduce((a, b) => a + b, 0) / tail.length
      : Math.log(trainer.corpus.alphabet.length),
    validationLoss: trainer.validationLoss(),
    uniformLoss: Math.log(trainer.corpus.alphabet.length),
    sample: trainer.generate(
      240,
      s.temperature,
      4,
      s.corpus === 'hamlet' ? 'to ' : s.corpus === 'grammar' ? 'the ' : 'twinkle',
    ),
  };
}
