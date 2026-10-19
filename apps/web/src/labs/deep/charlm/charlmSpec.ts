import { CELL_NAMES, cellInfo } from '@sim/deep';
import type { ParamSpec } from '@/shared/lab-kit';
import { grammarValidity, wordValidity, type CharLmRun, type CharLmScenario } from './charlmCore';
import { CORPORA, type CorpusId } from './charlmData';

export const charlmParams: ParamSpec[] = [
  {
    key: 'corpus',
    label: 'Text to learn',
    kind: 'select',
    options: (Object.keys(CORPORA) as CorpusId[]).map((id) => ({
      value: id,
      label: CORPORA[id].label,
    })),
  },
  {
    key: 'cell',
    label: 'Recurrent cell',
    kind: 'segmented',
    options: CELL_NAMES.map((c) => ({ value: c, label: cellInfo[c].label })),
  },
  {
    key: 'hidden',
    label: 'Memory size (hidden units)',
    kind: 'ladder',
    values: [2, 4, 8, 16, 32, 48],
    format: (v) => String(v),
  },
  {
    key: 'learningRate',
    label: 'Learning rate',
    kind: 'ladder',
    values: [0.002, 0.005, 0.02, 0.05],
    format: (v) => String(v),
  },
  {
    key: 'temperature',
    label: 'Sampling temperature',
    kind: 'slider',
    min: 0.1,
    max: 3,
    step: 0.1,
    format: (v) => v.toFixed(1),
    hint: 'Low = always the likeliest character; high = more adventurous (and more typos).',
  },
];

export interface CharLmExperiment {
  id: string;
  title: string;
  text: string;
  scenario: Partial<CharLmScenario>;
  claim: (run: CharLmRun) => boolean;
}

export const charlmExperiments: CharLmExperiment[] = [
  {
    id: 'untrained',
    title: 'Before any learning',
    text: 'An untrained network guesses every character with about equal probability: its loss is that of a uniform guess (the log of the alphabet size) and what it writes is a random scramble.',
    scenario: { corpus: 'grammar', steps: 0 },
    claim: (r) =>
      Math.abs(r.trainLoss - r.uniformLoss) < 0.1 * r.uniformLoss &&
      grammarValidity(r.sample) === 0,
  },
  {
    id: 'learns-grammar',
    title: 'Learning a grammar',
    text: 'After 400 small updates the network writes complete sentences that all obey the template: the words are spelled correctly, in the right order. It has discovered the structure from characters alone.',
    scenario: { corpus: 'grammar', steps: 400 },
    claim: (r) => grammarValidity(r.sample) > 0.95 && r.trainLoss < 0.25,
  },
  {
    id: 'entropy-floor',
    title: 'The floor is not zero',
    text: 'In the grammar the animal, the verb and the place are drawn at random, so even a perfect model cannot know them in advance. The loss levels off at the entropy of those choices (about 0.12 nats per character); a loss of exactly zero would mean memorising the text.',
    scenario: { corpus: 'grammar', steps: 600 },
    claim: (r) => r.trainLoss > 0.1 && r.trainLoss < 0.25,
  },
  {
    id: 'tiny-memory',
    title: 'Memory too small',
    text: 'With two hidden units the network cannot keep track of where it is in a sentence: its loss stays high and its sentences are mostly broken. Eight units are enough.',
    scenario: { corpus: 'grammar', hidden: 2, steps: 400 },
    claim: (r) => r.trainLoss > 0.8 && grammarValidity(r.sample) < 0.5,
  },
  {
    id: 'memorise-rhyme',
    title: 'Memorising a rhyme',
    text: 'The nursery rhyme is short and repeated: the network learns it by heart in a hundred steps and recites it.',
    scenario: { corpus: 'rhyme', steps: 150 },
    claim: (r) =>
      r.trainLoss < 0.3 && wordValidity(r.sample, 'rhyme') > 0.9 && /wonder/.test(r.sample),
  },
  {
    id: 'overfit',
    title: 'Learning Hamlet by heart',
    text: "Shakespeare's lines are too varied to generalise from. The network reproduces the training text almost exactly (training loss under 0.5) but on the last 80 characters, which it never saw, it does worse than guessing at random.",
    scenario: { corpus: 'hamlet', steps: 400 },
    claim: (r) => r.trainLoss < 0.5 && r.validationLoss > r.uniformLoss,
  },
  {
    id: 'cold',
    title: 'Low temperature: careful',
    text: 'Sampling at temperature 0.2 almost always takes the most likely next character: nearly every word is a real word from the text, though it may loop.',
    scenario: { corpus: 'hamlet', hidden: 48, steps: 500, temperature: 0.2 },
    claim: (r) => wordValidity(r.sample, 'hamlet') > 0.9,
  },
  {
    id: 'hot',
    title: 'High temperature: reckless',
    text: 'At temperature 3 the distribution is flattened: unlikely characters are drawn almost as often as likely ones, and most of what comes out is not a word.',
    scenario: { corpus: 'hamlet', hidden: 48, steps: 500, temperature: 3 },
    claim: (r) => wordValidity(r.sample, 'hamlet') < 0.3,
  },
];
