import type { ParamSpec } from '@/shared/lab-kit';
import type { HmmRun, HmmScenario } from './hmmCore';

export const hmmParams: ParamSpec[] = [
  {
    key: 'stay',
    label: 'Chance of keeping the same die',
    kind: 'slider',
    min: 0.5,
    max: 0.99,
    step: 0.01,
    format: (v) => v.toFixed(2),
    hint: 'Low = the croupier swaps all the time; high = long runs with one die.',
  },
  {
    key: 'sixes',
    label: 'Chance of a six with the loaded die',
    kind: 'slider',
    min: 0.1,
    max: 0.9,
    step: 0.01,
    format: (v) => v.toFixed(2),
    hint: 'A fair die gives 0.17.',
  },
  {
    key: 'length',
    label: 'Rolls',
    kind: 'ladder',
    values: [30, 60, 100, 200, 400, 600],
    format: (v) => String(v),
  },
  {
    key: 'inference',
    label: 'The model used to read the rolls',
    kind: 'segmented',
    options: [
      { value: 'truth', label: 'The true model' },
      { value: 'naive', label: 'A naive guess' },
      { value: 'learned', label: 'Learned (Baum-Welch)' },
    ],
  },
  {
    key: 'iterations',
    label: 'Baum-Welch iterations',
    kind: 'slider',
    min: 1,
    max: 100,
    step: 1,
    showIf: (v) => v.inference === 'learned',
  },
];

export interface HmmExperiment {
  id: string;
  title: string;
  text: string;
  scenario: Partial<HmmScenario>;
  claim: (run: HmmRun) => boolean;
}

export const hmmExperiments: HmmExperiment[] = [
  {
    id: 'casino',
    title: 'Catch the croupier',
    text: 'Knowing the true model, the filter reads the die from the rolls alone: runs of sixes raise the probability of the loaded die, and the Viterbi path follows the real switches closely.',
    scenario: {},
    claim: (r) => r.viterbiAccuracy > 0.88 && r.posteriorAccuracy > 0.88,
  },
  {
    id: 'blatant',
    title: 'A blatantly loaded die',
    text: 'A die that shows a six 80% of the time gives itself away within a few rolls.',
    scenario: { sixes: 0.8 },
    claim: (r) => r.posteriorAccuracy > 0.93,
  },
  {
    id: 'flicker',
    title: 'Swapping all the time',
    text: 'When the croupier swaps dice 30% of the time there are no long runs to learn from: even the true model labels barely two thirds of the rolls correctly.',
    scenario: { stay: 0.7 },
    claim: (r) => r.viterbiAccuracy < 0.7,
  },
  {
    id: 'subtle',
    title: 'An undetectable cheat',
    text: 'If the loaded die is almost fair (a six 17% of the time) the two states emit the same rolls: no algorithm can tell them apart, and the accuracy is that of a coin flip.',
    scenario: { sixes: 0.17 },
    claim: (r) => r.viterbiAccuracy < 0.6,
  },
  {
    id: 'naive',
    title: 'A naive model',
    text: 'Reading the same rolls with a wrong model (it expects frequent swaps and only a mildly loaded die) makes the sequence look less likely and the states less reliable.',
    scenario: { inference: 'naive' },
    claim: (r) => r.viterbiAccuracy < 0.8 && r.trueLogLikelihood - r.logLikelihood > 15,
  },
  {
    id: 'learn-short',
    title: 'Learning from 200 rolls',
    text: 'Baum-Welch starts from the naive guess and climbs the likelihood. With few rolls it ends up explaining them better than the truth does, which means it overfits: the dice it describes are not the real ones.',
    scenario: { inference: 'learned' },
    claim: (r) => r.logLikelihood > r.trueLogLikelihood && r.posteriorAccuracy < 0.75,
  },
  {
    id: 'learn-long',
    title: 'Learning from 600 rolls',
    text: 'With three times the rolls the learned model becomes about as good as the true one, both in likelihood and in naming the die.',
    scenario: { inference: 'learned', length: 600, iterations: 80 },
    claim: (r) => r.posteriorAccuracy > 0.83 && Math.abs(r.logLikelihood - r.trueLogLikelihood) < 8,
  },
];
