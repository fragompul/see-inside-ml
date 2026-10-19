import type { ParamSpec } from '@/shared/lab-kit';
import { ORDERS, runLm, type LmRun, type LmScenario } from './lmCore';

export const lmParams: ParamSpec[] = [
  {
    key: 'trainSentences',
    label: 'Training sentences',
    kind: 'ladder',
    values: [5, 10, 30, 70],
    format: (v) => String(v),
    hint: 'Each is one of 27 possible sentences.',
  },
  {
    key: 'smoothing',
    label: 'N-gram smoothing',
    kind: 'select',
    options: [
      { value: 'none', label: 'None (raw counts)' },
      { value: 'addk', label: 'Add a pseudo-count to everything' },
      { value: 'interpolated', label: 'Mix with shorter contexts' },
    ],
  },
  {
    key: 'neuralSteps',
    label: 'Neural training steps',
    kind: 'ladder',
    values: [0, 100, 400, 1500],
    format: (v) => (v === 0 ? 'none' : String(v)),
  },
];

export interface LmExperiment {
  id: string;
  title: string;
  text: string;
  scenario: Partial<LmScenario>;
  claim: (run: LmRun) => boolean;
}

const same = (r: LmRun, change: Partial<LmScenario>) => runLm({ ...r.scenario, ...change });
const at = (order: number) => ORDERS.indexOf(order);

export const lmExperiments: LmExperiment[] = [
  {
    id: 'context',
    title: 'Longer context, better predictions',
    text: 'Predicting from one character is hardly better than guessing (perplexity 12.5 against 18 for pure guessing); three characters give 1.3; eight reach within 3% of the best any model could do on this grammar. The price is a table that grows with every context seen.',
    scenario: { trainSentences: 70, smoothing: 'none', neuralSteps: 0 },
    claim: (r) =>
      Math.exp(r.test[at(1)] as number) > 12 &&
      Math.exp(r.test[at(3)] as number) < 1.4 &&
      (r.test[at(8)] as number) < 1.05 * r.floor &&
      r.test.slice(0, at(8) + 1).every((v, i, a) => i === 0 || v <= (a[i - 1] as number)),
  },
  {
    id: 'unseen',
    title: 'Never seen means impossible',
    text: 'With only ten training sentences a model that works from raw counts meets contexts in the new text that it has never seen, gives them probability zero, and its perplexity is infinite. Adding a small pseudo-count to every count keeps it finite.',
    scenario: { trainSentences: 10, smoothing: 'none', neuralSteps: 0 },
    claim: (r) =>
      r.test.slice(at(3)).every((v) => v === Infinity) &&
      same(r, { smoothing: 'addk' }).test.every(Number.isFinite),
  },
  {
    id: 'smoothing',
    title: 'Smoothing by backing off',
    text: 'Adding the same pseudo-count everywhere wastes probability on impossible characters: with ten sentences its best order is only 3 and it gets worse beyond. Mixing in the estimates of shorter contexts keeps improving with order and ends at half the loss.',
    scenario: { trainSentences: 10, smoothing: 'interpolated', neuralSteps: 0 },
    claim: (r) => {
      const addk = same(r, { smoothing: 'addk' }).test;
      return (
        Math.min(...r.test) < 0.6 * Math.min(...addk) &&
        r.test.indexOf(Math.min(...r.test)) >= at(6) &&
        addk.indexOf(Math.min(...addk)) <= at(4)
      );
    },
  },
  {
    id: 'neural-wins',
    title: 'A small network generalises further',
    text: 'Trained on ten sentences, the recurrent network predicts new text better than the best n-gram of any order and smoothing, because it learns what each part of a sentence has in common rather than memorising exact contexts.',
    scenario: { trainSentences: 10, smoothing: 'interpolated', neuralSteps: 400 },
    claim: (r) =>
      r.neuralTest < Math.min(...r.test) &&
      r.neuralTest < Math.min(...same(r, { smoothing: 'addk' }).test),
  },
  {
    id: 'overfit',
    title: 'Too much training memorises',
    text: 'Keep training the same network on the same ten sentences and its loss on them keeps falling while its loss on new text rises: after 1,500 steps it is worse on new sentences than after 400. This is the moment to stop.',
    scenario: { trainSentences: 10, smoothing: 'interpolated', neuralSteps: 1500 },
    claim: (r) => {
      const early = same(r, { neuralSteps: 400 });
      return r.neuralTest > 1.5 * early.neuralTest && r.neuralTrain < early.neuralTrain;
    },
  },
  {
    id: 'floor',
    title: 'The best possible score is known',
    text: 'Every sentence is one of 27 equally likely choices, so no model can do better than 0.130 nats per character, a perplexity of 1.14. With enough text and training, the network gets within 5% of it.',
    scenario: { trainSentences: 70, smoothing: 'interpolated', neuralSteps: 1500 },
    claim: (r) =>
      Math.abs(r.floor - 0.1301) < 1e-3 && r.neuralTest > r.floor && r.neuralTest < 1.05 * r.floor,
  },
];
