import type { ParamSpec } from '@/shared/lab-kit';
import { runProbing, type ProbingResult, type ProbingScenario } from './probingCore';

export const probingParams: ParamSpec[] = [
  {
    key: 'model',
    label: 'Model to probe',
    kind: 'segmented',
    options: [
      { value: 'digits', label: 'Digit network' },
      { value: 'story', label: 'Story model' },
    ],
  },
];

export interface ProbingExperiment {
  id: string;
  title: string;
  text: string;
  scenario: Partial<ProbingScenario>;
  claim: (run: ProbingResult) => Promise<boolean> | boolean;
}

const task = (r: ProbingResult, start: string) => {
  const found = r.tasks.find((t) => t.name.startsWith(start));
  if (!found) throw new Error(`no task ${start}`);
  return found.test;
};
const other = (model: ProbingScenario['model']) => runProbing({ model });

export const probingExperiments: ProbingExperiment[] = [
  {
    id: 'control',
    title: 'A wide layer lets the probe memorise noise',
    text: 'On the 256 raw pixels a probe fits random labels perfectly on its training digits (100%) and scores chance on new ones (11%). Three layers later, with only 16 units, it can no longer memorise: 17% on its training digits. The control task shows how much of a probe’s score could be its own doing.',
    scenario: { model: 'digits' },
    claim: (r) =>
      (r.controlTrain[0] as number) >= 0.95 &&
      r.controlTest.every((v) => v < 0.2) &&
      (r.controlTrain[3] as number) < 0.3,
  },
  {
    id: 'ink',
    title: 'Some facts about the picture survive every layer',
    text: 'How much ink a digit has can be read off the pixels, and still can after the first, second and third hidden layers, at 80% to 87% against a chance level of 33%. A network trained only to name digits keeps the stroke weight around.',
    scenario: { model: 'digits' },
    claim: (r) => task(r, 'How much ink').every((v) => v >= 0.78),
  },
  {
    id: 'position',
    title: 'Others fade with depth',
    text: 'Whether a digit sits to the left, in the middle or to the right can be read from the pixels at 85%. After each further layer it is harder: 79%, 59% and 54%. Naming a digit does not need to know where it is, so the network discards it.',
    scenario: { model: 'digits' },
    claim: (r) => {
      const t = task(r, 'Where it sits');
      return (
        (t[0] as number) > 0.8 &&
        (t[3] as number) < 0.6 &&
        t.every((v, i) => i === 0 || v < (t[i - 1] as number))
      );
    },
  },
  {
    id: 'moves',
    title: 'The gender moves from the name to the full stop',
    text: 'In the story model the embedding of a name already tells its gender (a probe trained on twelve names gets four unseen ones right every time), while at the full stop that ends the sentence the embedding says nothing (50%). After the first layer the full stop carries the gender perfectly: attention has copied it there.',
    scenario: { model: 'story' },
    claim: (r) => {
      const name = task(r, 'Gender of the name, read at the name');
      const stop = task(r, 'Gender of the name, read at the full stop');
      return (
        (name[0] as number) >= 0.95 &&
        (stop[0] as number) <= 0.6 &&
        (stop[1] as number) >= 0.95 &&
        (stop[2] as number) >= 0.95
      );
    },
  },
  {
    id: 'used',
    title: 'Information is used, then dropped',
    text: 'Which object was found can be read at the full stop after the first layer (81% against 13% by chance) and hardly at all after the second (28%), once it has done its job. At the name itself the gender is also gone by then: the probe, which worked on the embeddings, gets unseen names wrong.',
    scenario: { model: 'story' },
    claim: (r) => {
      const object = task(r, 'Which object');
      const name = task(r, 'Gender of the name, read at the name');
      return (
        (object[1] as number) > 0.7 && (object[2] as number) < 0.4 && (name[2] as number) <= 0.5
      );
    },
  },
  {
    id: 'both-controls',
    title: 'Both models pass the control',
    text: 'For both models the probe cannot generalise random labels to new examples: scores stay near chance on the held-out data, so the high scores above are about the layers and not about the probe.',
    scenario: { model: 'story' },
    claim: async (r) => {
      const digits = await other('digits');
      return digits.controlTest.every((v) => v < 0.2) && r.controlTest.every((v) => v < 0.7);
    },
  },
];
