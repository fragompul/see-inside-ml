import type { ParamSpec } from '@/shared/lab-kit';
import {
  METHOD_LABELS,
  runLora,
  type LoraRun,
  type LoraScenario,
  type TuneMethod,
} from './loraCore';

export const loraParams: ParamSpec[] = [
  {
    key: 'method',
    label: 'How to adapt',
    kind: 'segmented',
    options: (Object.keys(METHOD_LABELS) as TuneMethod[]).map((m) => ({
      value: m,
      label: METHOD_LABELS[m],
    })),
  },
  {
    key: 'rank',
    label: 'LoRA rank',
    kind: 'ladder',
    values: [1, 2, 4, 8, 16],
    format: (v) => String(v),
    showIf: (v) => v.method === 'lora',
    hint: 'The size of the low-rank update: each layer learns a (inputs × rank) and a (rank × outputs) matrix.',
  },
  {
    key: 'perClass',
    label: 'Labelled examples per digit in the new situation',
    kind: 'ladder',
    values: [2, 5, 20],
    format: (v) => String(v),
  },
  {
    key: 'learningRate',
    label: 'Learning rate',
    kind: 'ladder',
    values: [0.001, 0.003, 0.01],
    format: (v) => String(v),
  },
  {
    key: 'epochs',
    label: 'Epochs',
    kind: 'ladder',
    values: [10, 20, 40, 80],
    format: (v) => String(v),
  },
];

export interface LoraExperiment {
  id: string;
  title: string;
  text: string;
  scenario: Partial<LoraScenario>;
  claim: (run: LoraRun) => boolean;
}

export const loraExperiments: LoraExperiment[] = [
  {
    id: 'shift',
    title: 'A small change breaks the model',
    text: 'The network reads upright digits at 99%. Shift every image three pixels right and two down (the camera moved) and it gets 5%: worse than chance. Nothing in what it learned is wrong, it simply is not where it expects.',
    scenario: { method: 'none' },
    claim: (r) => r.originalBefore > 0.97 && r.shiftedBefore < 0.1,
  },
  {
    id: 'head',
    title: 'Retraining only the last layer',
    text: 'With 5 labelled examples per digit, re-learning only the last layer (330 numbers) lifts accuracy from 5% to about half: the features it was given are not the right ones for shifted digits.',
    scenario: { method: 'head' },
    claim: (r) => r.shiftedAfter > 0.3 && r.shiftedAfter < 0.65,
  },
  {
    id: 'full',
    title: 'Full fine-tuning',
    text: 'Updating all 18,858 numbers adapts best, to about 90%, but it also overwrites what the network knew: back on upright digits it now gets about a third right. This is catastrophic forgetting.',
    scenario: { method: 'full' },
    claim: (r) => r.shiftedAfter > 0.85 && r.originalAfter < 0.5,
  },
  {
    id: 'lora',
    title: 'LoRA: a tenth of the parameters',
    text: 'Freeze the network and learn, for each layer, a rank-4 update A·B: 1,832 numbers (under a tenth of the network). It recovers most of the accuracy of full fine-tuning, and since the original weights are untouched the adapter can be switched off to get the old network back.',
    scenario: { method: 'lora', rank: 4 },
    claim: (r) => r.shiftedAfter >= 0.78 && r.trainable / r.total < 0.11,
  },
  {
    id: 'rank',
    title: 'More rank, more capacity',
    text: 'The rank decides how rich the adaptation can be. At rank 1 the update is a single direction per layer and reaches 43%; rank 4 reaches 82%; rank 16 about 91%, nearly matching a full update with 40% of the parameters.',
    scenario: { method: 'lora', rank: 16 },
    claim: (r) => {
      const low = runLora({ ...r.scenario, rank: 1 });
      const mid = runLora({ ...r.scenario, rank: 4 });
      return (
        low.shiftedAfter < mid.shiftedAfter &&
        mid.shiftedAfter < r.shiftedAfter &&
        r.shiftedAfter > 0.88
      );
    },
  },
  {
    id: 'few-examples',
    title: 'Only two examples per digit',
    text: 'With 20 labelled digits in total, LoRA (58%) still beats re-training the last layer (33%), though full fine-tuning (74%) can do more with the same little data here; the more parameters you free, the more a few examples can pull.',
    scenario: { method: 'lora', rank: 4, perClass: 2 },
    claim: (r) => runLora({ ...r.scenario, method: 'head' }).shiftedAfter < r.shiftedAfter,
  },
  {
    id: 'plenty',
    title: 'With twenty per digit',
    text: 'With plenty of labelled examples LoRA reaches 97% with a tenth of the parameters, within a point or two of full fine-tuning (98%).',
    scenario: { method: 'lora', rank: 4, perClass: 20 },
    claim: (r) =>
      r.shiftedAfter > 0.95 &&
      runLora({ ...r.scenario, method: 'full' }).shiftedAfter - r.shiftedAfter < 0.03,
  },
];
