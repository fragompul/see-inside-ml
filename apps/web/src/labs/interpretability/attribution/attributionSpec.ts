import type { AttributionMethod } from '@sim/deep';
import type { ParamSpec } from '@/shared/lab-kit';
import {
  EXAMPLES,
  METHOD_LABEL,
  STEPS,
  type AttributionResult,
  type AttributionScenario,
} from './attributionCore';

export const attributionParams: ParamSpec[] = [
  {
    key: 'example',
    label: 'Digit',
    kind: 'ladder',
    values: Array.from({ length: EXAMPLES }, (_, i) => i),
    format: (v) => `#${v + 1}`,
  },
  {
    key: 'method',
    label: 'Method shown',
    kind: 'select',
    options: (Object.keys(METHOD_LABEL) as AttributionMethod[]).map((id) => ({
      value: id,
      label: METHOD_LABEL[id],
    })),
  },
  {
    key: 'steps',
    label: 'Integration steps',
    kind: 'ladder',
    values: STEPS,
    format: (v) => String(v),
    hint: 'For integrated gradients.',
  },
];

export interface AttributionExperiment {
  id: string;
  title: string;
  text: string;
  scenario: Partial<AttributionScenario>;
  claim: (run: AttributionResult) => boolean;
}

const step = (r: AttributionResult, n: number) => r.completeness[STEPS.indexOf(n)] as number;

export const attributionExperiments: AttributionExperiment[] = [
  {
    id: 'completeness',
    title: 'Integrated gradients add up exactly',
    text: 'The attributions of integrated gradients sum to the difference between the network’s score for the image and for a blank image. With one step the sum is off by about 20%; with 32 it is off by 0.6% and with 128 by 0.2%. For the digit shown, 4.31 against 4.34.',
    scenario: { example: 0, steps: 32, method: 'integrated' },
    claim: (r) =>
      step(r, 1) > 0.1 &&
      step(r, 32) < 0.01 &&
      step(r, 128) < 0.005 &&
      r.completeness.every((v, i) => i === 0 || v < (r.completeness[i - 1] as number)) &&
      Math.abs(r.sum - r.gap) < 0.02 * Math.abs(r.gap),
  },
  {
    id: 'faithful',
    title: 'Faithful maps break the answer faster',
    text: 'Blank the pixels one chunk at a time, most important first, and track the score of the class. With integrated gradients or gradient × input the score collapses (average −6); with the size of the gradient alone it hardly moves (−0.8), though even that beats removing pixels at random (+0.5).',
    scenario: { method: 'integrated' },
    claim: (r) =>
      r.deletion.integrated < r.deletion.saliency - 4 &&
      r.deletion['gradient-input'] < r.deletion.saliency - 4 &&
      r.deletion.saliency < r.randomDeletion,
  },
  {
    id: 'smoothing',
    title: 'Smoother is not truer',
    text: 'SmoothGrad averages the gradient over noisy copies and gives a calmer picture, but it is no more faithful: its deletion score (−0.9) is within half a point of plain saliency (−0.8) and nowhere near integrated gradients.',
    scenario: { method: 'smoothgrad' },
    claim: (r) =>
      Math.abs(r.deletion.smoothgrad - r.deletion.saliency) < 0.5 &&
      r.deletion.smoothgrad > r.deletion.integrated + 4,
  },
  {
    id: 'sanity',
    title: 'A sanity check: does the map depend on what was learned?',
    text: 'Replace the output layer by random weights and compute the maps again. The maps of saliency and SmoothGrad stay partly alike (correlation 0.33 and 0.45): much of them is edges of the image, which is there whatever the network learned. Gradient × input and integrated gradients drop to zero.',
    scenario: { method: 'saliency' },
    claim: (r) =>
      r.sanity.saliency > 0.25 &&
      r.sanity.smoothgrad > 0.25 &&
      Math.abs(r.sanity['gradient-input']) < 0.1 &&
      Math.abs(r.sanity.integrated) < 0.1,
  },
  {
    id: 'sign',
    title: 'Evidence has a sign',
    text: 'The size of a gradient cannot tell evidence for the class from evidence against it. Integrated gradients can: some pixels of the same digit push the score down, and these negative attributions are what saliency discards.',
    scenario: { method: 'integrated' },
    claim: (r) => {
      const ig = r.maps.integrated;
      return Math.min(...ig) < -0.01 && Math.max(...ig) > 0.01 && Math.min(...r.maps.saliency) >= 0;
    },
  },
];
