import type { ParamSpec } from '@/shared/lab-kit';
import {
  REGULARISER_LABEL,
  runVis,
  type Regulariser,
  type VisResult,
  type VisScenario,
} from './visCore';

export const visParams: ParamSpec[] = [
  {
    key: 'kind',
    label: 'What to visualise',
    kind: 'segmented',
    options: [
      { value: 'class', label: 'A digit class' },
      { value: 'channel', label: 'A feature map' },
    ],
  },
  {
    key: 'regulariser',
    label: 'Keep the picture natural by',
    kind: 'select',
    options: (Object.keys(REGULARISER_LABEL) as Regulariser[]).map((id) => ({
      value: id,
      label: REGULARISER_LABEL[id],
    })),
  },
  {
    key: 'steps',
    label: 'Gradient steps',
    kind: 'ladder',
    values: [10, 30, 120],
    format: (v) => String(v),
  },
];

export interface VisExperiment {
  id: string;
  title: string;
  text: string;
  scenario: Partial<VisScenario>;
  claim: (run: VisResult) => boolean;
}

const sibling = (r: VisResult, change: Partial<VisScenario>) =>
  runVis({ ...r.scenario, ...change });

export const visExperiments: VisExperiment[] = [
  {
    id: 'super',
    title: 'The network can be made more sure than any real digit',
    text: 'Climbing the score of a class with nothing but a limit on pixel values reaches a score of 17, more than four times the 3.7 of a real digit of that class, and every picture is assigned to the class it was made for. What excites the network most is not a typical digit.',
    scenario: { kind: 'class', regulariser: 'none' },
    claim: (r) => r.meanScore > 4 * r.realScore && r.hitRate === 1,
  },
  {
    id: 'noise',
    title: 'And it looks like static',
    text: 'The unregularised pictures vary from pixel to pixel almost three times as much as real digits (total variation 179 against 64): they excite the network through fine patterns that a person would not recognise as a digit.',
    scenario: { kind: 'class', regulariser: 'none' },
    claim: (r) => r.meanNoise > 2.5 * r.realNoise,
  },
  {
    id: 'decay',
    title: 'Pulling towards zero brings back the digit',
    text: 'Removing a little of the picture at every step stops the ink from spreading over the whole frame. The pictures now correlate twice as strongly with the average real digit of their class (0.47 against 0.22) and are less noisy (106).',
    scenario: { kind: 'class', regulariser: 'decay' },
    claim: (r) => {
      const none = sibling(r, { regulariser: 'none' });
      return r.meanLikeness > 1.8 * none.meanLikeness && r.meanNoise < 0.7 * none.meanNoise;
    },
  },
  {
    id: 'blur',
    title: 'Blurring gives smooth pictures, at a price',
    text: 'Blurring a little at every step produces smooth pictures, smoother even than real digits (total variation 28 against 64), but they score only 6.1 against 17.3: every regulariser trades some of what the unit wants for a more natural image.',
    scenario: { kind: 'class', regulariser: 'blur' },
    claim: (r) => {
      const none = sibling(r, { regulariser: 'none' });
      const decay = sibling(r, { regulariser: 'decay' });
      return (
        r.meanNoise < 0.6 * r.realNoise &&
        none.meanScore > decay.meanScore &&
        decay.meanScore > r.meanScore
      );
    },
  },
  {
    id: 'units',
    title: 'The same method on a single feature map',
    text: 'Asking instead what one feature map of the last convolution prefers gives the same story: unrestrained, a field of static (179); with blur, soft blobs and strokes (total variation 15, a quarter of a real digit’s).',
    scenario: { kind: 'channel', regulariser: 'blur' },
    claim: (r) =>
      r.meanNoise < 0.35 * r.realNoise &&
      sibling(r, { regulariser: 'none' }).meanNoise > 2.5 * r.realNoise,
  },
];
