import type { ParamSpec } from '@/shared/lab-kit';
import { WEIGHTS, STEP_COUNTS, type GuidanceResult, type GuidanceScenario } from './guidanceCore';

export const guidanceParams: ParamSpec[] = [
  {
    key: 'mode',
    label: 'View',
    kind: 'segmented',
    options: [
      { value: 'guidance', label: 'Guidance' },
      { value: 'flow', label: 'Flow matching' },
    ],
  },
  {
    key: 'weight',
    label: 'Guidance weight w',
    kind: 'ladder',
    values: WEIGHTS,
    format: (v) => String(v),
    hint: '0 is the conditional model alone.',
    showIf: (v) => v.mode === 'guidance',
  },
  {
    key: 'asked',
    label: 'Class asked for',
    kind: 'ladder',
    values: [0, 1, 2, 3, 4, 5],
    format: (v) => String(v + 1),
    showIf: (v) => v.mode === 'guidance',
  },
  {
    key: 'steps',
    label: 'Sampling steps',
    kind: 'ladder',
    values: STEP_COUNTS,
    format: (v) => String(v),
    showIf: (v) => v.mode === 'flow',
  },
];

export interface GuidanceExperiment {
  id: string;
  title: string;
  text: string;
  scenario: Partial<GuidanceScenario>;
  claim: (run: GuidanceResult) => boolean;
}

const at = (values: number[], weight: number) => values[WEIGHTS.indexOf(weight)] as number;
const steps = (values: number[], n: number) => values[STEP_COUNTS.indexOf(n)] as number;

export const guidanceExperiments: GuidanceExperiment[] = [
  {
    id: 'sharpen',
    title: 'A little guidance sharpens the class',
    text: 'The six classes overlap, so a model that merely knows the class still puts about 1% of its points nearer a neighbour. With guidance weight 1 every point lands in the class asked for, and the points bunch closer to the class centre.',
    scenario: { mode: 'guidance', weight: 1 },
    claim: (r) =>
      r.accuracy[1] === 1 &&
      r.accuracy[1] > (r.accuracy[0] as number) &&
      at(r.spread, 1) < 0.75 * at(r.spread, 0),
  },
  {
    id: 'variety',
    title: 'Sharper means less varied',
    text: 'Real points sit about 0.53 from their blob centre on average. Guided samples sit at 0.25, less than half that: more typical, and much less like the real spread. Guidance trades variety for fidelity to the prompt.',
    scenario: { mode: 'guidance', weight: 1 },
    claim: (r) => at(r.spread, 1) < 0.55 * r.realSpread && at(r.spread, 0) > 0.7 * r.realSpread,
  },
  {
    id: 'too-much',
    title: 'Too much guidance breaks the picture',
    text: 'Already at weight 3 the prediction is pushed past the data and only 68% of the points land in the right class. At weight 8 they overshoot and fly off: 12% are right, and the average distance from the centre jumps from 0.4 to over 7.',
    scenario: { mode: 'guidance', weight: 8 },
    claim: (r) => at(r.accuracy, 3) < 0.8 && at(r.accuracy, 8) < 0.2 && at(r.spread, 8) > 5,
  },
  {
    id: 'few-steps',
    title: 'Flow matching needs far fewer steps',
    text: 'With the same network, data and training, four Euler steps along the learned field come closer to the data than 32 steps of DDIM diffusion (distance 0.23 against 0.35), and far closer than four DDIM steps (1.05).',
    scenario: { mode: 'flow', steps: 4 },
    claim: (r) =>
      steps(r.flowDistance, 4) < steps(r.diffusionDistance, 32) &&
      steps(r.flowDistance, 4) < 0.25 * steps(r.diffusionDistance, 4),
  },
  {
    id: 'one-step',
    title: 'One step is not enough, eight nearly are',
    text: 'A single Euler step jumps along the average direction and lands at distance 0.92. Two steps reach 0.39, four 0.23 and eight 0.13, within a factor of two of what 32 steps achieve (0.10).',
    scenario: { mode: 'flow', steps: 1 },
    claim: (r) => {
      const d = [1, 2, 4, 8, 16].map((n) => steps(r.flowDistance, n));
      return (
        d.every((v, i) => i === 0 || v < (d[i - 1] as number)) &&
        steps(r.flowDistance, 8) < 2 * steps(r.flowDistance, 32)
      );
    },
  },
];
