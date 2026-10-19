import type { ParamSpec } from '@/shared/lab-kit';
import {
  DESIGN_LABEL,
  runConvArith,
  type ConvArithRun,
  type ConvArithScenario,
  type Design,
} from './convArithCore';

export const convArithParams: ParamSpec[] = [
  {
    key: 'design',
    label: 'Design',
    kind: 'select',
    options: (Object.keys(DESIGN_LABEL) as Design[]).map((id) => ({
      value: id,
      label: DESIGN_LABEL[id],
    })),
  },
  {
    key: 'depth',
    label: 'Convolution layers',
    kind: 'ladder',
    values: [1, 2, 3, 4, 6, 8],
    format: (v) => String(v),
  },
  {
    key: 'kernel',
    label: 'Kernel width',
    kind: 'ladder',
    values: [3, 5, 7],
    format: (v) => String(v),
  },
  {
    key: 'input',
    label: 'Input width',
    kind: 'ladder',
    values: [64, 128, 224],
    format: (v) => String(v),
  },
];

export interface ConvArithExperiment {
  id: string;
  title: string;
  text: string;
  scenario: Partial<ConvArithScenario>;
  claim: (run: ConvArithRun) => boolean;
}

const plain = (r: ConvArithRun): ConvArithRun => runConvArith({ ...r.scenario, design: 'plain' });

export const convArithExperiments: ConvArithExperiment[] = [
  {
    id: 'small-kernels',
    title: 'Three small filters see what one big one sees',
    text: 'Three 3-wide layers in a row let a value see 7 positions, exactly like one 7-wide filter, but with 6,960 weights instead of 12,560 (for 16 channels in an image, where a filter is square). Stacking small filters is cheaper and adds extra non-linearities.',
    scenario: { design: 'plain', depth: 3, kernel: 3, input: 64 },
    claim: (r) => r.finalField === 7 && r.parameters < 0.6 * r.singleKernelParameters,
  },
  {
    id: 'slow-growth',
    title: 'Plain layers see very little',
    text: 'Every extra plain layer adds only two positions to what a value can see. Eight layers see 17 positions: a quarter of a 64-wide input, and less than a tenth of a 224-wide one.',
    scenario: { design: 'plain', depth: 8, input: 64 },
    claim: (r) => r.finalField === 17 && r.coverage < 0.3 && r.finalSize === 64,
  },
  {
    id: 'pooling',
    title: 'Pooling widens the view for free',
    text: 'Pooling has no weights, but after it every step in the next layer covers two input positions. Three convolutions with pooling see 22 positions, three times as many as without, with exactly the same number of weights, and a picture eight times smaller.',
    scenario: { design: 'pooled', depth: 3, input: 64 },
    claim: (r) =>
      r.finalField > 3 * plain(r).finalField &&
      r.parameters === plain(r).parameters &&
      r.finalSize === 8,
  },
  {
    id: 'stride',
    title: 'Stride trades resolution for reach',
    text: 'Using stride 2 in every second layer does the same job as pooling. Six layers see 29 positions instead of 13, and the picture is cut to an eighth of its width on the way.',
    scenario: { design: 'strided', depth: 6, input: 224 },
    claim: (r) => r.finalField === 29 && plain(r).finalField === 13 && r.finalSize === 28,
  },
  {
    id: 'dilation',
    title: 'Dilation widens the view without shrinking the picture',
    text: 'Spacing out the taps of the filter, doubling the gap at each layer, makes the field grow exponentially: eight layers see 511 positions while keeping all 224 outputs, with the same weights as the plain stack that sees 17.',
    scenario: { design: 'dilated', depth: 8, input: 224 },
    claim: (r) =>
      r.finalField === 511 &&
      r.finalSize === 224 &&
      r.parameters === plain(r).parameters &&
      plain(r).finalField === 17,
  },
  {
    id: 'traced',
    title: 'Counted by hand, the same answer',
    text: 'The receptive field is computed from a formula, and checked here by following one output value back through every layer and noting which input positions it reads. The two agree whenever the input is wide enough to hold the whole field.',
    scenario: { design: 'pooled', depth: 6, input: 224 },
    claim: (r) => r.traced === r.finalField && r.finalField < r.scenario.input,
  },
];
