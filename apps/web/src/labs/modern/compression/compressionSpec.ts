import type { ParamSpec } from '@/shared/lab-kit';
import type { CompressionRun, CompressionScenario } from './compressionCore';

export const compressionParams: ParamSpec[] = [
  {
    key: 'mode',
    label: 'Technique',
    kind: 'segmented',
    options: [
      { value: 'quantize', label: 'Quantisation' },
      { value: 'prune', label: 'Pruning' },
      { value: 'distill', label: 'Distillation' },
    ],
  },
  {
    key: 'bits',
    label: 'Bits per weight',
    kind: 'ladder',
    values: [2, 3, 4, 6, 8, 32],
    format: (v) => (v === 32 ? '32 (original)' : String(v)),
    showIf: (v) => v.mode === 'quantize',
  },
  {
    key: 'sparsity',
    label: 'Weights removed',
    kind: 'ladder',
    values: [0, 0.5, 0.8, 0.9, 0.95, 0.99],
    format: (v) => `${Math.round(v * 100)}%`,
    showIf: (v) => v.mode === 'prune',
  },
  {
    key: 'fineTune',
    label: 'Fine-tune the survivors afterwards',
    kind: 'switch',
    showIf: (v) => v.mode === 'prune',
  },
  {
    key: 'labelled',
    label: 'Labelled digits the student may use alone',
    kind: 'ladder',
    values: [20, 60, 150],
    format: (v) => String(v),
    showIf: (v) => v.mode === 'distill',
  },
  {
    key: 'temperature',
    label: 'Distillation temperature',
    kind: 'ladder',
    values: [1, 3, 8],
    format: (v) => String(v),
    showIf: (v) => v.mode === 'distill',
    hint: "Higher = the teacher's probabilities are softened, so wrong-but-similar classes show up.",
  },
];

export interface CompressionExperiment {
  id: string;
  title: string;
  text: string;
  scenario: Partial<CompressionScenario>;
  claim: (run: CompressionRun) => boolean;
}

export const compressionExperiments: CompressionExperiment[] = [
  {
    id: 'eight-bits',
    title: 'Eight bits are free',
    text: 'Storing every weight in 8 bits instead of 32 makes the network four times smaller and changes its accuracy by nothing at all: the rounding error is far smaller than the weights themselves.',
    scenario: { mode: 'quantize', bits: 8 },
    claim: (r) => r.accuracy === r.teacherAccuracy && r.fullBytes / r.bytes >= 3.99,
  },
  {
    id: 'four-bits',
    title: 'Four bits: eight times smaller',
    text: 'With only 16 possible values per weight the network is eight times smaller and loses well under a point of accuracy.',
    scenario: { mode: 'quantize', bits: 4 },
    claim: (r) => r.teacherAccuracy - r.accuracy <= 0.01 && r.fullBytes / r.bytes >= 7.9,
  },
  {
    id: 'cliff',
    title: 'Two bits fall off a cliff',
    text: 'With three levels per weight (−, 0, +) most weights round to zero and the signal is destroyed: accuracy collapses. Compression is cheap until the rounding error becomes as big as the weights.',
    scenario: { mode: 'quantize', bits: 2 },
    claim: (r) => r.accuracy < 0.5,
  },
  {
    id: 'half-pruned',
    title: 'Half the weights are not needed',
    text: "Removing the 50% smallest weights, with no retraining, changes nothing: most of a trained network's weights are close to zero and contribute little.",
    scenario: { mode: 'prune', sparsity: 0.5 },
    claim: (r) => r.accuracy === r.teacherAccuracy,
  },
  {
    id: 'pruned-hard',
    title: 'Pruning 90% hurts',
    text: 'Cutting 90% of the weights in one go leaves a network that gets only about 84% of the digits right.',
    scenario: { mode: 'prune', sparsity: 0.9 },
    claim: (r) => r.accuracy < 0.9 && r.accuracy > 0.7,
  },
  {
    id: 'fine-tune',
    title: 'Fine-tuning heals the scar',
    text: 'The same 90% pruning, followed by a few epochs of training that keeps the removed weights at zero: the survivors re-learn the job and accuracy returns to within two points of the full network.',
    scenario: { mode: 'prune', sparsity: 0.9, fineTune: true },
    claim: (r) => r.accuracy >= r.teacherAccuracy - 0.02 && r.baseline < 0.9,
  },
  {
    id: 'too-far',
    title: 'Pruning too far',
    text: 'At 99% sparsity only 1 weight in 100 survives: whole layers are cut and not even fine-tuning can rebuild the network.',
    scenario: { mode: 'prune', sparsity: 0.99, fineTune: true },
    claim: (r) => r.accuracy < 0.4,
  },
  {
    id: 'distill',
    title: 'A small student with a big teacher',
    text: "A student seven times smaller sees only 20 labelled digits: alone it gets 63%. If instead it imitates the teacher's softened outputs on a large pool of unlabelled digits, it reaches 99%, as good as the teacher.",
    scenario: { mode: 'distill', labelled: 20 },
    claim: (r) => r.accuracy >= r.teacherAccuracy - 0.01 && r.baseline < 0.7,
  },
  {
    id: 'plenty-labels',
    title: 'With enough labels the teacher adds little',
    text: 'Given 150 labelled digits the student alone gets 97%, so the extra value of the teacher shrinks to a couple of points: distillation matters most when labels are scarce.',
    scenario: { mode: 'distill', labelled: 150 },
    claim: (r) => r.baseline > 0.95 && r.accuracy - r.baseline < 0.03,
  },
];
