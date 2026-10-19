import type { ParamSpec } from '@/shared/lab-kit';
import type { ToolboxRun, ToolboxScenario } from './toolboxCore';

export const toolboxParams: ParamSpec[] = [
  {
    key: 'dataset',
    label: 'Task',
    kind: 'segmented',
    options: [
      { value: 'moons', label: 'Moons' },
      { value: 'circles', label: 'Circles' },
      { value: 'spirals', label: 'Spirals' },
      { value: 'xor', label: 'XOR' },
    ],
  },
  {
    key: 'trainSize',
    label: 'Training points',
    kind: 'ladder',
    values: [40, 80, 200],
    format: (v) => String(v),
  },
  {
    key: 'noise',
    label: 'Noise in the data',
    kind: 'slider',
    min: 0,
    max: 0.8,
    step: 0.05,
    format: (v) => v.toFixed(2),
  },
  {
    key: 'hidden',
    label: 'Units per hidden layer',
    kind: 'ladder',
    values: [8, 16, 32, 64],
    format: (v) => String(v),
  },
  { key: 'depth', label: 'Hidden layers', kind: 'slider', min: 1, max: 6, step: 1 },
  {
    key: 'activation',
    label: 'Activation',
    kind: 'segmented',
    options: [
      { value: 'relu', label: 'ReLU' },
      { value: 'tanh', label: 'tanh' },
    ],
  },
  {
    key: 'init',
    label: 'Weight initialisation',
    kind: 'select',
    options: [
      { value: 'zeros', label: 'All zeros' },
      { value: 'small', label: 'Tiny random (0.01)' },
      { value: 'xavier', label: 'Xavier / Glorot' },
      { value: 'he', label: 'He' },
      { value: 'large', label: 'Large random (1.5)' },
    ],
  },
  {
    key: 'dropout',
    label: 'Dropout',
    kind: 'ladder',
    values: [0, 0.2, 0.5],
    format: (v) => String(v),
  },
  { key: 'batchNorm', label: 'Batch normalisation', kind: 'switch' },
  {
    key: 'weightDecay',
    label: 'Weight decay',
    kind: 'ladder',
    values: [0, 0.01, 0.05, 0.1],
    format: (v) => String(v),
  },
  {
    key: 'optimizer',
    label: 'Optimiser',
    kind: 'segmented',
    options: [
      { value: 'sgd', label: 'SGD' },
      { value: 'momentum', label: 'Momentum' },
      { value: 'adam', label: 'Adam' },
    ],
  },
  {
    key: 'learningRate',
    label: 'Learning rate',
    kind: 'ladder',
    values: [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3],
    format: (v) => String(v),
  },
  {
    key: 'epochs',
    label: 'Epochs',
    kind: 'ladder',
    values: [30, 60, 100, 200, 400],
    format: (v) => String(v),
  },
  {
    key: 'patience',
    label: 'Early stopping patience (epochs)',
    kind: 'ladder',
    values: [0, 10, 20, 40],
    format: (v) => (v === 0 ? 'off' : String(v)),
    hint: 'Stop when the loss on held-out data has not improved for this many epochs, and go back to the best weights.',
  },
];

export interface ToolboxExperiment {
  id: string;
  title: string;
  text: string;
  scenario: Partial<ToolboxScenario>;
  claim: (run: ToolboxRun) => boolean;
}

const hard = { noise: 0.5, trainSize: 40, hidden: 64, epochs: 200 } as const;

export const toolboxExperiments: ToolboxExperiment[] = [
  {
    id: 'zeros',
    title: 'Everything starts at zero',
    text: 'If every weight starts at zero, every hidden unit computes the same thing and receives the same gradient, so they stay identical for ever: the layer behaves like a single unit, and the network is stuck at a coin flip. Random initialisation exists to break this symmetry.',
    scenario: { init: 'zeros' },
    claim: (r) => r.testAccuracy < 0.55 && r.result.model.distinctFirstLayerUnits() <= 1,
  },
  {
    id: 'overfit',
    title: 'Training too long',
    text: 'A 64-unit network on 40 noisy points: given enough epochs it fits every training point (100%), but the loss on fresh data is more than ten times that of the best moment, and a 13-point accuracy gap opens between what it knows and what it can do.',
    scenario: { ...hard, epochs: 400 },
    claim: (r) => r.finalTrainAccuracy >= 0.99 && r.gap > 0.1 && r.testLoss > 0.9,
  },
  {
    id: 'early-stop',
    title: 'Early stopping',
    text: 'Watching the loss on held-out data and stopping when it has not improved for 20 epochs ends training after about 40 epochs (instead of 400), goes back to the best weights, and gives a much better test loss.',
    scenario: { ...hard, epochs: 400, patience: 20 },
    claim: (r) =>
      r.result.history.length < 100 && r.testLoss < 0.5 && r.gap < 0.05 && r.result.stoppedEarly,
  },
  {
    id: 'dropout',
    title: 'Dropout',
    text: 'Randomly silencing half of the hidden units at every step stops units from relying on one another: the network cannot memorise the training points as easily, and the test loss after 400 epochs is clearly below that of the unregularised network.',
    scenario: { ...hard, epochs: 400, dropout: 0.5 },
    claim: (r) => r.testLoss < 0.9 && r.gap < 0.08,
  },
  {
    id: 'decay',
    title: 'Weight decay',
    text: 'A small penalty on the size of the weights keeps the curve smooth: the same network ends with a test loss of about 0.35 instead of 0.83 and almost no gap.',
    scenario: { ...hard, weightDecay: 0.05 },
    claim: (r) => r.testLoss < 0.5 && r.gap < 0.05,
  },
  {
    id: 'big-rate',
    title: 'A learning rate that is too big',
    text: 'Plain gradient descent with a learning rate of 3 overshoots every step, the loss explodes and the network ends at chance.',
    scenario: { optimizer: 'sgd', learningRate: 3 },
    claim: (r) => r.testAccuracy < 0.6,
  },
  {
    id: 'batch-norm',
    title: 'Batch normalisation tolerates it',
    text: 'The same absurd learning rate, now with every layer re-centred and re-scaled on each mini-batch: the activations cannot blow up, and the network learns (about 93% on fresh data).',
    scenario: { optimizer: 'sgd', learningRate: 3, batchNorm: true },
    claim: (r) => r.testAccuracy > 0.9,
  },
  {
    id: 'finder',
    title: 'Finding the learning rate',
    text: 'Sweep the rate upwards from tiny to huge over 150 mini-batches and watch the loss: it falls, bottoms out, and then blows up. A tenth of the rate at the bottom is a safe choice, here between 0.05 and 1 for plain gradient descent.',
    scenario: { optimizer: 'sgd', learningRate: 0.3 },
    claim: (r) => r.finder.suggested > 0.05 && r.finder.suggested < 1 && r.testAccuracy > 0.95,
  },
];
