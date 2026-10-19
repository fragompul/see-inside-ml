import { DEFAULT_CONFIG, type HiddenLayerConfig, type PlaygroundConfig } from '@sim/deep';

export interface Preset {
  id: string;
  title: string;
  /** The lesson: what to look for. Shown while the preset is active. */
  lesson: string;
  config: Partial<PlaygroundConfig>;
  /** Suggested speed (epochs per frame). */
  speed?: number;
}

const layers = (
  activation: HiddenLayerConfig['activation'],
  ...neurons: number[]
): HiddenLayerConfig[] => neurons.map((n) => ({ neurons: n, activation }));

/** Guided experiments. Each one is a complete configuration that teaches one idea. */
export const PRESETS: readonly Preset[] = [
  {
    id: 'xor-linear',
    title: 'Linear can’t do XOR',
    lesson:
      'No hidden layers means a straight-line boundary. XOR needs opposite corners to match, so accuracy stalls near 50% no matter how long you train.',
    config: { dataset: 'xor', noise: 0.2, hidden: [], features: ['X1', 'X2'] },
  },
  {
    id: 'xor-feature',
    title: 'Feature engineering',
    lesson:
      'Same linear model, one new input: x₁·x₂. In that space XOR is trivially separable and accuracy jumps to ~100%.',
    config: { dataset: 'xor', noise: 0.2, hidden: [], features: ['X1', 'X2', 'X1*X2'] },
  },
  {
    id: 'xor-hidden',
    title: 'Let the net find it',
    lesson:
      'Instead of handing over x₁·x₂, add a small hidden layer. Hover the hidden neurons: each learns a half-plane, and the output combines them.',
    config: { dataset: 'xor', noise: 0.2, hidden: layers('tanh', 4, 3), features: ['X1', 'X2'] },
  },
  {
    id: 'spiral',
    title: 'Spiral challenge',
    lesson:
      'A hard, twisting boundary. Sine features give the network smooth building blocks; give it a few hundred epochs (try speed 25).',
    config: {
      dataset: 'spiral',
      noise: 0.1,
      samples: 400,
      hidden: layers('tanh', 8, 8, 6),
      features: ['X1', 'X2', 'sin(X1)', 'sin(X2)'],
    },
    speed: 25,
  },
  {
    id: 'overfit',
    title: 'Overfitting',
    lesson:
      'Few noisy samples and a big network. Watch the curves: training loss falls to zero while test loss turns around and climbs. The model memorises noise.',
    config: {
      dataset: 'gaussian',
      classes: 2,
      noise: 0.8,
      samples: 60,
      trainRatio: 50,
      hidden: layers('tanh', 16, 16, 16),
      features: ['X1', 'X2'],
    },
    speed: 10,
  },
  {
    id: 'regularize',
    title: '…and L2 to the rescue',
    lesson:
      'Identical setup plus L2 regularization. Weights are kept small, the boundary smooths out, and the gap between train and test loss shrinks.',
    config: {
      dataset: 'gaussian',
      classes: 2,
      noise: 0.8,
      samples: 60,
      trainRatio: 50,
      hidden: layers('tanh', 16, 16, 16),
      features: ['X1', 'X2'],
      regularization: 'l2',
      regularizationRate: 0.03,
    },
    speed: 10,
  },
  {
    id: 'vanishing',
    title: 'Vanishing gradients',
    lesson:
      'Five sigmoid layers trained with plain SGD barely move: gradients shrink at every layer. Compare the width of the edges near the input.',
    config: {
      dataset: 'circle',
      noise: 0.1,
      hidden: layers('sigmoid', 6, 6, 6, 6, 6),
      optimizer: 'sgd',
      lr: 0.1,
      features: ['X1', 'X2'],
    },
    speed: 10,
  },
  {
    id: 'relu-fix',
    title: 'Same net, ReLU',
    lesson:
      'Swap sigmoid for ReLU and nothing else changes, yet it learns in a few hundred epochs. Activation choice matters a lot in deep networks.',
    config: {
      dataset: 'circle',
      noise: 0.1,
      hidden: layers('relu', 6, 6, 6, 6, 6),
      optimizer: 'sgd',
      lr: 0.1,
      features: ['X1', 'X2'],
    },
    speed: 10,
  },
  {
    id: 'waves',
    title: 'Regression: waves',
    lesson:
      'Same machinery, continuous output: the surface colour is the predicted value (frost = negative, ember = positive). Hover to read ŷ.',
    config: {
      dataset: 'saddle',
      noise: 0.1,
      hidden: layers('tanh', 8, 8),
      features: ['X1', 'X2'],
      lr: 0.01,
      loss: 'mse',
    },
    speed: 10,
  },
  {
    id: 'multiclass',
    title: 'Three classes',
    lesson:
      'With more than two classes the output layer becomes a softmax over classes. Hover the plane to see the probabilities add up to 100%.',
    config: {
      dataset: 'gaussian',
      classes: 3,
      noise: 0.3,
      hidden: layers('tanh', 4),
      features: ['X1', 'X2'],
    },
  },
];

/** Every preset is complete, so applying one never inherits leftovers from the previous setup. */
export const presetConfig = (preset: Preset): PlaygroundConfig => ({
  ...DEFAULT_CONFIG,
  ...preset.config,
});
