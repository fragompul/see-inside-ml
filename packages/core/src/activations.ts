import { Matrix } from './matrix';

export const ACTIVATION_NAMES = [
  'relu',
  'leaky_relu',
  'tanh',
  'sigmoid',
  'elu',
  'swish',
  'linear',
  'softmax',
] as const;

export type ActivationName = (typeof ACTIVATION_NAMES)[number];

/** Hidden-layer choices (softmax is reserved for classification outputs). */
export const HIDDEN_ACTIVATIONS = ACTIVATION_NAMES.filter(
  (name): name is Exclude<ActivationName, 'softmax'> => name !== 'softmax',
);

export const LEAKY_SLOPE = 0.02;
const CLIP = 500;

const clip = (z: number) => Math.min(CLIP, Math.max(-CLIP, z));
const sigmoid = (z: number) => 1 / (1 + Math.exp(-clip(z)));

export function isActivationName(value: string): value is ActivationName {
  return (ACTIVATION_NAMES as readonly string[]).includes(value);
}

/** Initialisation family matching the activation: He for the ReLU family, Xavier otherwise. */
export function usesHeInit(name: ActivationName): boolean {
  return name === 'relu' || name === 'leaky_relu' || name === 'linear';
}

/** Applies the activation to pre-activations `z`. */
export function activate(name: ActivationName, z: Matrix): Matrix {
  switch (name) {
    case 'relu':
      return z.map((x) => Math.max(0, x));
    case 'leaky_relu':
      return z.map((x) => (x > 0 ? x : LEAKY_SLOPE * x));
    case 'tanh':
      return z.map(Math.tanh);
    case 'sigmoid':
      return z.map(sigmoid);
    case 'elu':
      return z.map((x) => (x > 0 ? x : Math.exp(clip(x)) - 1));
    case 'swish':
      return z.map((x) => x * sigmoid(x));
    case 'linear':
      return z.clone();
    case 'softmax': {
      const out = new Matrix(z.rows, z.cols);
      for (let i = 0; i < z.rows; i++) {
        let max = -Infinity;
        for (let j = 0; j < z.cols; j++) max = Math.max(max, z.get(i, j));
        let sum = 0;
        for (let j = 0; j < z.cols; j++) {
          const e = Math.exp(z.get(i, j) - max);
          out.set(i, j, e);
          sum += e;
        }
        for (let j = 0; j < z.cols; j++) out.set(i, j, out.get(i, j) / sum);
      }
      return out;
    }
  }
}

/**
 * dA/dZ element-wise, given pre-activations `z` and activations `a`.
 * Softmax returns ones: its Jacobian is folded into the loss gradient at the output layer
 * (dL/dZ = ŷ − y for cross-entropy), exactly as in the reference implementation.
 */
export function derivative(name: ActivationName, z: Matrix, a: Matrix): Matrix {
  switch (name) {
    case 'relu':
      return z.map((x) => (x > 0 ? 1 : 0));
    case 'leaky_relu':
      return z.map((x) => (x > 0 ? 1 : LEAKY_SLOPE));
    case 'tanh':
      return a.map((y) => 1 - y * y);
    case 'sigmoid':
      return a.map((y) => y * (1 - y));
    case 'elu': {
      const out = new Matrix(z.rows, z.cols);
      for (let i = 0; i < z.data.length; i++) {
        out.data[i] = (z.data[i] as number) > 0 ? 1 : (a.data[i] as number) + 1;
      }
      return out;
    }
    case 'swish': {
      const out = new Matrix(z.rows, z.cols);
      for (let i = 0; i < z.data.length; i++) {
        const s = sigmoid(z.data[i] as number);
        out.data[i] = s + (a.data[i] as number) * (1 - s);
      }
      return out;
    }
    case 'linear':
    case 'softmax':
      return z.map(() => 1);
  }
}

export interface ActivationInfo {
  label: string;
  /** KaTeX source, shown in tooltips and lessons. */
  formula: string;
  /** Short plain-language note for beginners. */
  note: string;
}

export const activationInfo: Record<ActivationName, ActivationInfo> = {
  relu: {
    label: 'ReLU',
    formula: String.raw`\max(0, z)`,
    note: 'Passes positives, blocks negatives. Fast and the default for deep nets.',
  },
  leaky_relu: {
    label: 'Leaky ReLU',
    formula: String.raw`\begin{cases} z & z>0 \\ 0.02\,z & z\le 0 \end{cases}`,
    note: 'Like ReLU, but negatives leak a little so neurons never fully die.',
  },
  tanh: {
    label: 'Tanh',
    formula: String.raw`\tanh(z)`,
    note: 'Squashes to (−1, 1), zero-centred. Smooth, but can saturate.',
  },
  sigmoid: {
    label: 'Sigmoid',
    formula: String.raw`\frac{1}{1+e^{-z}}`,
    note: 'Squashes to (0, 1). Classic, but gradients vanish in deep stacks.',
  },
  elu: {
    label: 'ELU',
    formula: String.raw`\begin{cases} z & z>0 \\ e^{z}-1 & z\le 0 \end{cases}`,
    note: 'Smooth negative side that pushes mean activations toward zero.',
  },
  swish: {
    label: 'Swish',
    formula: String.raw`z\,\sigma(z)`,
    note: 'Smooth, non-monotonic; often slightly better than ReLU in deep models.',
  },
  linear: {
    label: 'Linear',
    formula: String.raw`z`,
    note: 'No non-linearity: stacking linear layers is still just one linear map.',
  },
  softmax: {
    label: 'Softmax',
    formula: String.raw`\frac{e^{z_i}}{\sum_j e^{z_j}}`,
    note: 'Turns scores into class probabilities that sum to 1.',
  },
};
