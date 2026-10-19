import { createRng } from '@sim/core';

/**
 * A single neuron with two inputs, and the tiny datasets it can (or cannot) learn.
 * Everything here is small enough to be inspected by hand.
 */
export const NEURON_ACTIVATIONS = ['step', 'sigmoid', 'tanh', 'relu'] as const;
export type NeuronActivation = (typeof NEURON_ACTIVATIONS)[number];

export const neuronActivationInfo: Record<NeuronActivation, { label: string; note: string }> = {
  step: {
    label: 'Step',
    note: '1 if the weighted sum is positive, else 0: the original perceptron.',
  },
  sigmoid: {
    label: 'Sigmoid',
    note: 'A smooth step between 0 and 1: it can be trained by gradient descent.',
  },
  tanh: { label: 'Tanh', note: 'Like the sigmoid but centred on 0, between −1 and 1.' },
  relu: { label: 'ReLU', note: 'Zero for negative sums, the sum itself for positive ones.' },
};

export const neuronActivate = (kind: NeuronActivation, z: number): number => {
  switch (kind) {
    case 'step':
      return z > 0 ? 1 : 0;
    case 'sigmoid':
      return 1 / (1 + Math.exp(-z));
    case 'tanh':
      return Math.tanh(z);
    case 'relu':
      return Math.max(0, z);
  }
};

export interface NeuronParams {
  w1: number;
  w2: number;
  b: number;
}

/** The weighted sum `w₁x₁ + w₂x₂ + b` (before the activation). */
export const weightedSum = ({ w1, w2, b }: NeuronParams, x1: number, x2: number): number =>
  w1 * x1 + w2 * x2 + b;

export const NEURON_DATASETS = ['and', 'or', 'xor', 'blobs'] as const;
export type NeuronDataset = (typeof NEURON_DATASETS)[number];

export const neuronDatasetInfo: Record<NeuronDataset, { label: string; note: string }> = {
  and: { label: 'AND', note: 'Class 1 only when both inputs are 1.' },
  or: { label: 'OR', note: 'Class 1 when at least one input is 1.' },
  xor: {
    label: 'XOR',
    note: 'Class 1 when exactly one input is 1: no single straight line can separate it.',
  },
  blobs: { label: 'Two clouds', note: 'Two clusters of points: easy for one neuron.' },
};

export interface Points {
  /** `n × 2` inputs, row-major. */
  x: Float64Array;
  /** 0 or 1 for each point. */
  y: Uint8Array;
  count: number;
}

const LOGIC: Record<'and' | 'or' | 'xor', [number, number, number][]> = {
  and: [
    [0, 0, 0],
    [0, 1, 0],
    [1, 0, 0],
    [1, 1, 1],
  ],
  or: [
    [0, 0, 0],
    [0, 1, 1],
    [1, 0, 1],
    [1, 1, 1],
  ],
  xor: [
    [0, 0, 0],
    [0, 1, 1],
    [1, 0, 1],
    [1, 1, 0],
  ],
};

/** The points of a dataset (fixed, so every run and test sees the same ones). */
export function neuronData(dataset: NeuronDataset): Points {
  if (dataset !== 'blobs') {
    const rows = LOGIC[dataset];
    return {
      x: Float64Array.from(rows.flatMap(([a, b]) => [a, b])),
      y: Uint8Array.from(rows.map((r) => r[2])),
      count: rows.length,
    };
  }
  const rng = createRng(21);
  const x: number[] = [];
  const y: number[] = [];
  for (let i = 0; i < 24; i++) {
    const cls = i % 2;
    const [cx, cy] = cls === 0 ? [0.2, 0.25] : [0.8, 0.75];
    x.push(cx + rng.normal() * 0.12, cy + rng.normal() * 0.12);
    y.push(cls);
  }
  return { x: Float64Array.from(x), y: Uint8Array.from(y), count: 24 };
}

/** Fraction of points on the right side of the line `w·x + b = 0` (class 1 when positive). */
export function neuronAccuracy(data: Points, p: NeuronParams): number {
  let right = 0;
  for (let i = 0; i < data.count; i++) {
    const z = weightedSum(p, data.x[2 * i] as number, data.x[2 * i + 1] as number);
    if (z > 0 === (data.y[i] === 1)) right += 1;
  }
  return right / data.count;
}

/**
 * The best a single line can do on a dataset, found by trying 360 directions and, for each, every
 * place the line can cut the points. Returns the accuracy and one set of weights that reaches it.
 */
export function bestLine(data: Points): { accuracy: number; params: NeuronParams } {
  let best: { accuracy: number; params: NeuronParams } = {
    accuracy: -1,
    params: { w1: 0, w2: 0, b: 0 },
  };
  const scale = 4;
  for (let k = 0; k < 360; k++) {
    const angle = (k / 360) * 2 * Math.PI;
    const w1 = Math.cos(angle) * scale;
    const w2 = Math.sin(angle) * scale;
    const projections = Array.from(
      { length: data.count },
      (_, i) => w1 * (data.x[2 * i] as number) + w2 * (data.x[2 * i + 1] as number),
    );
    const sorted = [...projections].sort((a, b) => a - b);
    // thresholds half way between consecutive projections (and beyond both ends)
    const cuts = [
      sorted[0]! - 1,
      ...sorted.slice(1).map((v, i) => (v + sorted[i]!) / 2),
      sorted[sorted.length - 1]! + 1,
    ];
    for (const cut of cuts) {
      const params = { w1, w2, b: -cut };
      const acc = neuronAccuracy(data, params);
      if (acc > best.accuracy + 1e-12) best = { accuracy: acc, params };
    }
  }
  return best;
}

/** Binary cross-entropy of a sigmoid neuron on a dataset, and its gradient with respect to w₁, w₂, b. */
export function neuronLoss(data: Points, p: NeuronParams): { loss: number; grad: NeuronParams } {
  let loss = 0;
  let g1 = 0;
  let g2 = 0;
  let gb = 0;
  for (let i = 0; i < data.count; i++) {
    const x1 = data.x[2 * i] as number;
    const x2 = data.x[2 * i + 1] as number;
    const a = neuronActivate('sigmoid', weightedSum(p, x1, x2));
    const y = data.y[i] as number;
    loss -= y * Math.log(Math.max(1e-12, a)) + (1 - y) * Math.log(Math.max(1e-12, 1 - a));
    const d = a - y; // dL/dz for sigmoid + cross-entropy
    g1 += d * x1;
    g2 += d * x2;
    gb += d;
  }
  const n = data.count;
  return { loss: loss / n, grad: { w1: g1 / n, w2: g2 / n, b: gb / n } };
}

/** Loss over an `n × n` grid of (w₁, w₂) in `[-range, range]²` with the bias held fixed, row by row from the top (largest w₂ first). */
export function lossLandscape(data: Points, b: number, range: number, n: number): Float64Array {
  const out = new Float64Array(n * n);
  for (let row = 0; row < n; row++) {
    for (let col = 0; col < n; col++) {
      const w1 = ((col + 0.5) / n) * 2 * range - range;
      const w2 = range - ((row + 0.5) / n) * 2 * range;
      out[row * n + col] = neuronLoss(data, { w1, w2, b }).loss;
    }
  }
  return out;
}
