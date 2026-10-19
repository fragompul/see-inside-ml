import { Matrix, activate, derivative } from '@sim/core';
import {
  DEFAULT_DESCENT,
  Descent,
  bestLine,
  neuronData,
  type DescentConfig,
  type NeuronDataset,
} from '@sim/deep';

/**
 * One-click experiments. `claim.check` runs the setup from scratch and verifies the statement the
 * text makes; `lessons.test.ts` runs them all.
 */
export interface Lesson {
  id: string;
  title: string;
  /** How the descent is configured (the ball is released from its start). */
  descent?: Partial<DescentConfig>;
  /** The activation to show in section 02. */
  activation?: 'sigmoid' | 'relu';
  scrollTo: string;
  text: string;
  /** Whether the ball starts rolling by itself. */
  run: boolean;
  claim: { check: () => boolean };
}

function stepsToReach(config: Partial<DescentConfig>, target: number, max = 400): number {
  const d = new Descent({ ...DEFAULT_DESCENT, ...config });
  for (let i = 1; i <= max; i++) if (d.step().loss < target) return i;
  return Infinity;
}

const slope = (name: 'sigmoid' | 'relu', zs: number[]): number[] => {
  const z = new Matrix(1, zs.length, Float64Array.from(zs));
  return Array.from(derivative(name, z, activate(name, z)).data);
};

export const LESSONS: readonly Lesson[] = [
  {
    id: 'crawl',
    title: 'Steps that are too small',
    descent: { dataset: 'blobs', optimizer: 'sgd', lr: 0.01, bias: -3 },
    scrollTo: 'fnd-descent',
    run: true,
    text: 'With a learning rate of 0.01 the ball does roll downhill, but each step is so short that after hundreds of steps it has barely left the corner. Learning that is too slow is a failure too.',
    claim: { check: () => stepsToReach({ lr: 0.01 }, 0.3) === Infinity },
  },
  {
    id: 'quick',
    title: 'A bolder step',
    descent: { dataset: 'blobs', optimizer: 'sgd', lr: 0.5, bias: -3 },
    scrollTo: 'fnd-descent',
    run: true,
    text: 'Five times the learning rate and the ball reaches the valley floor in a few dozen steps. The slope tells it which way to go; the learning rate says how far.',
    claim: { check: () => stepsToReach({ lr: 0.5 }, 0.3) < 40 },
  },
  {
    id: 'momentum',
    title: 'Momentum',
    descent: { dataset: 'blobs', optimizer: 'momentum', lr: 0.1, bias: -3 },
    scrollTo: 'fnd-descent',
    run: true,
    text: 'Same learning rate as plain gradient descent, but the ball now remembers its velocity and keeps rolling through the flat stretch. It gets to the valley several times faster.',
    claim: {
      check: () =>
        stepsToReach({ optimizer: 'momentum', lr: 0.1 }, 0.3) < 30 &&
        stepsToReach({ optimizer: 'sgd', lr: 0.1 }, 0.3) > 80,
    },
  },
  {
    id: 'xor',
    title: 'No line solves XOR',
    descent: { dataset: 'xor', optimizer: 'sgd', lr: 0.5, bias: 0 },
    scrollTo: 'fnd-neuron',
    run: true,
    text: 'XOR puts the 1s on opposite corners. However you tilt or shift a line, at least one point is on the wrong side (75% is the best possible), and the landscape has no valley: the ball settles on a flat plateau at ln 2. To solve it you need more than one neuron.',
    claim: {
      check: () => {
        const best = bestLine(neuronData('xor' as NeuronDataset)).accuracy;
        const d = new Descent({ ...DEFAULT_DESCENT, dataset: 'xor', lr: 0.5, bias: 0 });
        for (let i = 0; i < 400; i++) d.step();
        return best === 0.75 && d.here().loss > Math.LN2 - 0.01 && d.gradientNorm() < 0.01;
      },
    },
  },
  {
    id: 'sigmoid',
    title: 'Why sigmoid fades',
    activation: 'sigmoid',
    scrollTo: 'fnd-activations',
    run: false,
    text: "The sigmoid's slope never exceeds 0.25 (and is almost 0 at both ends). Backpropagation multiplies one slope per layer, so ten sigmoid layers shrink the learning signal by at least 4¹⁰ ≈ a million. ReLU's slope is exactly 1 for positive inputs.",
    claim: {
      check: () => {
        const zs = Array.from({ length: 401 }, (_, i) => -10 + i * 0.05);
        const sig = slope('sigmoid', zs);
        const relu = slope('relu', [0.1, 1, 5, 40]);
        return Math.max(...sig) <= 0.25 + 1e-12 && 0.25 ** 10 < 1e-6 && relu.every((s) => s === 1);
      },
    },
  },
];
