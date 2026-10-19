import { Optimizer, Parameter, OPTIMIZER_NAMES, type OptimizerName } from '@sim/core';
import {
  NEURON_DATASETS,
  neuronData,
  neuronLoss,
  type NeuronDataset,
  type NeuronParams,
  type Points,
} from './neuron';

/**
 * Gradient descent on the loss landscape of one sigmoid neuron: the two weights are the position of
 * a ball, the bias is fixed so the landscape is a 2-D surface, and an optimiser rolls the ball
 * downhill. The same optimisers the rest of the app uses (SGD, momentum, RMSprop, Adam).
 */
export interface DescentConfig {
  dataset: NeuronDataset;
  optimizer: OptimizerName;
  lr: number;
  /** The bias, held fixed. */
  bias: number;
}

export const DEFAULT_DESCENT: DescentConfig = {
  dataset: 'blobs',
  optimizer: 'sgd',
  lr: 0.1,
  bias: -3,
};

export const DESCENT_LIMITS = { range: 8, lr: [0.001, 20], bias: [-4, 4] } as const;

export interface DescentPoint {
  w1: number;
  w2: number;
  loss: number;
}

export function validateDescent(c: DescentConfig): void {
  if (!(OPTIMIZER_NAMES as readonly string[]).includes(c.optimizer))
    throw new RangeError('Unknown optimizer');
  if (!(c.lr >= DESCENT_LIMITS.lr[0] && c.lr <= DESCENT_LIMITS.lr[1]))
    throw new RangeError(
      `Learning rate must be in [${DESCENT_LIMITS.lr[0]}, ${DESCENT_LIMITS.lr[1]}]`,
    );
  if (!(c.bias >= DESCENT_LIMITS.bias[0] && c.bias <= DESCENT_LIMITS.bias[1]))
    throw new RangeError(`Bias must be in [${DESCENT_LIMITS.bias[0]}, ${DESCENT_LIMITS.bias[1]}]`);
  if (!(NEURON_DATASETS as readonly string[]).includes(c.dataset))
    throw new RangeError('Unknown dataset');
}

export class Descent {
  private cfg: DescentConfig;
  private data: Points;
  private weights = new Parameter(new Float64Array(2), true);
  private optimizer: Optimizer;
  /** Where the ball has been, starting from its initial position. */
  path: DescentPoint[] = [];

  constructor(config: DescentConfig, start: { w1: number; w2: number } = { w1: -5, w2: 5 }) {
    validateDescent(config);
    this.cfg = { ...config };
    this.data = neuronData(config.dataset);
    this.optimizer = new Optimizer(config.optimizer);
    this.place(start.w1, start.w2);
  }

  get config(): Readonly<DescentConfig> {
    return this.cfg;
  }

  get position(): NeuronParams {
    return {
      w1: this.weights.data[0] as number,
      w2: this.weights.data[1] as number,
      b: this.cfg.bias,
    };
  }

  /** Puts the ball somewhere (clamped to the plot window) and forgets the path and any momentum. */
  place(w1: number, w2: number): void {
    const r = DESCENT_LIMITS.range;
    this.weights.data[0] = Math.max(-r, Math.min(r, w1));
    this.weights.data[1] = Math.max(-r, Math.min(r, w2));
    this.optimizer.reset();
    this.path = [this.here()];
  }

  /** Changes optimiser, learning rate, bias or dataset. A dataset or bias change moves the surface under the ball. */
  update(patch: Partial<DescentConfig>): void {
    const next = { ...this.cfg, ...patch };
    validateDescent(next);
    if (next.dataset !== this.cfg.dataset) this.data = neuronData(next.dataset);
    if (next.optimizer !== this.cfg.optimizer) this.optimizer = new Optimizer(next.optimizer);
    const surfaceChanged = next.dataset !== this.cfg.dataset || next.bias !== this.cfg.bias;
    this.cfg = next;
    if (surfaceChanged || patch.optimizer !== undefined) {
      this.optimizer.reset();
      this.path = [this.here()];
    }
  }

  /** One optimiser step; returns the new position. */
  step(): DescentPoint {
    const { loss, grad } = neuronLoss(this.data, this.position);
    void loss;
    this.weights.grad[0] = grad.w1;
    this.weights.grad[1] = grad.w2;
    this.optimizer.step([this.weights], this.cfg.lr);
    const r = DESCENT_LIMITS.range * 4; // far outside the window still counts as "ran away"
    for (let i = 0; i < 2; i++)
      this.weights.data[i] = Math.max(-r, Math.min(r, this.weights.data[i] as number));
    const point = this.here();
    this.path.push(point);
    return point;
  }

  /** Current position with its loss. */
  here(): DescentPoint {
    const p = this.position;
    return { w1: p.w1, w2: p.w2, loss: neuronLoss(this.data, p).loss };
  }

  /** Size of the gradient at the current position. */
  gradientNorm(): number {
    const { grad } = neuronLoss(this.data, this.position);
    return Math.hypot(grad.w1, grad.w2);
  }
}
