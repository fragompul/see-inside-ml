/** A tensor of learnable numbers plus its gradient buffer. */
export class Parameter {
  readonly grad: Float64Array;

  constructor(
    /** Shared with the owning layer's matrix, so updates are visible everywhere. */
    readonly data: Float64Array,
    /** Whether L1/L2 regularisation applies (weights yes, biases no). */
    readonly decay: boolean,
  ) {
    this.grad = new Float64Array(data.length);
  }
}

export interface Regularization {
  l1: number;
  l2: number;
}

export const NO_REGULARIZATION: Regularization = { l1: 0, l2: 0 };

export const OPTIMIZER_NAMES = ['sgd', 'momentum', 'rmsprop', 'adam'] as const;
export type OptimizerName = (typeof OPTIMIZER_NAMES)[number];

export interface OptimizerOptions {
  /** Velocity (momentum) or decay rate (RMSprop). */
  momentum?: number;
  beta1?: number;
  beta2?: number;
  epsilon?: number;
}

interface State {
  m: Float64Array;
  v: Float64Array;
}

/**
 * One optimiser step over all parameters. Regularisation gradients are added to the loss gradient
 * (L1: λ·sign(w), L2: λ·w) before the update rule, matching the reference implementation.
 */
export class Optimizer {
  /** Number of completed steps (drives Adam's bias correction). */
  steps = 0;
  private readonly state = new Map<Parameter, State>();
  private readonly momentum: number;
  private readonly beta1: number;
  private readonly beta2: number;
  private readonly epsilon: number;

  constructor(
    readonly name: OptimizerName,
    options: OptimizerOptions = {},
  ) {
    this.momentum = options.momentum ?? 0.9;
    this.beta1 = options.beta1 ?? 0.9;
    this.beta2 = options.beta2 ?? 0.999;
    this.epsilon = options.epsilon ?? 1e-8;
  }

  reset(): void {
    this.steps = 0;
    this.state.clear();
  }

  step(params: readonly Parameter[], lr: number, reg: Regularization = NO_REGULARIZATION): void {
    this.steps += 1;
    const t = this.steps;
    for (const p of params) {
      let s = this.state.get(p);
      if (!s) {
        s = { m: new Float64Array(p.data.length), v: new Float64Array(p.data.length) };
        this.state.set(p, s);
      }
      const { data, grad } = p;
      for (let i = 0; i < data.length; i++) {
        const w = data[i] as number;
        let g = grad[i] as number;
        if (p.decay) {
          if (reg.l1 > 0) g += reg.l1 * Math.sign(w);
          if (reg.l2 > 0) g += reg.l2 * w;
        }
        switch (this.name) {
          case 'sgd':
            data[i] = w - lr * g;
            break;
          case 'momentum': {
            const v = this.momentum * (s.v[i] as number) + g;
            s.v[i] = v;
            data[i] = w - lr * v;
            break;
          }
          case 'rmsprop': {
            const v = this.momentum * (s.v[i] as number) + (1 - this.momentum) * g * g;
            s.v[i] = v;
            data[i] = w - (lr * g) / (Math.sqrt(v) + this.epsilon);
            break;
          }
          case 'adam': {
            const m = this.beta1 * (s.m[i] as number) + (1 - this.beta1) * g;
            const v = this.beta2 * (s.v[i] as number) + (1 - this.beta2) * g * g;
            s.m[i] = m;
            s.v[i] = v;
            const mHat = m / (1 - this.beta1 ** t);
            const vHat = v / (1 - this.beta2 ** t);
            data[i] = w - (lr * mHat) / (Math.sqrt(vHat) + this.epsilon);
            break;
          }
        }
      }
    }
  }
}

export const optimizerInfo: Record<OptimizerName, { label: string; note: string }> = {
  sgd: { label: 'SGD', note: 'Plain gradient descent: step against the gradient.' },
  momentum: {
    label: 'Momentum',
    note: 'Accumulates a velocity so it keeps rolling through flat or noisy regions.',
  },
  rmsprop: {
    label: 'RMSprop',
    note: 'Scales each weight’s step by a running average of its recent gradient size.',
  },
  adam: {
    label: 'Adam',
    note: 'Momentum plus RMSprop-style scaling, with bias correction. A strong default.',
  },
};
