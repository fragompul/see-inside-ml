import { Matrix, Optimizer, createRng, type Parameter, type Rng } from '@sim/core';
import { DenseLayer } from '../dense';

/**
 * A normalising flow on two-dimensional data (RealNVP): an invertible map f from data to a standard Gaussian,
 * built from coupling layers. Each layer leaves one coordinate alone and rescales and shifts the other by an
 * amount that a small network computes from the first:
 *     y_c = x_c,      y_o = x_o * exp(s(x_c)) + t(x_c)
 * Its Jacobian is triangular, so log |det| = s(x_c) is free, and the layer is inverted exactly. The layers
 * alternate which coordinate is left alone. The density of a point is that of the Gaussian at f(x) times the
 * product of the determinants, so the model gives an exact log-likelihood, and sampling is f inverse of noise.
 */
const LOG_2PI = Math.log(2 * Math.PI);

export interface FlowConfig {
  layers: number;
  hidden: number;
  /** The scale s is limited to |s| <= scale (tanh), which keeps training stable. */
  scale: number;
}

export const DEFAULT_FLOW: FlowConfig = { layers: 4, hidden: 24, scale: 2 };

class Coupling {
  readonly net1: DenseLayer;
  readonly net2: DenseLayer;
  /** Index of the coordinate that is left alone. */
  readonly keep: number;
  private cache: { a: Matrix; x: Matrix } | null = null;

  constructor(
    keep: number,
    hidden: number,
    readonly scale: number,
    rng: Rng,
  ) {
    this.keep = keep;
    this.net1 = new DenseLayer(1, hidden, 'tanh', rng);
    this.net2 = new DenseLayer(hidden, 2, 'linear', rng);
    // start as the identity: scale and shift outputs near zero
    this.net2.weights.data.forEach((_, i) => {
      this.net2.weights.data[i] = (this.net2.weights.data[i] as number) * 0.05;
    });
  }

  get parameters(): Parameter[] {
    return [...this.net1.parameters, ...this.net2.parameters];
  }

  private params(x: Matrix, cache: boolean): { s: Float64Array; t: Float64Array; a: Matrix } {
    const c = new Matrix(x.rows, 1);
    for (let r = 0; r < x.rows; r++) c.data[r] = x.data[r * 2 + this.keep] as number;
    const out = this.net2.forward(this.net1.forward(c, cache), cache);
    const s = new Float64Array(x.rows);
    const t = new Float64Array(x.rows);
    for (let r = 0; r < x.rows; r++) {
      s[r] = this.scale * Math.tanh(out.data[r * 2] as number);
      t[r] = out.data[r * 2 + 1] as number;
    }
    return { s, t, a: out };
  }

  forward(x: Matrix, cache: boolean): { y: Matrix; logdet: Float64Array } {
    const { s, t, a } = this.params(x, cache);
    const y = new Matrix(x.rows, 2);
    const other = 1 - this.keep;
    for (let r = 0; r < x.rows; r++) {
      y.data[r * 2 + this.keep] = x.data[r * 2 + this.keep] as number;
      y.data[r * 2 + other] =
        (x.data[r * 2 + other] as number) * Math.exp(s[r] as number) + (t[r] as number);
    }
    if (cache) this.cache = { a, x };
    return { y, logdet: s };
  }

  inverse(y: Matrix): Matrix {
    const { s, t } = this.params(y, false);
    const x = new Matrix(y.rows, 2);
    const other = 1 - this.keep;
    for (let r = 0; r < y.rows; r++) {
      x.data[r * 2 + this.keep] = y.data[r * 2 + this.keep] as number;
      x.data[r * 2 + other] =
        ((y.data[r * 2 + other] as number) - (t[r] as number)) * Math.exp(-(s[r] as number));
    }
    return x;
  }

  /**
   * Backward for the per-row loss  l = ... - logdet: given dl/dy per row, returns dl/dx and fills the gradients.
   */
  backward(gy: Matrix): Matrix {
    const c = this.cache;
    if (!c) throw new Error('backward() called before forward()');
    const other = 1 - this.keep;
    const gx = new Matrix(gy.rows, 2);
    const dOut = new Matrix(gy.rows, 2);
    for (let r = 0; r < gy.rows; r++) {
      const raw = c.a.data[r * 2] as number;
      const th = Math.tanh(raw);
      const s = this.scale * th;
      const es = Math.exp(s);
      const go = gy.data[r * 2 + other] as number;
      gx.data[r * 2 + other] = go * es;
      const ds = go * (c.x.data[r * 2 + other] as number) * es - 1; // the -1 is from -logdet
      dOut.data[r * 2] = ds * this.scale * (1 - th * th);
      dOut.data[r * 2 + 1] = go;
    }
    const dc = this.net1.backward(this.net2.backward(dOut));
    for (let r = 0; r < gy.rows; r++)
      gx.data[r * 2 + this.keep] = (gy.data[r * 2 + this.keep] as number) + (dc.data[r] as number);
    return gx;
  }
}

export class Flow {
  readonly couplings: Coupling[];

  constructor(
    readonly config: FlowConfig,
    rng: Rng,
  ) {
    this.couplings = Array.from(
      { length: config.layers },
      (_, k) => new Coupling(k % 2, config.hidden, config.scale, rng),
    );
  }

  get parameters(): Parameter[] {
    return this.couplings.flatMap((c) => c.parameters);
  }

  /** Data to latent space, with the log-determinant of the whole map for every row. */
  forward(x: Matrix, cache = true): { z: Matrix; logdet: Float64Array; steps: Matrix[] } {
    let cur = x;
    const logdet = new Float64Array(x.rows);
    const steps: Matrix[] = [x];
    for (const layer of this.couplings) {
      const out = layer.forward(cur, cache);
      for (let r = 0; r < x.rows; r++)
        logdet[r] = (logdet[r] as number) + (out.logdet[r] as number);
      cur = out.y;
      steps.push(cur);
    }
    return { z: cur, logdet, steps };
  }

  inverse(z: Matrix): Matrix {
    let cur = z;
    for (let k = this.couplings.length - 1; k >= 0; k--)
      cur = (this.couplings[k] as Coupling).inverse(cur);
    return cur;
  }

  /** Log density of each row of x under the model. */
  logProb(x: Matrix): Float64Array {
    const { z, logdet } = this.forward(x, false);
    return Float64Array.from(
      { length: x.rows },
      (_, r) =>
        -0.5 * ((z.data[2 * r] as number) ** 2 + (z.data[2 * r + 1] as number) ** 2) -
        LOG_2PI +
        (logdet[r] as number),
    );
  }

  /** Average negative log-likelihood (nats per point) and its gradient into the parameters. */
  loss(x: Matrix, backprop: boolean): number {
    const { z, logdet } = this.forward(x, backprop);
    let total = 0;
    for (let r = 0; r < x.rows; r++)
      total +=
        0.5 * ((z.data[2 * r] as number) ** 2 + (z.data[2 * r + 1] as number) ** 2) +
        LOG_2PI -
        (logdet[r] as number);
    if (backprop) {
      let g = z;
      for (let k = this.couplings.length - 1; k >= 0; k--)
        g = (this.couplings[k] as Coupling).backward(g);
    }
    return total / x.rows;
  }

  sample(n: number, rng: Rng): Matrix {
    return this.inverse(
      new Matrix(
        n,
        2,
        Float64Array.from({ length: 2 * n }, () => rng.normal()),
      ),
    );
  }
}

export interface FlowTraining {
  steps: number;
  batch: number;
  learningRate: number;
  seed: number;
}

export const DEFAULT_FLOW_TRAINING: FlowTraining = {
  steps: 600,
  batch: 128,
  learningRate: 0.003,
  seed: 1,
};

export interface FlowRun {
  flow: Flow;
  /** Average negative log-likelihood on the training sample, sampled during training. */
  history: { step: number; nll: number }[];
}

/** Trains a flow by maximum likelihood on the rows of `data` (n x 2). */
export function trainFlow(
  data: Matrix,
  config: FlowConfig,
  options: FlowTraining,
  every = 50,
): FlowRun {
  const rng = createRng(options.seed);
  const flow = new Flow(config, createRng(options.seed + 1));
  const optimizer = new Optimizer('adam');
  const history: { step: number; nll: number }[] = [];
  for (let step = 1; step <= options.steps; step++) {
    const batch = new Matrix(options.batch, 2);
    for (let i = 0; i < options.batch; i++) {
      const pick = Math.floor(rng.next() * data.rows);
      batch.data.set(data.data.subarray(pick * 2, pick * 2 + 2), i * 2);
    }
    flow.loss(batch, true);
    optimizer.step(flow.parameters, options.learningRate);
    if (step % every === 0 || step === options.steps)
      history.push({ step, nll: flow.loss(data, false) });
  }
  return { flow, history };
}
