import { DenseLayer } from '../dense';
import { Matrix, type Parameter, type Rng } from '@sim/core';
import { TIME_FEATURES, noiseTo, timeFeatures, type NoiseSchedule } from './schedule';

/**
 * The denoiser ε_θ(x_t, t): a small MLP that reads a noisy point and the time, and predicts the
 * noise that was added to it.
 *
 *   [x, y, time features] → hidden → hidden → hidden → (ε_x, ε_y)
 *
 * Training minimises the mean squared error between its prediction and the true ε.
 */
export interface DenoiserArchitecture {
  hidden: number;
}

export const DENOISER_LIMITS = { hidden: [16, 128] } as const;

export function validateDenoiser(a: DenoiserArchitecture): void {
  const [lo, hi] = DENOISER_LIMITS.hidden;
  if (!Number.isInteger(a.hidden) || a.hidden < lo || a.hidden > hi)
    throw new RangeError(`Hidden size must be an integer in [${lo}, ${hi}]`);
}

const INPUTS = 2 + TIME_FEATURES;

export class Denoiser {
  readonly layers: DenseLayer[];

  constructor(
    readonly arch: DenoiserArchitecture,
    rng: Rng,
  ) {
    validateDenoiser(arch);
    this.layers = [
      new DenseLayer(INPUTS, arch.hidden, 'swish', rng),
      new DenseLayer(arch.hidden, arch.hidden, 'swish', rng),
      new DenseLayer(arch.hidden, arch.hidden, 'swish', rng),
      new DenseLayer(arch.hidden, 2, 'linear', rng),
    ];
  }

  get parameters(): Parameter[] {
    return this.layers.flatMap((l) => l.parameters);
  }

  get parameterCount(): number {
    return this.parameters.reduce((sum, p) => sum + p.data.length, 0);
  }

  /** Network input for points `x` (`n × 2`) at per-point times `t`. */
  private inputs(x: ArrayLike<number>, t: ArrayLike<number>, steps: number): Matrix {
    const n = x.length / 2;
    const m = new Matrix(n, INPUTS);
    for (let i = 0; i < n; i++) {
      m.data[i * INPUTS] = x[2 * i] as number;
      m.data[i * INPUTS + 1] = x[2 * i + 1] as number;
      m.data.set(timeFeatures(t[i] as number, steps), i * INPUTS + 2);
    }
    return m;
  }

  private run(input: Matrix, cache: boolean): Matrix {
    let a = input;
    for (const layer of this.layers) a = layer.forward(a, cache);
    return a;
  }

  /** Predicted noise for points `x` (`n × 2`) all at time `t`, without disturbing training. */
  predict(x: Float64Array, t: number, steps: number): Float64Array {
    const times = new Float64Array(x.length / 2).fill(t);
    return Float64Array.from(this.run(this.inputs(x, times, steps), false).data);
  }

  /**
   * Fills the gradients for one batch: clean points `x0`, a time per point and the noise `eps`
   * that corrupts them. Returns the mean squared error of the noise prediction.
   */
  gradients(
    schedule: NoiseSchedule,
    x0: Float64Array,
    t: ArrayLike<number>,
    eps: Float64Array,
  ): number {
    const n = x0.length / 2;
    const noisy = new Float64Array(x0.length);
    for (let i = 0; i < n; i++) {
      const point = noiseTo(
        schedule,
        x0.subarray(2 * i, 2 * i + 2),
        eps.subarray(2 * i, 2 * i + 2),
        t[i] as number,
      );
      noisy.set(point, 2 * i);
    }
    const out = this.run(this.inputs(noisy, t, schedule.steps), true);
    // loss per point = mean over the two coordinates of (pred − ε)², so dL/dpred = pred − ε
    const dOut = new Matrix(n, 2);
    let loss = 0;
    for (let i = 0; i < dOut.data.length; i++) {
      const diff = (out.data[i] as number) - (eps[i] as number);
      dOut.data[i] = diff;
      loss += diff * diff;
    }
    let d = dOut;
    for (let i = this.layers.length - 1; i >= 0; i--)
      d = (this.layers[i] as DenseLayer).backward(d);
    return loss / dOut.data.length;
  }
}
