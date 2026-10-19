import { DenseLayer } from '../dense';
import { Matrix, type Parameter, type Rng } from '@sim/core';

/**
 * A GAN on two-dimensional data.
 *
 *   generator      z (noise) → hidden → hidden → (x, y)
 *   discriminator  (x, y)    → hidden → hidden → probability "this point is real"
 *
 * The discriminator minimises the usual binary cross-entropy on a batch of real and a batch of
 * fake points. The generator can be trained with two losses:
 *   nonsaturating  −mean log D(G(z))       (the practical default)
 *   minimax        +mean log(1 − D(G(z)))  (the original paper: its gradient vanishes while the
 *                                           discriminator wins)
 */
export const GENERATOR_LOSSES = ['nonsaturating', 'minimax'] as const;
export type GeneratorLoss = (typeof GENERATOR_LOSSES)[number];

export const generatorLossInfo: Record<GeneratorLoss, { label: string; note: string }> = {
  nonsaturating: {
    label: 'Non-saturating',
    note: 'Maximise log D(G(z)): strong gradients even when the discriminator is winning.',
  },
  minimax: {
    label: 'Minimax',
    note: 'Minimise log(1 − D(G(z))): the original formulation; the gradient fades when D is confident.',
  },
};

export interface GanArchitecture {
  noiseDim: number;
  hidden: number;
}

export const GAN_LIMITS = { noiseDim: [2, 8], hidden: [8, 64] } as const;

export function validateGanArchitecture(a: GanArchitecture): void {
  const check = (v: number, [lo, hi]: readonly [number, number], name: string) => {
    if (!Number.isInteger(v) || v < lo || v > hi)
      throw new RangeError(`${name} must be an integer in [${lo}, ${hi}]`);
  };
  check(a.noiseDim, GAN_LIMITS.noiseDim, 'Noise size');
  check(a.hidden, GAN_LIMITS.hidden, 'Hidden size');
}

const EPS = 1e-12;
const clampP = (p: number) => Math.min(1 - EPS, Math.max(EPS, p));

export class Generator {
  readonly layers: DenseLayer[];

  constructor(arch: GanArchitecture, rng: Rng) {
    this.layers = [
      new DenseLayer(arch.noiseDim, arch.hidden, 'relu', rng),
      new DenseLayer(arch.hidden, arch.hidden, 'relu', rng),
      new DenseLayer(arch.hidden, 2, 'linear', rng),
    ];
  }

  get parameters(): Parameter[] {
    return this.layers.flatMap((l) => l.parameters);
  }

  /** Noise `rows × noiseDim` → points `rows × 2`. */
  forward(z: Matrix, cache = true): Matrix {
    let a = z;
    for (const layer of this.layers) a = layer.forward(a, cache);
    return a;
  }

  /** Backprop from dL/dPoints (parameter gradients are averaged over the rows). */
  backward(dPoints: Matrix): void {
    let d = dPoints;
    for (let i = this.layers.length - 1; i >= 0; i--)
      d = (this.layers[i] as DenseLayer).backward(d);
  }
}

export class Discriminator {
  readonly layers: DenseLayer[];

  constructor(arch: GanArchitecture, rng: Rng) {
    this.layers = [
      new DenseLayer(2, arch.hidden, 'leaky_relu', rng),
      new DenseLayer(arch.hidden, arch.hidden, 'leaky_relu', rng),
      new DenseLayer(arch.hidden, 1, 'sigmoid', rng),
    ];
  }

  get parameters(): Parameter[] {
    return this.layers.flatMap((l) => l.parameters);
  }

  /** Points `rows × 2` → probabilities of being real, `rows × 1`. */
  forward(x: Matrix, cache = true): Matrix {
    let a = x;
    for (const layer of this.layers) a = layer.forward(a, cache);
    return a;
  }

  /**
   * Backprop from dL/dLogit of the last layer. Parameter gradients are averaged over the rows;
   * the returned dL/dPoints is per point (not averaged).
   */
  backwardFromLogits(dZ: Matrix): Matrix {
    let d = (this.layers[2] as DenseLayer).backwardFromPreActivation(dZ);
    d = (this.layers[1] as DenseLayer).backward(d);
    return (this.layers[0] as DenseLayer).backward(d);
  }
}

export interface DiscriminatorStep {
  loss: number;
  /** Mean D(x) on the real points. */
  dReal: number;
  /** Mean D(x) on the fake points. */
  dFake: number;
}

export interface GeneratorStep {
  loss: number;
  /** Mean D(G(z)). */
  dFake: number;
  /** Mean per-point norm of the gradient the discriminator hands to the generator. */
  signal: number;
}

export class GanNet {
  readonly generator: Generator;
  readonly discriminator: Discriminator;

  constructor(
    readonly arch: GanArchitecture,
    rng: Rng,
  ) {
    validateGanArchitecture(arch);
    this.generator = new Generator(arch, rng);
    this.discriminator = new Discriminator(arch, rng);
  }

  /**
   * Fills the discriminator's gradients for one batch: real points (label 1) and the generator's
   * output for `z` (label 0, treated as constants). Loss is the mean BCE over both halves.
   */
  discriminatorGradients(real: Matrix, z: Matrix): DiscriminatorStep {
    const fake = this.generator.forward(z, false);
    const n = real.rows + fake.rows;
    const x = new Matrix(n, 2);
    x.data.set(real.data, 0);
    x.data.set(fake.data, real.data.length);
    const p = this.discriminator.forward(x, true);
    const dZ = new Matrix(n, 1);
    let loss = 0;
    let dReal = 0;
    let dFake = 0;
    for (let i = 0; i < n; i++) {
      const isReal = i < real.rows;
      const prob = p.data[i] as number;
      dZ.data[i] = prob - (isReal ? 1 : 0);
      loss -= Math.log(clampP(isReal ? prob : 1 - prob));
      if (isReal) dReal += prob;
      else dFake += prob;
    }
    this.discriminator.backwardFromLogits(dZ);
    return { loss: loss / n, dReal: dReal / real.rows, dFake: dFake / fake.rows };
  }

  /**
   * Fills the generator's gradients for noise `z` by backpropagating the chosen loss through the
   * discriminator. (The discriminator's own gradient buffers are overwritten and must not be used.)
   */
  generatorGradients(z: Matrix, loss: GeneratorLoss): GeneratorStep {
    const fake = this.generator.forward(z, true);
    const p = this.discriminator.forward(fake, true);
    const m = z.rows;
    const dZ = new Matrix(m, 1);
    let total = 0;
    let dFake = 0;
    for (let i = 0; i < m; i++) {
      const prob = p.data[i] as number;
      dFake += prob;
      if (loss === 'nonsaturating') {
        dZ.data[i] = prob - 1;
        total -= Math.log(clampP(prob));
      } else {
        dZ.data[i] = -prob;
        total += Math.log(clampP(1 - prob));
      }
    }
    const dPoints = this.discriminator.backwardFromLogits(dZ);
    let signal = 0;
    for (let i = 0; i < m; i++)
      signal += Math.hypot(dPoints.data[2 * i] as number, dPoints.data[2 * i + 1] as number);
    this.generator.backward(dPoints);
    return { loss: total / m, dFake: dFake / m, signal: signal / m };
  }

  /** Probability of "real" at each of `points` (`n × 2`), without disturbing training. */
  discriminate(points: Float64Array): Float64Array {
    const n = points.length / 2;
    return Float64Array.from(
      this.discriminator.forward(new Matrix(n, 2, Float64Array.from(points)), false).data,
    );
  }

  /** Generator output for noise `z` (`n × noiseDim`), without disturbing training. */
  generate(z: Float64Array): Float64Array {
    const n = z.length / this.arch.noiseDim;
    return Float64Array.from(
      this.generator.forward(new Matrix(n, this.arch.noiseDim, Float64Array.from(z)), false).data,
    );
  }
}
