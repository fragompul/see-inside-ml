import { DenseLayer } from '../dense';
import { Matrix, type Parameter, type Rng } from '@sim/core';

/**
 * Autoencoder and variational autoencoder over flat images in [0, 1].
 *
 *   x → [hidden, ReLU] → latent code → [hidden, ReLU] → pixels (sigmoid)
 *
 * The plain autoencoder (`ae`) squeezes the image into `latent` numbers and learns to rebuild it.
 * The variational one (`vae`) encodes a mean μ and a log-variance log σ² per latent dimension,
 * samples z = μ + σ·ε during training and adds a KL term that keeps the codes close to N(0, I).
 *
 * Loss per image (averaged over the batch):
 *   reconstruction = −Σ_pixels [ x·log p + (1 − x)·log(1 − p) ]
 *   kl             = −½ Σ_latent (1 + log σ² − μ² − σ²)             (VAE only)
 *   total          = reconstruction + β · kl
 */
export const AE_MODES = ['ae', 'vae'] as const;
export type AutoencoderMode = (typeof AE_MODES)[number];

export interface AutoencoderConfig {
  mode: AutoencoderMode;
  latent: number;
  hidden: number;
}

export const AE_LIMITS = { latent: [2, 8], hidden: [4, 128] } as const;

export function validateAutoencoder(c: AutoencoderConfig): void {
  if (!(AE_MODES as readonly string[]).includes(c.mode)) throw new RangeError('Unknown mode');
  const check = (v: number, [lo, hi]: readonly [number, number], name: string) => {
    if (!Number.isInteger(v) || v < lo || v > hi)
      throw new RangeError(`${name} must be an integer in [${lo}, ${hi}]`);
  };
  check(c.latent, AE_LIMITS.latent, 'Latent size');
  check(c.hidden, AE_LIMITS.hidden, 'Hidden size');
}

export interface AutoencoderLoss {
  /** Reconstruction term, per image. */
  recon: number;
  /** KL term, per image (0 for the plain autoencoder). */
  kl: number;
  /** `recon + beta · kl`. */
  total: number;
}

export interface AutoencoderOutput {
  mu: Matrix;
  logvar: Matrix | null;
  /** The code the decoder saw (`mu`, or `mu + σ·ε` when noise was supplied). */
  z: Matrix;
  /** Predicted pixel intensities. */
  xhat: Matrix;
}

const EPS_FLOOR = 1e-12;

export class AutoencoderNet {
  readonly enc1: DenseLayer;
  readonly encMu: DenseLayer;
  readonly encLogVar: DenseLayer | null;
  readonly dec1: DenseLayer;
  readonly dec2: DenseLayer;

  private cache: { mu: Matrix; logvar: Matrix | null; eps: Matrix | null } | null = null;

  constructor(
    readonly config: AutoencoderConfig,
    readonly pixels: number,
    rng: Rng,
  ) {
    validateAutoencoder(config);
    const { latent, hidden, mode } = config;
    this.enc1 = new DenseLayer(pixels, hidden, 'relu', rng);
    this.encMu = new DenseLayer(hidden, latent, 'linear', rng);
    this.encLogVar = mode === 'vae' ? new DenseLayer(hidden, latent, 'linear', rng) : null;
    if (this.encLogVar) {
      // start with σ ≈ 1·e^{-2}: small, so early training is not drowned in sampling noise
      this.encLogVar.biases.data.fill(-4);
      this.encLogVar.weights.data.forEach((_, i) => {
        this.encLogVar!.weights.data[i] = (this.encLogVar!.weights.data[i] as number) * 0.1;
      });
    }
    this.dec1 = new DenseLayer(latent, hidden, 'relu', rng);
    this.dec2 = new DenseLayer(hidden, pixels, 'sigmoid', rng);
  }

  get parameters(): Parameter[] {
    return [
      ...this.enc1.parameters,
      ...this.encMu.parameters,
      ...(this.encLogVar?.parameters ?? []),
      ...this.dec1.parameters,
      ...this.dec2.parameters,
    ];
  }

  get parameterCount(): number {
    return this.parameters.reduce((sum, p) => sum + p.data.length, 0);
  }

  /**
   * Forward pass over a batch (`rows × pixels`). For the VAE, `eps` (`rows × latent`, standard
   * normal) switches on sampling; without it the decoder is fed the mean.
   * `cache: false` leaves everything a pending backward pass needs untouched.
   */
  forward(x: Matrix, eps: Matrix | null = null, cache = true): AutoencoderOutput {
    const h = this.enc1.forward(x, cache);
    const mu = this.encMu.forward(h, cache);
    const logvar = this.encLogVar ? this.encLogVar.forward(h, cache) : null;
    let z = mu;
    const noise = logvar && eps ? eps : null;
    if (logvar && noise) {
      z = new Matrix(mu.rows, mu.cols);
      for (let i = 0; i < z.data.length; i++) {
        z.data[i] =
          (mu.data[i] as number) +
          Math.exp(0.5 * (logvar.data[i] as number)) * (noise.data[i] as number);
      }
    }
    const xhat = this.dec2.forward(this.dec1.forward(z, cache), cache);
    if (cache) this.cache = { mu, logvar, eps: noise };
    return { mu, logvar, z, xhat };
  }

  /** Loss of an output against the clean targets, averaged over the batch. */
  loss(target: Matrix, out: AutoencoderOutput, beta: number): AutoencoderLoss {
    const rows = target.rows;
    let recon = 0;
    for (let i = 0; i < target.data.length; i++) {
      const x = target.data[i] as number;
      const p = Math.min(1 - EPS_FLOOR, Math.max(EPS_FLOOR, out.xhat.data[i] as number));
      recon -= x * Math.log(p) + (1 - x) * Math.log(1 - p);
    }
    let kl = 0;
    if (out.logvar) {
      for (let i = 0; i < out.mu.data.length; i++) {
        const mu = out.mu.data[i] as number;
        const lv = out.logvar.data[i] as number;
        kl -= 0.5 * (1 + lv - mu * mu - Math.exp(lv));
      }
    }
    return { recon: recon / rows, kl: kl / rows, total: recon / rows + (beta * kl) / rows };
  }

  /** Backpropagation for the batch of the last cached `forward` (gradients averaged over rows). */
  backward(target: Matrix, out: AutoencoderOutput, beta: number): void {
    const c = this.cache;
    if (!c) throw new Error('backward() called before forward()');
    // sigmoid + cross-entropy fuse into (p − x)
    const dOut = new Matrix(target.rows, target.cols);
    for (let i = 0; i < dOut.data.length; i++)
      dOut.data[i] = (out.xhat.data[i] as number) - (target.data[i] as number);
    const dz = this.dec1.backward(this.dec2.backwardFromPreActivation(dOut));

    if (!c.logvar || !this.encLogVar) {
      this.enc1.backward(this.encMu.backwardFromPreActivation(dz));
      return;
    }
    const dMu = new Matrix(dz.rows, dz.cols);
    const dLogVar = new Matrix(dz.rows, dz.cols);
    for (let i = 0; i < dz.data.length; i++) {
      const mu = c.mu.data[i] as number;
      const lv = c.logvar.data[i] as number;
      const std = Math.exp(0.5 * lv);
      const eps = c.eps ? (c.eps.data[i] as number) : 0;
      const g = dz.data[i] as number;
      dMu.data[i] = g + beta * mu;
      dLogVar.data[i] = g * eps * 0.5 * std + beta * 0.5 * (Math.exp(lv) - 1);
    }
    const dhMu = this.encMu.backwardFromPreActivation(dMu);
    const dhLv = this.encLogVar.backwardFromPreActivation(dLogVar);
    const dh = new Matrix(dhMu.rows, dhMu.cols);
    for (let i = 0; i < dh.data.length; i++)
      dh.data[i] = (dhMu.data[i] as number) + (dhLv.data[i] as number);
    this.enc1.backward(dh);
  }

  /** The code of one image (mean and, for the VAE, log-variance). Does not disturb training. */
  encode(image: Float64Array): { mu: Float64Array; logvar: Float64Array | null } {
    const out = this.forward(new Matrix(1, this.pixels, Float64Array.from(image)), null, false);
    return {
      mu: Float64Array.from(out.mu.data),
      logvar: out.logvar && Float64Array.from(out.logvar.data),
    };
  }

  /** The image the decoder paints for one latent code. */
  decode(z: ArrayLike<number>): Float64Array {
    if (z.length !== this.config.latent) throw new RangeError('Wrong latent size');
    const code = new Matrix(1, z.length, Float64Array.from(z as ArrayLike<number>));
    return Float64Array.from(this.dec2.forward(this.dec1.forward(code, false), false).data);
  }

  /** Decodes many codes at once (`count × latent`, row-major) into `count × pixels` images. */
  decodeMany(codes: Float64Array, count: number): Float64Array {
    if (codes.length !== count * this.config.latent) throw new RangeError('Wrong number of codes');
    const code = new Matrix(count, this.config.latent, Float64Array.from(codes));
    return Float64Array.from(this.dec2.forward(this.dec1.forward(code, false), false).data);
  }

  /** Encode then decode (using the mean code). */
  reconstruct(image: Float64Array): Float64Array {
    return this.decode(this.encode(image).mu);
  }
}
