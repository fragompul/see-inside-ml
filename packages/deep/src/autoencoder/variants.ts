import { Matrix, Parameter, type Rng } from '@sim/core';
import { DenseLayer } from '../dense';

/**
 * Two ways of taking control of what a variational autoencoder produces.
 *
 * The conditional VAE (`ConditionalVae`) is told the class (a one-hot digit) both when it encodes and when it
 * decodes, so the code no longer has to carry "which digit": it is free to carry style, and the class can be
 * chosen at generation time.
 *
 * The vector-quantised autoencoder (`VqVae`) replaces the continuous code by a choice from a learned
 * codebook: the encoder's output is snapped to the nearest of K entries and the decoder sees only that entry.
 * Gradients pass through the snapping unchanged (the straight-through estimator); the codebook is pulled
 * towards the encoder outputs and the encoder is held close to its codebook entry.
 */

const EPS = 1e-12;

const concat = (a: Matrix, b: Matrix): Matrix => {
  const out = new Matrix(a.rows, a.cols + b.cols);
  for (let r = 0; r < a.rows; r++) {
    out.data.set(a.data.subarray(r * a.cols, (r + 1) * a.cols), r * out.cols);
    out.data.set(b.data.subarray(r * b.cols, (r + 1) * b.cols), r * out.cols + a.cols);
  }
  return out;
};

export const oneHot = (labels: ArrayLike<number>, classes: number): Matrix => {
  const m = new Matrix(labels.length, classes);
  for (let i = 0; i < labels.length; i++) m.data[i * classes + (labels[i] as number)] = 1;
  return m;
};

/** Cross-entropy of sigmoid outputs against pixel targets, summed over pixels and averaged over rows. */
function reconstruction(target: Matrix, xhat: Matrix): number {
  let recon = 0;
  for (let i = 0; i < target.data.length; i++) {
    const x = target.data[i] as number;
    const p = Math.min(1 - EPS, Math.max(EPS, xhat.data[i] as number));
    recon -= x * Math.log(p) + (1 - x) * Math.log(1 - p);
  }
  return recon / target.rows;
}

export interface VaeLoss {
  recon: number;
  kl: number;
  total: number;
}

export class ConditionalVae {
  readonly enc1: DenseLayer;
  readonly encMu: DenseLayer;
  readonly encLogVar: DenseLayer;
  readonly dec1: DenseLayer;
  readonly dec2: DenseLayer;
  private cache: { mu: Matrix; logvar: Matrix; eps: Matrix | null } | null = null;

  constructor(
    readonly pixels: number,
    readonly classes: number,
    readonly latent: number,
    readonly hidden: number,
    rng: Rng,
  ) {
    this.enc1 = new DenseLayer(pixels + classes, hidden, 'relu', rng);
    this.encMu = new DenseLayer(hidden, latent, 'linear', rng);
    this.encLogVar = new DenseLayer(hidden, latent, 'linear', rng);
    this.encLogVar.biases.data.fill(-4);
    this.encLogVar.weights.data.forEach((_, i) => {
      this.encLogVar.weights.data[i] = (this.encLogVar.weights.data[i] as number) * 0.1;
    });
    this.dec1 = new DenseLayer(latent + classes, hidden, 'relu', rng);
    this.dec2 = new DenseLayer(hidden, pixels, 'sigmoid', rng);
  }

  get parameters(): Parameter[] {
    return [
      ...this.enc1.parameters,
      ...this.encMu.parameters,
      ...this.encLogVar.parameters,
      ...this.dec1.parameters,
      ...this.dec2.parameters,
    ];
  }

  /** `eps` (rows × latent, standard normal) switches on sampling; without it the decoder gets the mean. */
  forward(x: Matrix, labels: ArrayLike<number>, eps: Matrix | null = null, cache = true) {
    const tag = oneHot(labels, this.classes);
    const h = this.enc1.forward(concat(x, tag), cache);
    const mu = this.encMu.forward(h, cache);
    const logvar = this.encLogVar.forward(h, cache);
    let z = mu;
    if (eps) {
      z = new Matrix(mu.rows, mu.cols);
      for (let i = 0; i < z.data.length; i++)
        z.data[i] =
          (mu.data[i] as number) +
          Math.exp(0.5 * (logvar.data[i] as number)) * (eps.data[i] as number);
    }
    const xhat = this.dec2.forward(this.dec1.forward(concat(z, tag), cache), cache);
    if (cache) this.cache = { mu, logvar, eps };
    return { mu, logvar, z, xhat };
  }

  loss(target: Matrix, out: { mu: Matrix; logvar: Matrix; xhat: Matrix }, beta: number): VaeLoss {
    const recon = reconstruction(target, out.xhat);
    let kl = 0;
    for (let i = 0; i < out.mu.data.length; i++) {
      const mu = out.mu.data[i] as number;
      const lv = out.logvar.data[i] as number;
      kl -= 0.5 * (1 + lv - mu * mu - Math.exp(lv));
    }
    kl /= target.rows;
    return { recon, kl, total: recon + beta * kl };
  }

  backward(target: Matrix, out: { xhat: Matrix }, beta: number): void {
    const c = this.cache;
    if (!c) throw new Error('backward() called before forward()');
    const dOut = new Matrix(target.rows, target.cols);
    for (let i = 0; i < dOut.data.length; i++)
      dOut.data[i] = (out.xhat.data[i] as number) - (target.data[i] as number);
    const dIn = this.dec1.backward(this.dec2.backwardFromPreActivation(dOut));
    const dMu = new Matrix(target.rows, this.latent);
    const dLogVar = new Matrix(target.rows, this.latent);
    for (let r = 0; r < target.rows; r++)
      for (let j = 0; j < this.latent; j++) {
        const i = r * this.latent + j;
        const mu = c.mu.data[i] as number;
        const lv = c.logvar.data[i] as number;
        const g = dIn.data[r * dIn.cols + j] as number;
        const eps = c.eps ? (c.eps.data[i] as number) : 0;
        dMu.data[i] = g + beta * mu;
        dLogVar.data[i] = g * eps * 0.5 * Math.exp(0.5 * lv) + beta * 0.5 * (Math.exp(lv) - 1);
      }
    const dhMu = this.encMu.backwardFromPreActivation(dMu);
    const dhLv = this.encLogVar.backwardFromPreActivation(dLogVar);
    const dh = new Matrix(dhMu.rows, dhMu.cols);
    for (let i = 0; i < dh.data.length; i++)
      dh.data[i] = (dhMu.data[i] as number) + (dhLv.data[i] as number);
    this.enc1.backward(dh);
  }

  /** The image painted for a code and a chosen class. */
  decode(z: ArrayLike<number>, label: number): Float64Array {
    const code = new Matrix(1, this.latent, Float64Array.from(z));
    const tag = oneHot([label], this.classes);
    return Float64Array.from(
      this.dec2.forward(this.dec1.forward(concat(code, tag), false), false).data,
    );
  }

  /** The mean code of an image with its own class. */
  encode(image: Float64Array, label: number): Float64Array {
    return Float64Array.from(
      this.forward(new Matrix(1, this.pixels, Float64Array.from(image)), [label], null, false).mu
        .data,
    );
  }
}

export interface VqLoss {
  recon: number;
  /** Mean squared distance between each code and its codebook entry, pulling the entry (stop-gradient on the encoder). */
  codebook: number;
  /** The same distance, pulling the encoder (stop-gradient on the codebook), before weighting. */
  commit: number;
  total: number;
}

export class VqVae {
  readonly enc1: DenseLayer;
  readonly encZ: DenseLayer;
  readonly dec1: DenseLayer;
  readonly dec2: DenseLayer;
  readonly codebook: Matrix;
  readonly codebookParam: Parameter;
  private cache: { z: Matrix; index: Int32Array } | null = null;

  constructor(
    readonly pixels: number,
    readonly entries: number,
    readonly dim: number,
    readonly hidden: number,
    rng: Rng,
  ) {
    this.enc1 = new DenseLayer(pixels, hidden, 'relu', rng);
    this.encZ = new DenseLayer(hidden, dim, 'linear', rng);
    this.dec1 = new DenseLayer(dim, hidden, 'relu', rng);
    this.dec2 = new DenseLayer(hidden, pixels, 'sigmoid', rng);
    this.codebook = new Matrix(entries, dim);
    for (let i = 0; i < this.codebook.data.length; i++) this.codebook.data[i] = rng.normal();
    this.codebookParam = new Parameter(this.codebook.data, false);
  }

  get parameters(): Parameter[] {
    return [
      ...this.enc1.parameters,
      ...this.encZ.parameters,
      this.codebookParam,
      ...this.dec1.parameters,
      ...this.dec2.parameters,
    ];
  }

  /** Sets every codebook entry to the encoder output of a randomly chosen image, so that no entry starts out far from the data. */
  initialiseCodebook(images: Matrix, rng: Rng): void {
    const z = this.encZ.forward(this.enc1.forward(images, false), false);
    for (let k = 0; k < this.entries; k++) {
      const pick = Math.floor(rng.next() * z.rows);
      this.codebook.data.set(z.data.subarray(pick * this.dim, (pick + 1) * this.dim), k * this.dim);
    }
  }

  nearest(z: Matrix): Int32Array {
    const index = new Int32Array(z.rows);
    for (let r = 0; r < z.rows; r++) {
      let best = Infinity;
      for (let k = 0; k < this.entries; k++) {
        let d = 0;
        for (let j = 0; j < this.dim; j++)
          d +=
            ((z.data[r * this.dim + j] as number) -
              (this.codebook.data[k * this.dim + j] as number)) **
            2;
        if (d < best) {
          best = d;
          index[r] = k;
        }
      }
    }
    return index;
  }

  forward(x: Matrix, cache = true) {
    const z = this.encZ.forward(this.enc1.forward(x, cache), cache);
    const index = this.nearest(z);
    const quantised = new Matrix(z.rows, this.dim);
    for (let r = 0; r < z.rows; r++)
      quantised.data.set(
        this.codebook.data.subarray(
          (index[r] as number) * this.dim,
          ((index[r] as number) + 1) * this.dim,
        ),
        r * this.dim,
      );
    const xhat = this.dec2.forward(this.dec1.forward(quantised, cache), cache);
    if (cache) this.cache = { z, index };
    return { z, index, quantised, xhat };
  }

  loss(
    target: Matrix,
    out: { z: Matrix; quantised: Matrix; xhat: Matrix },
    commitment: number,
  ): VqLoss {
    const recon = reconstruction(target, out.xhat);
    let dist = 0;
    for (let i = 0; i < out.z.data.length; i++)
      dist += ((out.z.data[i] as number) - (out.quantised.data[i] as number)) ** 2;
    dist /= target.rows;
    // the same squared distance is counted twice: once to move the codebook, once (weighted) to move the encoder
    return { recon, codebook: dist, commit: dist, total: recon + dist + commitment * dist };
  }

  backward(target: Matrix, out: { xhat: Matrix; quantised: Matrix }, commitment: number): void {
    const c = this.cache;
    if (!c) throw new Error('backward() called before forward()');
    const dOut = new Matrix(target.rows, target.cols);
    for (let i = 0; i < dOut.data.length; i++)
      dOut.data[i] = (out.xhat.data[i] as number) - (target.data[i] as number);
    // straight-through: the gradient of the decoder input is handed to the encoder output as it is
    const dq = this.dec1.backward(this.dec2.backwardFromPreActivation(dOut));
    const dz = new Matrix(target.rows, this.dim);
    this.codebookParam.grad.fill(0);
    const rows = target.rows;
    for (let r = 0; r < rows; r++) {
      const k = c.index[r] as number;
      for (let j = 0; j < this.dim; j++) {
        const z = c.z.data[r * this.dim + j] as number;
        const e = this.codebook.data[k * this.dim + j] as number;
        dz.data[r * this.dim + j] =
          (dq.data[r * this.dim + j] as number) + 2 * commitment * (z - e);
        this.codebookParam.grad[k * this.dim + j] =
          (this.codebookParam.grad[k * this.dim + j] as number) + (2 * (e - z)) / rows;
      }
    }
    this.enc1.backward(this.encZ.backwardFromPreActivation(dz));
  }

  /** The image painted for one codebook entry. */
  decodeEntry(k: number): Float64Array {
    const code = new Matrix(
      1,
      this.dim,
      this.codebook.data.slice(k * this.dim, (k + 1) * this.dim),
    );
    return Float64Array.from(this.dec2.forward(this.dec1.forward(code, false), false).data);
  }

  /** Which entry an image is snapped to. */
  assign(image: Float64Array): number {
    return this.forward(new Matrix(1, this.pixels, Float64Array.from(image)), false)
      .index[0] as number;
  }
}
