import { Matrix, Optimizer, createRng, type Rng } from '@sim/core';
import { GAN_DATASETS, hasModes, sampleReal, type GanDataset } from './datasets';
import { fractionNearData, modesCovered, slicedWasserstein } from './metrics';
import {
  GENERATOR_LOSSES,
  GanNet,
  validateGanArchitecture,
  type GanArchitecture,
  type GeneratorLoss,
} from './network';

export interface GanLabConfig extends GanArchitecture {
  dataset: GanDataset;
  lossG: GeneratorLoss;
  /** Discriminator updates per generator update. */
  dSteps: number;
  lrG: number;
  lrD: number;
  batchSize: number;
  seed: number;
}

export const DEFAULT_GAN_LAB: GanLabConfig = {
  dataset: 'gaussians8',
  noiseDim: 2,
  hidden: 32,
  lossG: 'nonsaturating',
  dSteps: 1,
  lrG: 0.002,
  lrD: 0.002,
  batchSize: 128,
  seed: 11,
};

export const GAN_LAB_LIMITS = { dSteps: [1, 10], batchSize: [16, 512] } as const;

/** Statistics are recorded every this many iterations (averaging the losses in between). */
export const RECORD_EVERY = 10;
/** Size of the fixed noise set and of the real reference set used to judge quality. */
export const EVAL_POINTS = 300;

export interface GanStats {
  iteration: number;
  lossD: number;
  lossG: number;
  /** Mean D(x) on real and on generated points. */
  dReal: number;
  dFake: number;
  /** Mean size of the gradient the generator receives from the discriminator. */
  signal: number;
  /** Sliced Wasserstein distance between generated and real points (0 = identical). */
  distance: number;
  /** Share of generated points that land close to real data. */
  near: number;
  /** Clusters covered (eight-Gaussians only). */
  modes: number | null;
}

export function validateGanLab(c: GanLabConfig): void {
  validateGanArchitecture(c);
  if (!(GAN_DATASETS as readonly string[]).includes(c.dataset))
    throw new RangeError('Unknown dataset');
  if (!(GENERATOR_LOSSES as readonly string[]).includes(c.lossG))
    throw new RangeError('Unknown generator loss');
  const [dLo, dHi] = GAN_LAB_LIMITS.dSteps;
  if (!Number.isInteger(c.dSteps) || c.dSteps < dLo || c.dSteps > dHi)
    throw new RangeError(`Discriminator steps must be an integer in [${dLo}, ${dHi}]`);
  const [bLo, bHi] = GAN_LAB_LIMITS.batchSize;
  if (!Number.isInteger(c.batchSize) || c.batchSize < bLo || c.batchSize > bHi)
    throw new RangeError(`Batch size must be an integer in [${bLo}, ${bHi}]`);
  if (!(c.lrG > 0) || !(c.lrD > 0)) throw new RangeError('Learning rates must be positive');
}

const stream = (seed: number, index: number) =>
  createRng((seed + Math.imul(index + 1, 0x9e3779b1)) >>> 0);

const dataKey = (c: GanLabConfig) => JSON.stringify([c.dataset, c.seed]);
const netKey = (c: GanLabConfig) => JSON.stringify([c.dataset, c.noiseDim, c.hidden, c.seed]);

const noise = (rng: Rng, rows: number, cols: number): Matrix =>
  new Matrix(
    rows,
    cols,
    Float64Array.from({ length: rows * cols }, () => rng.normal()),
  );

/** A GAN, its data stream and its training loop behind one small API. */
export class GanSession {
  private cfg: GanLabConfig;
  private keys = { data: '', net: '' };
  net!: GanNet;
  private optG!: Optimizer;
  private optD!: Optimizer;
  private dataRng = createRng(0);
  private noiseRng = createRng(0);
  private sampleRng = createRng(0);
  /** Real points used as the reference for quality metrics and for display. */
  reference!: Float64Array;
  /** Fixed noise: the same inputs every frame, so the generated cloud moves smoothly. */
  private fixedNoise!: Float64Array;
  iteration = 0;
  history: GanStats[] = [];
  private acc = { lossD: 0, lossG: 0, dReal: 0, dFake: 0, signal: 0, count: 0 };

  constructor(config: GanLabConfig) {
    validateGanLab(config);
    this.cfg = { ...config };
    this.build(true);
  }

  get config(): Readonly<GanLabConfig> {
    return this.cfg;
  }

  /** Hyper-parameters apply live; data or architecture changes rebuild (and restart training). */
  update(patch: Partial<GanLabConfig>): 'live' | 'rebuilt' {
    const next = { ...this.cfg, ...patch };
    validateGanLab(next);
    const dataChanged = dataKey(next) !== this.keys.data;
    const netChanged = netKey(next) !== this.keys.net;
    this.cfg = next;
    if (dataChanged || netChanged) {
      this.build(dataChanged);
      return 'rebuilt';
    }
    return 'live';
  }

  reset(): void {
    this.build(false);
  }

  /**
   * Runs `count` iterations: each is `dSteps` discriminator updates on fresh real/fake batches
   * followed by one generator update. Returns the statistics recorded meanwhile.
   */
  train(count: number): GanStats[] {
    const recorded: GanStats[] = [];
    const { batchSize, dSteps, lossG, lrD, lrG, dataset, noiseDim } = this.cfg;
    const { net } = this;
    for (let n = 0; n < count; n++) {
      let d = { loss: 0, dReal: 0, dFake: 0 };
      for (let k = 0; k < dSteps; k++) {
        const real = new Matrix(batchSize, 2, sampleReal(dataset, batchSize, this.dataRng));
        d = net.discriminatorGradients(real, noise(this.noiseRng, batchSize, noiseDim));
        this.optD.step(net.discriminator.parameters, lrD);
      }
      const g = net.generatorGradients(noise(this.noiseRng, batchSize, noiseDim), lossG);
      this.optG.step(net.generator.parameters, lrG);

      this.iteration += 1;
      const a = this.acc;
      a.lossD += d.loss;
      a.lossG += g.loss;
      a.dReal += d.dReal;
      a.dFake += g.dFake;
      a.signal += g.signal;
      a.count += 1;
      if (this.iteration % RECORD_EVERY === 0) {
        const stats = this.snapshot();
        this.history.push(stats);
        recorded.push(stats);
      }
    }
    return recorded;
  }

  /** Statistics of the current network (losses are averaged over the iterations since the last record). */
  snapshot(): GanStats {
    const a = this.acc;
    const c = Math.max(1, a.count);
    const fake = this.fixedFake();
    const stats: GanStats = {
      iteration: this.iteration,
      lossD: a.lossD / c,
      lossG: a.lossG / c,
      dReal: a.dReal / c,
      dFake: a.dFake / c,
      signal: a.signal / c,
      distance: slicedWasserstein(fake, this.reference),
      near: fractionNearData(fake, this.reference),
      modes: hasModes(this.cfg.dataset) ? modesCovered(fake).count : null,
    };
    this.acc = { lossD: 0, lossG: 0, dReal: 0, dFake: 0, signal: 0, count: 0 };
    return stats;
  }

  /** The generator's output for the fixed noise set (`EVAL_POINTS × 2`). */
  fixedFake(): Float64Array {
    return this.net.generate(this.fixedNoise);
  }

  /** Fresh generated points. */
  generate(count: number): Float64Array {
    return this.net.generate(noise(this.sampleRng, count, this.cfg.noiseDim).data);
  }

  /** `n × n` probabilities D(x) over the square `[-extent, extent]²`, row by row from the top. */
  discriminatorGrid(extent: number, n: number): Float64Array {
    const points = new Float64Array(n * n * 2);
    for (let row = 0; row < n; row++) {
      for (let col = 0; col < n; col++) {
        const i = row * n + col;
        points[2 * i] = ((col + 0.5) / n) * 2 * extent - extent;
        points[2 * i + 1] = extent - ((row + 0.5) / n) * 2 * extent;
      }
    }
    return this.net.discriminate(points);
  }

  /**
   * Where the generator sends a regular `n × n` grid of its first two noise dimensions over
   * `[-span, span]²` (other dimensions are 0): shows how it folds and stretches its input.
   */
  warpedGrid(n: number, span = 2): Float64Array {
    const { noiseDim } = this.cfg;
    const z = new Float64Array(n * n * noiseDim);
    for (let row = 0; row < n; row++) {
      for (let col = 0; col < n; col++) {
        const i = row * n + col;
        z[i * noiseDim] = (col / (n - 1)) * 2 * span - span;
        z[i * noiseDim + 1] = (row / (n - 1)) * 2 * span - span;
      }
    }
    return this.net.generate(z);
  }

  /** Fresh real points, e.g. for plotting more than the reference set. */
  real(count: number): Float64Array {
    return sampleReal(this.cfg.dataset, count, this.sampleRng);
  }

  private build(regenerate: boolean): void {
    const c = this.cfg;
    if (regenerate) {
      this.reference = sampleReal(c.dataset, EVAL_POINTS, stream(c.seed, 0));
    }
    this.net = new GanNet(c, stream(c.seed, 1));
    this.optG = new Optimizer('adam', { beta1: 0.5 });
    this.optD = new Optimizer('adam', { beta1: 0.5 });
    this.dataRng = stream(c.seed, 2);
    this.noiseRng = stream(c.seed, 3);
    this.sampleRng = stream(c.seed, 4);
    this.fixedNoise = noise(stream(c.seed, 5), EVAL_POINTS, c.noiseDim).data;
    this.iteration = 0;
    this.history = [];
    this.acc = { lossD: 0, lossG: 0, dReal: 0, dFake: 0, signal: 0, count: 0 };
    this.keys = { data: dataKey(c), net: netKey(c) };
  }
}
