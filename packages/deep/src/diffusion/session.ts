import { GAN_DATASETS, sampleReal, type GanDataset } from '../gan/datasets';
import { slicedWasserstein } from '../gan/metrics';
import { Optimizer, OPTIMIZER_NAMES, type OptimizerName, createRng } from '@sim/core';
import { Denoiser, validateDenoiser } from './network';
import { SAMPLERS, sampleDiffusion, type SampleResult, type SamplerName } from './sampler';
import { makeSchedule, noiseTo, type NoiseSchedule, type ScheduleKind } from './schedule';

export interface DiffusionLabConfig {
  dataset: GanDataset;
  /** Number of noising steps T. */
  steps: number;
  schedule: ScheduleKind;
  hidden: number;
  optimizer: OptimizerName;
  lr: number;
  batchSize: number;
  sampler: SamplerName;
  /** DDIM: number of denoising steps used when sampling. */
  sampleSteps: number;
  seed: number;
}

export const DEFAULT_DIFFUSION_LAB: DiffusionLabConfig = {
  dataset: 'gaussians8',
  steps: 50,
  schedule: 'linear',
  hidden: 64,
  optimizer: 'adam',
  lr: 0.002,
  batchSize: 128,
  sampler: 'ddim',
  sampleSteps: 20,
  seed: 13,
};

export const DIFFUSION_LAB_LIMITS = { batchSize: [16, 1024], sampleSteps: [1, 200] } as const;

/** Training loss is averaged and recorded every this many iterations. */
export const DIFFUSION_RECORD_EVERY = 10;
/** The sample-quality distance is measured every this many iterations. */
export const DIFFUSION_EVALUATE_EVERY = 100;
/** Points in the fixed noise set (and in the reference set). */
export const DIFFUSION_EVAL_POINTS = 300;
/** Points whose whole path is recorded for the trajectory plot: all of the fixed noise set. */
export const DIFFUSION_TRACKED = DIFFUSION_EVAL_POINTS;

export interface DiffusionStats {
  iteration: number;
  loss: number;
  /** Sliced Wasserstein distance of the samples to the data, or null if not measured here. */
  distance: number | null;
}

export function validateDiffusionLab(c: DiffusionLabConfig): void {
  validateDenoiser(c);
  makeSchedule(c.schedule, c.steps); // validates steps and kind
  if (!(GAN_DATASETS as readonly string[]).includes(c.dataset))
    throw new RangeError('Unknown dataset');
  if (!(SAMPLERS as readonly string[]).includes(c.sampler)) throw new RangeError('Unknown sampler');
  if (!(OPTIMIZER_NAMES as readonly string[]).includes(c.optimizer))
    throw new RangeError('Unknown optimizer');
  if (!(c.lr > 0)) throw new RangeError('Learning rate must be positive');
  const [bLo, bHi] = DIFFUSION_LAB_LIMITS.batchSize;
  if (!Number.isInteger(c.batchSize) || c.batchSize < bLo || c.batchSize > bHi)
    throw new RangeError(`Batch size must be an integer in [${bLo}, ${bHi}]`);
  const [sLo, sHi] = DIFFUSION_LAB_LIMITS.sampleSteps;
  if (!Number.isInteger(c.sampleSteps) || c.sampleSteps < sLo || c.sampleSteps > sHi)
    throw new RangeError(`Sampling steps must be an integer in [${sLo}, ${sHi}]`);
}

const stream = (seed: number, index: number) =>
  createRng((seed + Math.imul(index + 1, 0x9e3779b1)) >>> 0);

const dataKey = (c: DiffusionLabConfig) => JSON.stringify([c.dataset, c.seed]);
const netKey = (c: DiffusionLabConfig) =>
  JSON.stringify([c.dataset, c.steps, c.schedule, c.hidden, c.seed]);

/** A diffusion model, its data and its training loop behind one small API. */
export class DiffusionSession {
  private cfg: DiffusionLabConfig;
  private keys = { data: '', net: '' };
  schedule!: NoiseSchedule;
  net!: Denoiser;
  private optimizer!: Optimizer;
  private dataRng = createRng(0);
  private noiseRng = createRng(0);
  private sampleRng = createRng(0);
  /** Real points for display, the noise that corrupts them in the forward view, and the start of sampling. */
  reference!: Float64Array;
  private forwardNoise!: Float64Array;
  private startNoise!: Float64Array;
  iteration = 0;
  history: DiffusionStats[] = [];
  private acc = { loss: 0, count: 0 };

  constructor(config: DiffusionLabConfig) {
    validateDiffusionLab(config);
    this.cfg = { ...config };
    this.build(true);
  }

  get config(): Readonly<DiffusionLabConfig> {
    return this.cfg;
  }

  /** Hyper-parameters apply live; data, schedule or architecture changes rebuild. */
  update(patch: Partial<DiffusionLabConfig>): 'live' | 'rebuilt' {
    const next = { ...this.cfg, ...patch };
    validateDiffusionLab(next);
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

  /** Runs `count` training iterations; returns the statistics recorded meanwhile. */
  train(count: number): DiffusionStats[] {
    const recorded: DiffusionStats[] = [];
    const { batchSize, lr, dataset, steps } = this.cfg;
    for (let n = 0; n < count; n++) {
      const x0 = sampleReal(dataset, batchSize, this.dataRng);
      const t = Float64Array.from(
        { length: batchSize },
        () => 1 + Math.floor(this.noiseRng.next() * steps),
      );
      const eps = Float64Array.from({ length: batchSize * 2 }, () => this.noiseRng.normal());
      const loss = this.net.gradients(this.schedule, x0, t, eps);
      this.optimizer.step(this.net.parameters, lr);
      this.iteration += 1;
      this.acc.loss += loss;
      this.acc.count += 1;
      if (this.iteration % DIFFUSION_RECORD_EVERY === 0) {
        const stats: DiffusionStats = {
          iteration: this.iteration,
          loss: this.acc.loss / this.acc.count,
          distance: this.iteration % DIFFUSION_EVALUATE_EVERY === 0 ? this.distance() : null,
        };
        this.acc = { loss: 0, count: 0 };
        this.history.push(stats);
        recorded.push(stats);
      }
    }
    return recorded;
  }

  /** Samples from the current model with the configured (or overridden) sampler. */
  sample(
    count: number,
    options: Partial<{ method: SamplerName; steps: number; track: number; seed: number }> = {},
  ): SampleResult {
    const rng = stream(options.seed ?? this.cfg.seed, 6);
    const init = Float64Array.from({ length: count * 2 }, () => rng.normal());
    return this.run(init, options);
  }

  /** Samples starting from the fixed noise set, with the paths of the first `DIFFUSION_TRACKED` points. */
  fixedSample(options: Partial<{ method: SamplerName; steps: number }> = {}): SampleResult {
    return this.run(this.startNoise, { ...options, track: DIFFUSION_TRACKED });
  }

  /** Sliced Wasserstein distance between fixed-noise samples (DDIM, 20 steps) and the data. */
  distance(): number {
    const { samples } = this.run(this.startNoise, { method: 'ddim', steps: 20 });
    return slicedWasserstein(samples, this.reference);
  }

  /** The reference points noised to time `t` with a fixed noise (`DIFFUSION_EVAL_POINTS × 2`). */
  forward(t: number): Float64Array {
    return noiseTo(this.schedule, this.reference, this.forwardNoise, t);
  }

  /** Predicted noise at time `t` for arbitrary points (`n × 2`). */
  predictNoise(points: Float64Array, t: number): Float64Array {
    return this.net.predict(points, t, this.cfg.steps);
  }

  /**
   * The model's score at time `t` over an `n × n` grid of `[-extent, extent]²` (row by row from
   * the top): `−ε̂ / √(1 − ᾱ_t)`, the direction that moves a point towards more likely data.
   */
  scoreField(t: number, n: number, extent: number): { points: Float64Array; score: Float64Array } {
    const points = new Float64Array(n * n * 2);
    for (let row = 0; row < n; row++) {
      for (let col = 0; col < n; col++) {
        const i = row * n + col;
        points[2 * i] = ((col + 0.5) / n) * 2 * extent - extent;
        points[2 * i + 1] = extent - ((row + 0.5) / n) * 2 * extent;
      }
    }
    const eps = this.predictNoise(points, t);
    const k = 1 / (this.schedule.sqrtOneMinusAlphaBar[t] as number);
    return { points, score: eps.map((v) => -v * k) };
  }

  private run(
    init: Float64Array,
    options: Partial<{ method: SamplerName; steps: number; track: number; seed: number }>,
  ): SampleResult {
    return sampleDiffusion((x, t) => this.predictNoise(x, t), this.schedule, init, {
      method: options.method ?? this.cfg.sampler,
      steps: options.steps ?? this.cfg.sampleSteps,
      rng: this.sampleRng,
      track: options.track ?? 0,
    });
  }

  private build(regenerate: boolean): void {
    const c = this.cfg;
    if (regenerate)
      this.reference = sampleReal(c.dataset, DIFFUSION_EVAL_POINTS, stream(c.seed, 0));
    this.schedule = makeSchedule(c.schedule, c.steps);
    this.net = new Denoiser(c, stream(c.seed, 1));
    this.optimizer = new Optimizer(c.optimizer);
    this.dataRng = stream(c.seed, 2);
    this.noiseRng = stream(c.seed, 3);
    this.sampleRng = stream(c.seed, 4);
    const noise = stream(c.seed, 5);
    this.forwardNoise = Float64Array.from({ length: DIFFUSION_EVAL_POINTS * 2 }, () =>
      noise.normal(),
    );
    this.startNoise = Float64Array.from({ length: DIFFUSION_EVAL_POINTS * 2 }, () =>
      noise.normal(),
    );
    this.iteration = 0;
    this.history = [];
    this.acc = { loss: 0, count: 0 };
    this.keys = { data: dataKey(c), net: netKey(c) };
  }
}
