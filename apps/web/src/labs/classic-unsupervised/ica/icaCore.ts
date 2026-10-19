import { FastICA, PCA, kurtosis, sourceRecovery, type IcaNonlinearity } from '@sim/classic';
import { Matrix, createRng, type Rng } from '@sim/core';

export type SourceKind = 'waves' | 'uniform' | 'laplace' | 'gaussian';

export interface IcaScenario {
  sources: SourceKind;
  /** Number of sources = number of microphones. */
  count: 2 | 3;
  /** Standard deviation of the noise added to every microphone (sources have unit variance). */
  noise: number;
  nonlinearity: IcaNonlinearity;
  seed: number;
  samples: number;
}

export const DEFAULT_ICA: IcaScenario = {
  sources: 'waves',
  count: 3,
  noise: 0,
  nonlinearity: 'logcosh',
  seed: 5,
  samples: 400,
};

export const SOURCE_LABELS: Record<SourceKind, string> = {
  waves: 'Sine, square, saw',
  uniform: 'Uniform noise',
  laplace: 'Sparse spikes',
  gaussian: 'Gaussian noise',
};

export const SOURCE_NOTES: Record<SourceKind, string> = {
  waves: 'Three clean periodic signals, like instruments in a room.',
  uniform: 'Independent flat noise: mixed together they form a tilted square or cube.',
  laplace: 'Mostly quiet with occasional bursts, like speech.',
  gaussian:
    'Bell-shaped noise. Mixed Gaussians are still Gaussian whichever way you rotate them, so there is nothing for ICA to find.',
};

function standardise(col: Float64Array): void {
  const n = col.length;
  let m = 0;
  for (const v of col) m += v / n;
  let s = 0;
  for (const v of col) s += (v - m) ** 2 / n;
  const sd = Math.sqrt(s) || 1;
  for (let i = 0; i < n; i++) col[i] = ((col[i] as number) - m) / sd;
}

function oneSource(kind: SourceKind, k: number, n: number, rng: Rng): Float64Array {
  const out = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const t = i / n;
    if (kind === 'waves') {
      if (k === 0) out[i] = Math.sin(2 * Math.PI * 4 * t);
      else if (k === 1) out[i] = Math.sign(Math.sin(2 * Math.PI * 6.3 * t + 0.4));
      else out[i] = ((9.7 * t) % 1) * 2 - 1;
    } else if (kind === 'uniform') out[i] = rng.uniform(-1, 1);
    else if (kind === 'laplace') {
      const u = rng.uniform(-0.4999, 0.4999);
      out[i] = -Math.sign(u) * Math.log(1 - 2 * Math.abs(u));
    } else out[i] = rng.normal();
  }
  standardise(out);
  return out;
}

export interface IcaRun {
  scenario: IcaScenario;
  /** `n × count` true sources, observations, and the signals recovered by ICA and PCA. */
  sources: Matrix;
  mixed: Matrix;
  ica: Matrix;
  pca: Matrix;
  /** For each true source, the best |correlation| with any recovered signal. */
  icaRecovery: number[];
  pcaRecovery: number[];
  kurtosis: number[];
  iterations: number;
  converged: boolean;
  model: FastICA;
}

export function runIca(s: IcaScenario): IcaRun {
  const rng = createRng(s.seed);
  const n = s.samples;
  const k = s.count;
  const cols = Array.from({ length: k }, (_, c) => oneSource(s.sources, c, n, rng));
  const sources = new Matrix(n, k);
  cols.forEach((col, c) => col.forEach((v, i) => sources.set(i, c, v)));
  // a random but well-conditioned mixing matrix: every microphone hears every source
  const a = new Matrix(k, k);
  for (let i = 0; i < k; i++)
    for (let j = 0; j < k; j++)
      a.set(i, j, rng.uniform(0.2, 1) * (rng.next() < 0.3 ? -1 : 1) + (i === j ? 0.6 : 0));
  const mixed = new Matrix(n, k);
  for (let t = 0; t < n; t++)
    for (let i = 0; i < k; i++) {
      let v = 0;
      for (let j = 0; j < k; j++) v += a.get(i, j) * sources.get(t, j);
      mixed.set(t, i, v + s.noise * rng.normal());
    }
  const model = new FastICA({
    components: k,
    nonlinearity: s.nonlinearity,
    seed: 1,
    maxIterations: 300,
  }).fit(mixed);
  const ica = model.transform(mixed);
  const pca = new PCA({ components: k }).fit(mixed).transform(mixed);
  return {
    scenario: s,
    sources,
    mixed,
    ica,
    pca,
    icaRecovery: sourceRecovery(sources, ica),
    pcaRecovery: sourceRecovery(sources, pca),
    kurtosis: Array.from({ length: k }, (_, c) =>
      kurtosis(Array.from({ length: n }, (_, i) => ica.get(i, c))),
    ),
    iterations: model.iterations,
    converged: model.converged,
    model,
  };
}

export const worst = (v: readonly number[]): number => Math.min(...v);
export const mean = (v: readonly number[]): number => v.reduce((a, b) => a + b, 0) / v.length;
