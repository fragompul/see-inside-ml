import { NMF, PCA, nmfError, type NmfLoss } from '@sim/classic';
import { Matrix, createRng } from '@sim/core';
import { generateDigits } from '@sim/deep';

export interface NmfScenario {
  components: number;
  loss: NmfLoss;
  iterations: number;
  /** Digits drawn: all ten classes, or a small subset that is easier to read. */
  digits: 'all' | 'few';
  perClass: number;
  seed: number;
}

export const DEFAULT_NMF: NmfScenario = {
  components: 10,
  loss: 'frobenius',
  iterations: 100,
  digits: 'all',
  perClass: 12,
  seed: 3,
};

export interface NmfRun {
  scenario: NmfScenario;
  data: Matrix;
  labels: Int32Array;
  model: NMF;
  pca: PCA;
  /** ‖V − WH‖ / ‖V‖ for NMF and for PCA with the same number of components. */
  nmfRelError: number;
  pcaRelError: number;
  /** Hoyer sparsity (0 = all pixels equal, 1 = a single pixel) averaged over the parts. */
  nmfSparsity: number;
  pcaSparsity: number;
  /** Share of PCA component entries that are negative (NMF has none by construction). */
  pcaNegative: number;
}

const FEW = [0, 1, 4, 7];

export function digitsMatrix(s: Pick<NmfScenario, 'digits' | 'perClass' | 'seed'>): {
  x: Matrix;
  labels: Int32Array;
} {
  const set = generateDigits(s.perClass, createRng(s.seed));
  const keep: number[] = [];
  for (let i = 0; i < set.count; i++)
    if (s.digits === 'all' || FEW.includes(set.labels[i] as number)) keep.push(i);
  const x = new Matrix(keep.length, 256);
  keep.forEach((src, i) => {
    for (let j = 0; j < 256; j++) x.set(i, j, Math.max(0, set.images[src * 256 + j] as number));
  });
  return { x, labels: Int32Array.from(keep, (i) => set.labels[i] as number) };
}

export function hoyer(row: ArrayLike<number>): number {
  const n = row.length;
  let l1 = 0;
  let l2 = 0;
  for (let i = 0; i < n; i++) {
    const a = Math.abs(row[i] as number);
    l1 += a;
    l2 += a * a;
  }
  if (l2 === 0) return 0;
  return (Math.sqrt(n) - l1 / Math.sqrt(l2)) / (Math.sqrt(n) - 1);
}

export function runNmf(s: NmfScenario): NmfRun {
  const { x, labels } = digitsMatrix(s);
  const model = new NMF({
    components: s.components,
    loss: s.loss,
    iterations: s.iterations,
    seed: s.seed,
  }).fit(x);
  const pca = new PCA({ components: s.components }).fit(x);
  let norm = 0;
  for (const v of x.data) norm += v * v;
  const nmfRel = Math.sqrt(nmfError(x, model.w, model.h, 'frobenius') / norm);
  const back = pca.inverseTransform(pca.transform(x));
  let pe = 0;
  for (let i = 0; i < x.data.length; i++)
    pe += ((x.data[i] as number) - (back.data[i] as number)) ** 2;
  const rows = (m: Matrix) =>
    Array.from({ length: m.rows }, (_, r) => m.data.slice(r * m.cols, (r + 1) * m.cols));
  const avg = (m: Matrix) => rows(m).reduce((a, r) => a + hoyer(r), 0) / m.rows;
  let neg = 0;
  for (const v of pca.components.data) if (v < -1e-9) neg++;
  return {
    scenario: s,
    data: x,
    labels,
    model,
    pca,
    nmfRelError: nmfRel,
    pcaRelError: Math.sqrt(pe / norm),
    nmfSparsity: avg(model.h),
    pcaSparsity: avg(pca.components),
    pcaNegative: neg / pca.components.data.length,
  };
}

/** Correlation of the first learned part with the average digit (1 = the part IS the average picture). */
export function partMeanCorrelation(run: NmfRun): number {
  const m = new Float64Array(256);
  for (let i = 0; i < run.data.rows; i++)
    for (let j = 0; j < 256; j++) m[j] = (m[j] as number) + run.data.get(i, j) / run.data.rows;
  const h = run.model.h.data.slice(0, 256);
  const mean = (a: ArrayLike<number>) => Array.from(a).reduce((s, v) => s + v, 0) / a.length;
  const ma = mean(h);
  const mb = mean(m);
  let num = 0;
  let da = 0;
  let db = 0;
  for (let j = 0; j < 256; j++) {
    num += ((h[j] as number) - ma) * ((m[j] as number) - mb);
    da += ((h[j] as number) - ma) ** 2;
    db += ((m[j] as number) - mb) ** 2;
  }
  return num / Math.sqrt(da * db || 1);
}
