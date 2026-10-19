import type { Matrix } from './matrix';
import { mean } from './stats';

/** Scores for regression, classification and clustering, shared by every model. */

export const mse = (yTrue: ArrayLike<number>, yPred: ArrayLike<number>): number => {
  let s = 0;
  for (let i = 0; i < yTrue.length; i++) s += ((yTrue[i] as number) - (yPred[i] as number)) ** 2;
  return s / yTrue.length;
};

export const rmse = (yTrue: ArrayLike<number>, yPred: ArrayLike<number>): number =>
  Math.sqrt(mse(yTrue, yPred));

export const mae = (yTrue: ArrayLike<number>, yPred: ArrayLike<number>): number => {
  let s = 0;
  for (let i = 0; i < yTrue.length; i++) s += Math.abs((yTrue[i] as number) - (yPred[i] as number));
  return s / yTrue.length;
};

/** Coefficient of determination `R² = 1 − SS_res / SS_tot` (can be negative for a bad model). */
export function r2(yTrue: ArrayLike<number>, yPred: ArrayLike<number>): number {
  const m = mean(yTrue);
  let res = 0;
  let tot = 0;
  for (let i = 0; i < yTrue.length; i++) {
    res += ((yTrue[i] as number) - (yPred[i] as number)) ** 2;
    tot += ((yTrue[i] as number) - m) ** 2;
  }
  return tot === 0 ? (res === 0 ? 1 : 0) : 1 - res / tot;
}

export function accuracy(yTrue: ArrayLike<number>, yPred: ArrayLike<number>): number {
  let right = 0;
  for (let i = 0; i < yTrue.length; i++) if (yTrue[i] === yPred[i]) right += 1;
  return right / yTrue.length;
}

/** `k × k` counts: rows = true class, columns = predicted class. */
export function confusionMatrix(
  yTrue: ArrayLike<number>,
  yPred: ArrayLike<number>,
  k: number,
): number[][] {
  const m = Array.from({ length: k }, () => new Array<number>(k).fill(0));
  for (let i = 0; i < yTrue.length; i++) {
    const row = m[yTrue[i] as number];
    if (row) row[yPred[i] as number] = (row[yPred[i] as number] as number) + 1;
  }
  return m;
}

export interface ClassScores {
  precision: number;
  recall: number;
  f1: number;
  support: number;
}

/** Precision, recall and F1 of each class from a confusion matrix (0 where undefined). */
export function classScores(confusion: number[][]): ClassScores[] {
  return confusion.map((row, c) => {
    const tp = row[c] as number;
    const support = row.reduce((a, b) => a + b, 0);
    const predicted = confusion.reduce((a, r) => a + (r[c] as number), 0);
    const precision = predicted === 0 ? 0 : tp / predicted;
    const recall = support === 0 ? 0 : tp / support;
    return {
      precision,
      recall,
      f1: precision + recall === 0 ? 0 : (2 * precision * recall) / (precision + recall),
      support,
    };
  });
}

/** Mean cross-entropy of predicted class probabilities (`n × k`) against integer labels. */
export function logLoss(probs: Matrix, yTrue: ArrayLike<number>): number {
  let s = 0;
  for (let i = 0; i < probs.rows; i++)
    s -= Math.log(Math.max(1e-15, probs.get(i, yTrue[i] as number)));
  return s / probs.rows;
}

/** Receiver operating characteristic of a binary scorer: points from (0, 0) to (1, 1) as the threshold falls. */
export function rocCurve(
  scores: ArrayLike<number>,
  yTrue: ArrayLike<number>,
): { fpr: number[]; tpr: number[]; thresholds: number[] } {
  const n = scores.length;
  const order = Array.from({ length: n }, (_, i) => i).sort(
    (a, b) => (scores[b] as number) - (scores[a] as number),
  );
  let positives = 0;
  for (let i = 0; i < n; i++) positives += yTrue[i] === 1 ? 1 : 0;
  const negatives = n - positives;
  const fpr = [0];
  const tpr = [0];
  const thresholds = [Infinity];
  let tp = 0;
  let fp = 0;
  for (let k = 0; k < n; k++) {
    const i = order[k] as number;
    if (yTrue[i] === 1) tp += 1;
    else fp += 1;
    const last = k === n - 1 || scores[order[k + 1] as number] !== scores[i];
    if (last) {
      fpr.push(negatives === 0 ? 0 : fp / negatives);
      tpr.push(positives === 0 ? 0 : tp / positives);
      thresholds.push(scores[i] as number);
    }
  }
  return { fpr, tpr, thresholds };
}

/** Area under a piecewise-linear curve (trapezoid rule). */
export function auc(x: number[], y: number[]): number {
  let a = 0;
  for (let i = 1; i < x.length; i++)
    a +=
      (((x[i] as number) - (x[i - 1] as number)) * ((y[i] as number) + (y[i - 1] as number))) / 2;
  return a;
}

/** Mean silhouette coefficient of a clustering (−1 … 1, higher is better). Noise labelled −1 is ignored. */
export function silhouette(x: Matrix, labels: ArrayLike<number>): number {
  const n = x.rows;
  const ks = [...new Set(Array.from(labels).filter((l) => l >= 0))];
  if (ks.length < 2) return 0;
  const dist = (i: number, j: number) => {
    let s = 0;
    for (let c = 0; c < x.cols; c++) s += (x.get(i, c) - x.get(j, c)) ** 2;
    return Math.sqrt(s);
  };
  let total = 0;
  let counted = 0;
  for (let i = 0; i < n; i++) {
    const li = labels[i] as number;
    if (li < 0) continue;
    const sums = new Map<number, [number, number]>();
    for (let j = 0; j < n; j++) {
      const lj = labels[j] as number;
      if (j === i || lj < 0) continue;
      const e = sums.get(lj) ?? [0, 0];
      e[0] += dist(i, j);
      e[1] += 1;
      sums.set(lj, e);
    }
    const own = sums.get(li);
    if (!own || own[1] === 0) continue;
    const a = own[0] / own[1];
    let b = Infinity;
    for (const [l, [s, c]] of sums) if (l !== li && c > 0) b = Math.min(b, s / c);
    total += Math.max(a, b) === 0 ? 0 : (b - a) / Math.max(a, b);
    counted += 1;
  }
  return counted === 0 ? 0 : total / counted;
}

/** Adjusted Rand index between two labelings (1 = identical partitions, ~0 = random). */
export function adjustedRandIndex(a: ArrayLike<number>, b: ArrayLike<number>): number {
  const n = a.length;
  const table = new Map<string, number>();
  const rows = new Map<number, number>();
  const cols = new Map<number, number>();
  for (let i = 0; i < n; i++) {
    const key = `${a[i]}|${b[i]}`;
    table.set(key, (table.get(key) ?? 0) + 1);
    rows.set(a[i] as number, (rows.get(a[i] as number) ?? 0) + 1);
    cols.set(b[i] as number, (cols.get(b[i] as number) ?? 0) + 1);
  }
  const c2 = (v: number) => (v * (v - 1)) / 2;
  let sumIj = 0;
  for (const v of table.values()) sumIj += c2(v);
  let sumA = 0;
  for (const v of rows.values()) sumA += c2(v);
  let sumB = 0;
  for (const v of cols.values()) sumB += c2(v);
  const expected = (sumA * sumB) / c2(n);
  const max = (sumA + sumB) / 2;
  return max === expected ? 1 : (sumIj - expected) / (max - expected);
}
