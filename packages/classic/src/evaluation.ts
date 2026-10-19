import { LogisticRegression } from './logistic';
import { Matrix, createRng } from '@sim/core';

/**
 * Binary-classification evaluation beyond accuracy: metrics at a threshold, the precision-recall
 * curve, calibration (do predicted probabilities mean what they say?), probability recalibration
 * (Platt scaling), class rebalancing (oversampling and SMOTE) and cost-based thresholds.
 */
export interface ThresholdMetrics {
  tp: number;
  fp: number;
  tn: number;
  fn: number;
  accuracy: number;
  precision: number;
  recall: number;
  /** True negative rate. */
  specificity: number;
  f1: number;
  /** Matthews correlation coefficient: a single balanced score, 0 for chance and 1 for perfect. */
  mcc: number;
}

export function metricsAt(
  scores: ArrayLike<number>,
  y: ArrayLike<number>,
  threshold: number,
): ThresholdMetrics {
  let tp = 0;
  let fp = 0;
  let tn = 0;
  let fn = 0;
  for (let i = 0; i < scores.length; i++) {
    const p = (scores[i] as number) >= threshold;
    if (y[i] === 1) {
      if (p) tp++;
      else fn++;
    } else if (p) fp++;
    else tn++;
  }
  const div = (a: number, b: number) => (b === 0 ? 0 : a / b);
  const precision = div(tp, tp + fp);
  const recall = div(tp, tp + fn);
  const mcc = div(tp * tn - fp * fn, Math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)));
  return {
    tp,
    fp,
    tn,
    fn,
    accuracy: div(tp + tn, scores.length),
    precision,
    recall,
    specificity: div(tn, tn + fp),
    f1: div(2 * precision * recall, precision + recall),
    mcc,
  };
}

export interface PrCurve {
  precision: number[];
  recall: number[];
  thresholds: number[];
  /** Average precision: sum over thresholds of (recall step) x precision, as in scikit-learn. */
  averagePrecision: number;
}

/** Precision-recall curve (thresholds from high to low; each distinct score once). */
export function prCurve(scores: ArrayLike<number>, y: ArrayLike<number>): PrCurve {
  const n = scores.length;
  const order = Array.from({ length: n }, (_, i) => i).sort(
    (a, b) => (scores[b] as number) - (scores[a] as number),
  );
  let positives = 0;
  for (let i = 0; i < n; i++) positives += y[i] === 1 ? 1 : 0;
  const precision: number[] = [];
  const recall: number[] = [];
  const thresholds: number[] = [];
  let tp = 0;
  let fp = 0;
  for (let k = 0; k < n; k++) {
    const i = order[k] as number;
    if (y[i] === 1) tp++;
    else fp++;
    if (k === n - 1 || scores[order[k + 1] as number] !== scores[i]) {
      precision.push(tp / (tp + fp));
      recall.push(positives === 0 ? 0 : tp / positives);
      thresholds.push(scores[i] as number);
    }
  }
  let ap = 0;
  let prev = 0;
  for (let k = 0; k < precision.length; k++) {
    ap += ((recall[k] as number) - prev) * (precision[k] as number);
    prev = recall[k] as number;
  }
  return { precision, recall, thresholds, averagePrecision: ap };
}

export interface CalibrationBin {
  /** Mean predicted probability of the points in the bin. */
  predicted: number;
  /** Share of them that really are positive. */
  observed: number;
  count: number;
}

/** Reliability diagram data: equal-width bins on [0, 1] (empty bins are skipped). */
export function calibrationBins(
  probs: ArrayLike<number>,
  y: ArrayLike<number>,
  bins = 10,
): CalibrationBin[] {
  const sumP = new Array<number>(bins).fill(0);
  const sumY = new Array<number>(bins).fill(0);
  const count = new Array<number>(bins).fill(0);
  for (let i = 0; i < probs.length; i++) {
    const b = Math.min(bins - 1, Math.floor((probs[i] as number) * bins));
    sumP[b] = (sumP[b] as number) + (probs[i] as number);
    sumY[b] = (sumY[b] as number) + (y[i] as number);
    count[b] = (count[b] as number) + 1;
  }
  const out: CalibrationBin[] = [];
  for (let b = 0; b < bins; b++)
    if ((count[b] as number) > 0)
      out.push({
        predicted: (sumP[b] as number) / (count[b] as number),
        observed: (sumY[b] as number) / (count[b] as number),
        count: count[b] as number,
      });
  return out;
}

/** Brier score: mean squared difference between the predicted probability and the 0/1 outcome. */
export function brierScore(probs: ArrayLike<number>, y: ArrayLike<number>): number {
  let s = 0;
  for (let i = 0; i < probs.length; i++) s += ((probs[i] as number) - (y[i] as number)) ** 2;
  return s / probs.length;
}

/** Expected calibration error: bin gaps weighted by how many points each bin holds. */
export function expectedCalibrationError(
  probs: ArrayLike<number>,
  y: ArrayLike<number>,
  bins = 10,
): number {
  return calibrationBins(probs, y, bins).reduce(
    (s, b) => s + (b.count / probs.length) * Math.abs(b.predicted - b.observed),
    0,
  );
}

/**
 * Platt scaling: fit a logistic regression of the outcome on the logit of the model's score, using
 * data the model was not trained on. Returns a function mapping raw scores to calibrated probabilities.
 */
export function plattScaler(
  scores: ArrayLike<number>,
  y: ArrayLike<number>,
): (s: number) => number {
  const eps = 1e-4;
  const logit = (p: number) =>
    Math.log(Math.min(1 - eps, Math.max(eps, p)) / (1 - Math.min(1 - eps, Math.max(eps, p))));
  const x = new Matrix(
    scores.length,
    1,
    Float64Array.from(scores, (p) => logit(p)),
  );
  const lr = new LogisticRegression({ C: 1e6 }).fit(x, y);
  const w = lr.weights.get(0, 0);
  const b = lr.intercepts[0] as number;
  return (s) => 1 / (1 + Math.exp(-(w * logit(s) + b)));
}

/** Duplicates randomly chosen minority rows until both classes have the same size. */
export function oversample(x: Matrix, y: ArrayLike<number>, seed = 1): { x: Matrix; y: number[] } {
  const rng = createRng(seed);
  const idx1: number[] = [];
  const idx0: number[] = [];
  for (let i = 0; i < x.rows; i++) (y[i] === 1 ? idx1 : idx0).push(i);
  const [minor, major] = idx1.length <= idx0.length ? [idx1, idx0] : [idx0, idx1];
  const extra = Array.from(
    { length: major.length - minor.length },
    () => minor[Math.floor(rng.next() * minor.length)] as number,
  );
  const all = [...Array.from({ length: x.rows }, (_, i) => i), ...extra];
  const out = new Matrix(all.length, x.cols);
  all.forEach((r, i) => out.data.set(x.data.subarray(r * x.cols, (r + 1) * x.cols), i * x.cols));
  return { x: out, y: all.map((r) => y[r] as number) };
}

/**
 * SMOTE: synthesise new minority points on the segments between a minority point and one of its `k`
 * nearest minority neighbours, until the classes are balanced.
 */
export function smote(
  x: Matrix,
  y: ArrayLike<number>,
  k = 5,
  seed = 1,
): { x: Matrix; y: number[] } {
  const rng = createRng(seed);
  const idx1: number[] = [];
  const idx0: number[] = [];
  for (let i = 0; i < x.rows; i++) (y[i] === 1 ? idx1 : idx0).push(i);
  const minorityLabel = idx1.length <= idx0.length ? 1 : 0;
  const minor = minorityLabel === 1 ? idx1 : idx0;
  const need = Math.abs(idx0.length - idx1.length);
  if (minor.length < 2 || need === 0) return oversample(x, y, seed);
  const d = (a: number, b: number) => {
    let s = 0;
    for (let c = 0; c < x.cols; c++) s += (x.get(a, c) - x.get(b, c)) ** 2;
    return s;
  };
  const neighbours = minor.map((a) =>
    minor
      .filter((b) => b !== a)
      .sort((p, q) => d(a, p) - d(a, q))
      .slice(0, k),
  );
  const synthetic: number[][] = [];
  for (let s = 0; s < need; s++) {
    const m = Math.floor(rng.next() * minor.length);
    const a = minor[m] as number;
    const nb = neighbours[m] as number[];
    const b = nb[Math.floor(rng.next() * nb.length)] as number;
    const t = rng.next();
    synthetic.push(
      Array.from({ length: x.cols }, (_, c) => x.get(a, c) + t * (x.get(b, c) - x.get(a, c))),
    );
  }
  const out = new Matrix(x.rows + need, x.cols);
  out.data.set(x.data);
  synthetic.forEach((row, i) => out.data.set(row, (x.rows + i) * x.cols));
  return {
    x: out,
    y: [
      ...Array.from({ length: x.rows }, (_, i) => y[i] as number),
      ...new Array<number>(need).fill(minorityLabel),
    ],
  };
}

/** Total cost at a threshold, with separate prices for the two kinds of mistake. */
export function costAt(
  scores: ArrayLike<number>,
  y: ArrayLike<number>,
  threshold: number,
  costFalseNegative: number,
  costFalsePositive: number,
): number {
  const m = metricsAt(scores, y, threshold);
  return m.fn * costFalseNegative + m.fp * costFalsePositive;
}

/** The threshold (among the scores themselves) with the smallest total cost. */
export function bestThreshold(
  scores: ArrayLike<number>,
  y: ArrayLike<number>,
  costFalseNegative: number,
  costFalsePositive: number,
): { threshold: number; cost: number } {
  const cand = [...new Set(Array.from(scores))].sort((a, b) => a - b);
  let best = { threshold: 0.5, cost: Infinity };
  for (const t of [0, ...cand, 1.0000001]) {
    const c = costAt(scores, y, t, costFalseNegative, costFalsePositive);
    if (c < best.cost) best = { threshold: t, cost: c };
  }
  return best;
}
