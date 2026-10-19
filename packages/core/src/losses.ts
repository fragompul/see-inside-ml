import type { Matrix } from './matrix';
import { sub } from './matrix';

export type Problem = 'classification' | 'regression';
export type LossType = 'cce' | 'mse' | 'mae' | 'huber';

export const HUBER_DELTA = 1;
const PROB_EPS = 1e-15;

export const REGRESSION_LOSSES = ['mse', 'mae', 'huber'] as const;

/** Default loss for a problem type. Classification always uses categorical cross-entropy. */
export function defaultLoss(problem: Problem): LossType {
  return problem === 'classification' ? 'cce' : 'mse';
}

export function isValidLoss(problem: Problem, loss: LossType): boolean {
  return problem === 'classification' ? loss === 'cce' : loss !== 'cce';
}

/** Mean loss over the batch (no regularisation term). */
export function lossValue(loss: LossType, out: Matrix, y: Matrix): number {
  if (out.rows !== y.rows || out.cols !== y.cols) throw new RangeError('loss: shape mismatch');
  const n = out.data.length;
  switch (loss) {
    case 'cce': {
      let total = 0;
      for (let i = 0; i < out.data.length; i++) {
        const p = Math.min(1 - PROB_EPS, Math.max(PROB_EPS, out.data[i] as number));
        total += (y.data[i] as number) * Math.log(p);
      }
      return -total / out.rows;
    }
    case 'mse': {
      let total = 0;
      for (let i = 0; i < n; i++) total += ((out.data[i] as number) - (y.data[i] as number)) ** 2;
      return (0.5 * total) / n;
    }
    case 'mae': {
      let total = 0;
      for (let i = 0; i < n; i++)
        total += Math.abs((out.data[i] as number) - (y.data[i] as number));
      return total / n;
    }
    case 'huber': {
      let total = 0;
      for (let i = 0; i < n; i++) {
        const d = Math.abs((out.data[i] as number) - (y.data[i] as number));
        total += d <= HUBER_DELTA ? 0.5 * d * d : HUBER_DELTA * (d - 0.5 * HUBER_DELTA);
      }
      return total / n;
    }
  }
}

/**
 * dL/dZ at the output layer (loss derivative already combined with the output activation):
 * softmax + cross-entropy gives ŷ − y; linear outputs give the plain error derivative.
 */
export function outputGradient(loss: LossType, out: Matrix, y: Matrix): Matrix {
  const d = sub(out, y);
  switch (loss) {
    case 'cce':
    case 'mse':
      return d;
    case 'mae':
      return d.map(Math.sign);
    case 'huber':
      return d.map((x) => (Math.abs(x) > HUBER_DELTA ? HUBER_DELTA * Math.sign(x) : x));
  }
}

export const lossInfo: Record<LossType, { label: string; formula: string }> = {
  cce: { label: 'Cross-entropy', formula: String.raw`-\tfrac1m\sum_i\sum_k y_{ik}\log\hat y_{ik}` },
  mse: { label: 'Mean squared error', formula: String.raw`\tfrac12\,\overline{(\hat y-y)^2}` },
  mae: { label: 'Mean absolute error', formula: String.raw`\overline{|\hat y-y|}` },
  huber: {
    label: 'Huber',
    formula: String.raw`\begin{cases}\tfrac12 d^2 & |d|\le1\\ |d|-\tfrac12 & |d|>1\end{cases}`,
  },
};
