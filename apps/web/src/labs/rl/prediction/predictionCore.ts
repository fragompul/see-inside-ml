import {
  PREDICTION_METHODS,
  averageError,
  learnValues,
  trueValues,
  type PredictionMethod,
} from '@sim/deep';

export interface PredictionScenario {
  states: number;
  alpha: number;
  n: number;
  lambda: number;
  episodes: number;
  seed: number;
}

export const DEFAULT_PREDICTION: PredictionScenario = {
  states: 19,
  alpha: 0.1,
  n: 4,
  lambda: 0.8,
  episodes: 40,
  seed: 5,
};

export const METHOD_LABELS: Record<PredictionMethod, string> = {
  mc: 'Monte Carlo',
  td: 'TD(0)',
  nstep: 'n-step TD',
  lambda: 'TD(λ)',
};

export const RUNS = 60;

export interface PredictionRun {
  scenario: PredictionScenario;
  truth: number[];
  /** One learned value function per method (a single run) and the average error curve over many runs. */
  values: Record<PredictionMethod, number[]>;
  curves: Record<PredictionMethod, number[]>;
  /** Mean error over the whole run (area under the curve): lower is better. */
  area: Record<PredictionMethod, number>;
}

export function runPrediction(s: PredictionScenario): PredictionRun {
  const values = {} as Record<PredictionMethod, number[]>;
  const curves = {} as Record<PredictionMethod, number[]>;
  const area = {} as Record<PredictionMethod, number>;
  for (const method of PREDICTION_METHODS) {
    const o = { ...s, method };
    values[method] = learnValues(o).values;
    curves[method] = averageError(o, RUNS);
    area[method] = curves[method].reduce((a, b) => a + b, 0) / curves[method].length;
  }
  return { scenario: s, truth: trueValues(s.states), values, curves, area };
}
