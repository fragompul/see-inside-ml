import { AdditiveModel, LinearRegression, Mars } from '@sim/classic';
import { Matrix, createRng } from '@sim/core';

/**
 * Two inputs, two truths. In the first the output is a sum of a function of each input (what an additive model
 * assumes); in the second it is a product of hinges, an interaction that no sum of one-input functions can express.
 */
export type Truth = 'additive' | 'interaction';

export const TRUTH_LABEL: Record<Truth, string> = {
  additive: 'sin(2·x₁) + x₂² − 1',
  interaction: '6 · max(0, x₁ − 0.4) · max(0, x₂ − 0.4)',
};

const f1 = (v: number) => Math.sin(2 * v);
const f2 = (v: number) => v * v - 1;

const truthOf = (kind: Truth, x1: number, x2: number): number =>
  kind === 'additive' ? f1(x1) + f2(x2) : 6 * Math.max(0, x1 - 0.4) * Math.max(0, x2 - 0.4);

export interface TwoInputsResult {
  truth: Truth;
  /** Held-out R² of each model. */
  r2: { linear: number; additive: number; marsAdditive: number; marsInteracting: number };
  /** For the additive truth: the recovered and true shape of each input on a grid. */
  grid: number[];
  shapes: { recovered: number[][]; truth: number[][] };
}

const r2Of = (pred: Float64Array, y: number[]): number => {
  const mean = y.reduce((a, v) => a + v, 0) / y.length;
  let sse = 0;
  let tot = 0;
  y.forEach((v, i) => {
    sse += (v - (pred[i] as number)) ** 2;
    tot += (v - mean) ** 2;
  });
  return 1 - sse / tot;
};

export function twoInputs(truth: Truth): TwoInputsResult {
  const rng = createRng(truth === 'additive' ? 8 : 5);
  const range = truth === 'additive' ? [-2, 2] : [0, 1];
  const sample = (n: number) => {
    const x = new Matrix(
      n,
      2,
      Float64Array.from({ length: 2 * n }, () =>
        rng.uniform(range[0] as number, range[1] as number),
      ),
    );
    const y = Array.from(
      { length: n },
      (_, i) =>
        truthOf(truth, x.data[2 * i] as number, x.data[2 * i + 1] as number) + 0.2 * rng.normal(),
    );
    return { x, y };
  };
  const train = sample(400);
  const test = sample(400);
  const additive = new AdditiveModel({ bumps: 12, lambda: 1, iterations: 100 }).fit(
    train.x,
    train.y,
  );
  const grid = Array.from(
    { length: 41 },
    (_, i) => (range[0] as number) + (i / 40) * ((range[1] as number) - (range[0] as number)),
  );
  return {
    truth,
    r2: {
      linear: r2Of(new LinearRegression().fit(train.x, train.y).predict(test.x), test.y),
      additive: r2Of(additive.predict(test.x), test.y),
      marsAdditive: r2Of(
        new Mars({ maxTerms: 11, degree: 1, knots: 10, penalty: 2 })
          .fit(train.x, train.y)
          .predict(test.x),
        test.y,
      ),
      marsInteracting: r2Of(
        new Mars({ maxTerms: 11, degree: 2, knots: 10, penalty: 3 })
          .fit(train.x, train.y)
          .predict(test.x),
        test.y,
      ),
    },
    grid,
    shapes: {
      recovered: [Array.from(additive.shape(0, grid)), Array.from(additive.shape(1, grid))],
      truth: [grid.map(f1), grid.map(f2)],
    },
  };
}
