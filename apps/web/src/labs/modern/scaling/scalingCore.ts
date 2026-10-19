import {
  Matrix,
  choleskySolve,
  cholesky,
  createRng,
  fitPowerLaw,
  matMul,
  transpose,
  type PowerLaw,
} from '@sim/core';

/**
 * Scaling laws in a model small enough to solve exactly. The "model" is a ridge regression on N random cosine
 * features of five input numbers; the "data" is D noisy examples of a target made of 150 waves whose strengths fall
 * off as 1/j. Because the target has no end, more features and more data keep helping, and the test error follows a
 * power law in each, down to the noise that no model can remove.
 */
export interface ScalingScenario {
  /** Standard deviation of the noise on the training labels. */
  noise: number;
  seed: number;
}

export const DEFAULT_SCALING: ScalingScenario = { noise: 0.1, seed: 6 };

export const FEATURES = [8, 16, 32, 64, 128, 256];
export const DATA = [64, 128, 256, 512, 1024, 2048, 4096];
const DIM = 5;
const WAVES = 150;
const TEST = 2000;
const RIDGE = 3e-2;

export interface ScalingRun {
  scenario: ScalingScenario;
  /** Test loss (mean squared error against the noisy labels' expectation) for every feature count (rows) and data size (columns). */
  loss: number[][];
  floor: number;
  /** Laws fitted with the floor known: against data for the biggest model, against features for the most data. */
  dataLaw: PowerLaw;
  featureLaw: PowerLaw;
  /** For each budget (features × examples), the cell of the grid with the lowest loss that fits within it. */
  frontier: { budget: number; features: number; data: number; loss: number }[];
}

export const BUDGETS = [4096, 16384, 65536, 262144, 1048576];

const cache = new Map<string, ScalingRun>();

export function runScaling(s: ScalingScenario): ScalingRun {
  const key = JSON.stringify(s);
  const hit = cache.get(key);
  if (hit) return hit;
  const rng = createRng(s.seed);
  const normal = () => rng.normal();

  const waves = Array.from({ length: WAVES }, (_, j) => ({
    w: Float64Array.from({ length: DIM }, normal),
    b: rng.next() * 2 * Math.PI,
    amp: 1 / (j + 1),
  }));
  const target = (x: ArrayLike<number>, row: number) => {
    let v = 0;
    for (const { w, b, amp } of waves) {
      let dot = b;
      for (let k = 0; k < DIM; k++) dot += (w[k] as number) * (x[row * DIM + k] as number);
      v += amp * Math.sin(dot);
    }
    return v;
  };
  const points = (n: number) => {
    const x = Float64Array.from({ length: n * DIM }, normal);
    return { x, y: Float64Array.from({ length: n }, (_, i) => target(x, i)) };
  };

  const maxFeatures = Math.max(...FEATURES);
  const projection = Float64Array.from({ length: maxFeatures * DIM }, normal);
  const phase = Float64Array.from({ length: maxFeatures }, () => rng.next() * 2 * Math.PI);
  const features = (x: Float64Array, n: number, count: number) => {
    const out = new Matrix(n, count);
    for (let i = 0; i < n; i++)
      for (let f = 0; f < count; f++) {
        let dot = phase[f] as number;
        for (let k = 0; k < DIM; k++)
          dot += (projection[f * DIM + k] as number) * (x[i * DIM + k] as number);
        out.data[i * count + f] = Math.cos(dot) * Math.SQRT2;
      }
    return out;
  };

  const train = points(Math.max(...DATA));
  const noisy = Float64Array.from(train.y, (v) => v + s.noise * normal());
  const test = points(TEST);

  const fitCell = (count: number, n: number): number => {
    const phi = features(train.x.subarray(0, n * DIM), n, count);
    const gram = matMul(transpose(phi), phi);
    for (let i = 0; i < count; i++)
      gram.data[i * count + i] = (gram.data[i * count + i] as number) + RIDGE * n;
    const rhs = new Matrix(count, 1);
    for (let f = 0; f < count; f++) {
      let sum = 0;
      for (let i = 0; i < n; i++) sum += (phi.data[i * count + f] as number) * (noisy[i] as number);
      rhs.data[f] = sum;
    }
    const weights = choleskySolve(cholesky(gram), rhs);
    const phiTest = features(test.x, TEST, count);
    let err = 0;
    for (let i = 0; i < TEST; i++) {
      let pred = 0;
      for (let f = 0; f < count; f++)
        pred += (phiTest.data[i * count + f] as number) * (weights.data[f] as number);
      err += (pred - (test.y[i] as number)) ** 2;
    }
    return err / TEST + s.noise ** 2;
  };

  const loss = FEATURES.map((count) => DATA.map((n) => fitCell(count, n)));
  const floor = s.noise ** 2;
  const frontier = BUDGETS.map((budget) => {
    let best = { budget, features: 0, data: 0, loss: Infinity };
    FEATURES.forEach((features, i) =>
      DATA.forEach((data, j) => {
        const l = (loss[i] as number[])[j] as number;
        if (features * data <= budget && l < best.loss) best = { budget, features, data, loss: l };
      }),
    );
    return best;
  });
  const run: ScalingRun = {
    scenario: s,
    loss,
    floor,
    dataLaw: fitPowerLaw(DATA, loss[loss.length - 1] as number[], floor),
    frontier,
    featureLaw: fitPowerLaw(
      FEATURES,
      loss.map((row) => row[row.length - 1] as number),
      floor,
    ),
  };
  cache.set(key, run);
  return run;
}
