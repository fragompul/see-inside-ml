import { Matrix, createRng } from '@sim/core';

/**
 * Self-organising map (Kohonen): a grid of units, each holding a weight vector in data space. For
 * every sample the *best-matching unit* and its grid neighbours are pulled towards it, by less and
 * less as training goes on. The grid ends up a topology-preserving sheet laid over the data:
 * neighbouring units hold similar prototypes, so the map is a 2-D picture of high-dimensional data.
 */
export interface SomOptions {
  rows: number;
  cols: number;
  epochs?: number;
  /** Learning rate at the start (decays linearly to ~0). */
  learningRate?: number;
  /** Neighbourhood radius at the start, in grid cells (decays to ~0.5). */
  radius?: number;
  seed?: number;
}

export interface SomState {
  epoch: number;
  /** `rows·cols × d` prototype vectors, unit `r·cols + c`. */
  weights: Matrix;
  quantizationError: number;
}

const bmu = (w: Matrix, x: Matrix, i: number): { best: number; second: number; dist: number } => {
  let best = 0;
  let second = 1;
  let bd = Infinity;
  let sd = Infinity;
  for (let u = 0; u < w.rows; u++) {
    let s = 0;
    for (let j = 0; j < x.cols; j++) s += (x.get(i, j) - w.get(u, j)) ** 2;
    if (s < bd) {
      second = best;
      sd = bd;
      bd = s;
      best = u;
    } else if (s < sd) {
      sd = s;
      second = u;
    }
  }
  return { best, second, dist: Math.sqrt(bd) };
};

export function quantizationError(w: Matrix, x: Matrix): number {
  let s = 0;
  for (let i = 0; i < x.rows; i++) s += bmu(w, x, i).dist;
  return s / x.rows;
}

/** Share of samples whose two best units are not neighbours on the grid (0 = the map is topologically faithful). */
export function topographicError(w: Matrix, x: Matrix, cols: number): number {
  let bad = 0;
  for (let i = 0; i < x.rows; i++) {
    const { best, second } = bmu(w, x, i);
    const dr = Math.abs(Math.floor(best / cols) - Math.floor(second / cols));
    const dc = Math.abs((best % cols) - (second % cols));
    if (Math.max(dr, dc) > 1) bad++;
  }
  return bad / x.rows;
}

/** Average distance between each unit's weights and those of its grid neighbours: high values mark cluster borders. */
export function uMatrix(w: Matrix, rows: number, cols: number): Float64Array {
  const out = new Float64Array(rows * cols);
  for (let r = 0; r < rows; r++)
    for (let c = 0; c < cols; c++) {
      let sum = 0;
      let n = 0;
      for (const [dr, dc] of [
        [-1, 0],
        [1, 0],
        [0, -1],
        [0, 1],
      ] as const) {
        const rr = r + dr;
        const cc = c + dc;
        if (rr < 0 || rr >= rows || cc < 0 || cc >= cols) continue;
        let s = 0;
        for (let j = 0; j < w.cols; j++)
          s += (w.get(r * cols + c, j) - w.get(rr * cols + cc, j)) ** 2;
        sum += Math.sqrt(s);
        n++;
      }
      out[r * cols + c] = n ? sum / n : 0;
    }
  return out;
}

/** Training as a generator: state 0 is the random start, then one state per epoch. */
export function* somSteps(x: Matrix, options: SomOptions): Generator<SomState, void, undefined> {
  const { rows, cols, epochs = 40, learningRate = 0.5, seed = 1 } = options;
  const radius0 = options.radius ?? Math.max(rows, cols) / 2;
  const rng = createRng(seed);
  const units = rows * cols;
  // start on a small random patch of the data (rows picked at random, plus a little noise)
  const w = new Matrix(units, x.cols);
  for (let u = 0; u < units; u++) {
    const src = Math.floor(rng.next() * x.rows);
    for (let j = 0; j < x.cols; j++) w.set(u, j, x.get(src, j) + rng.normal() * 1e-3);
  }
  const snapshot = (epoch: number): SomState => {
    const copy = new Matrix(units, x.cols, Float64Array.from(w.data));
    return { epoch, weights: copy, quantizationError: quantizationError(copy, x) };
  };
  yield snapshot(0);
  const total = Math.max(1, epochs * x.rows);
  let step = 0;
  for (let epoch = 1; epoch <= epochs; epoch++) {
    const order = Array.from({ length: x.rows }, (_, i) => i);
    for (let i = order.length - 1; i > 0; i--) {
      const j = Math.floor(rng.next() * (i + 1));
      [order[i], order[j]] = [order[j] as number, order[i] as number];
    }
    for (const i of order) {
      const t = step++ / total;
      const lr = learningRate * (1 - t) + 0.005;
      const sigma = Math.max(0.5, radius0 * (1 - t));
      const { best } = bmu(w, x, i);
      const br = Math.floor(best / cols);
      const bc = best % cols;
      for (let u = 0; u < units; u++) {
        const d2 = (Math.floor(u / cols) - br) ** 2 + ((u % cols) - bc) ** 2;
        const h = Math.exp(-d2 / (2 * sigma * sigma));
        if (h < 1e-3) continue;
        for (let j = 0; j < x.cols; j++)
          w.data[u * x.cols + j] =
            (w.data[u * x.cols + j] as number) + lr * h * (x.get(i, j) - w.get(u, j));
      }
    }
    yield snapshot(epoch);
  }
}

export class SOM {
  weights: Matrix = new Matrix(0, 0);
  history: number[] = [];

  constructor(readonly options: SomOptions) {}

  fit(x: Matrix): this {
    this.history = [];
    for (const s of somSteps(x, this.options)) {
      this.weights = s.weights;
      this.history.push(s.quantizationError);
    }
    return this;
  }

  /** The grid cell (row, col) of the best-matching unit of every row. */
  assign(x: Matrix): { row: number; col: number }[] {
    return Array.from({ length: x.rows }, (_, i) => {
      const u = bmu(this.weights, x, i).best;
      return { row: Math.floor(u / this.options.cols), col: u % this.options.cols };
    });
  }
}
