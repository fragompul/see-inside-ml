import type { Matrix, Regressor } from '@sim/core';

/**
 * Isotonic regression by the pool-adjacent-violators algorithm: the best non-decreasing (or non-increasing) function
 * of one input in the least-squares sense. Neighbouring points that break the order are pooled into blocks that take
 * their weighted mean, until no violation is left. Predictions between blocks are interpolated linearly, as in
 * scikit-learn, and clipped outside the range of the data.
 */
export interface IsotonicOptions {
  increasing: boolean;
}

export interface Block {
  /** First and last x of the block, its fitted value and the total weight of the points in it. */
  first: number;
  last: number;
  value: number;
  weight: number;
}

/** Pool-adjacent-violators on points already sorted by x with weights; returns the blocks. */
export function pava(
  x: readonly number[],
  y: readonly number[],
  weights?: readonly number[],
): { blocks: Block[]; steps: Block[][] } {
  const blocks: Block[] = x.map((xi, i) => ({
    first: xi,
    last: xi,
    value: y[i] as number,
    weight: weights ? (weights[i] as number) : 1,
  }));
  const stack: Block[] = [];
  const steps: Block[][] = [];
  for (const b of blocks) {
    stack.push({ ...b });
    while (
      stack.length > 1 &&
      (stack[stack.length - 2] as Block).value > (stack[stack.length - 1] as Block).value
    ) {
      const top = stack.pop() as Block;
      const below = stack.pop() as Block;
      const w = below.weight + top.weight;
      stack.push({
        first: below.first,
        last: top.last,
        value: (below.value * below.weight + top.value * top.weight) / w,
        weight: w,
      });
    }
    steps.push(stack.map((s) => ({ ...s })));
  }
  return { blocks: stack, steps };
}

export class IsotonicRegression implements Regressor {
  blocks: Block[] = [];
  /** The blocks after each point was added (for stepping through the algorithm). */
  steps: Block[][] = [];
  private flip = 1;

  constructor(private readonly options: IsotonicOptions = { increasing: true }) {}

  fit(x: Matrix, y: ArrayLike<number>): this {
    if (x.cols !== 1) throw new RangeError('Isotonic regression takes a single input');
    this.flip = this.options.increasing ? 1 : -1;
    const order = Array.from({ length: x.rows }, (_, i) => i).sort(
      (a, b) => (x.data[a] as number) - (x.data[b] as number),
    );
    // identical x values are averaged first
    const xs: number[] = [];
    const ys: number[] = [];
    const ws: number[] = [];
    for (const i of order) {
      const xi = x.data[i] as number;
      const yi = this.flip * (y[i] as number);
      if (xs.length && xs[xs.length - 1] === xi) {
        const w = ws[ws.length - 1] as number;
        ys[ys.length - 1] = ((ys[ys.length - 1] as number) * w + yi) / (w + 1);
        ws[ws.length - 1] = w + 1;
      } else {
        xs.push(xi);
        ys.push(yi);
        ws.push(1);
      }
    }
    const { blocks, steps } = pava(xs, ys, ws);
    this.blocks = blocks.map((b) => ({ ...b, value: this.flip * b.value }));
    this.steps = steps.map((s) => s.map((b) => ({ ...b, value: this.flip * b.value })));
    return this;
  }

  predictOne(x: number): number {
    const b = this.blocks;
    if (!b.length) throw new Error('fit() has not been called');
    if (x <= (b[0] as Block).first) return (b[0] as Block).value;
    if (x >= (b[b.length - 1] as Block).last) return (b[b.length - 1] as Block).value;
    for (let k = 0; k < b.length; k++) {
      const cur = b[k] as Block;
      if (x >= cur.first && x <= cur.last) return cur.value;
      const next = b[k + 1];
      if (next && x > cur.last && x < next.first) {
        const t = (x - cur.last) / (next.first - cur.last);
        return cur.value + t * (next.value - cur.value);
      }
    }
    return (b[b.length - 1] as Block).value;
  }

  predict(x: Matrix): Float64Array {
    return Float64Array.from({ length: x.rows }, (_, i) => this.predictOne(x.data[i] as number));
  }

  /** Number of distinct steps in the fitted function. */
  get pieces(): number {
    return this.blocks.length;
  }
}
