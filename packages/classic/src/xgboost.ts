import { Matrix, type Classifier, type Regressor } from '@sim/core';

/**
 * XGBoost-style gradient boosting: each round fits a tree to the first and second derivatives
 * (g, h) of the loss, scoring a split by
 *
 *   gain = G_L²/(H_L+λ) + G_R²/(H_R+λ) − (G_L+G_R)²/(H_L+H_R+λ)   (xgboost's loss_chg)
 *
 * keeping a split only if its gain exceeds γ (the paper writes this as ½·gain − γ > 0 up to scale), and
 * setting a leaf to −G/(H+λ). λ shrinks leaves, γ is the price of one more split, and
 * `minChildWeight` keeps leaves backed by enough curvature. The exact greedy algorithm, as in
 * `tree_method="exact"`.
 */
export interface XgbOptions {
  rounds?: number;
  /** Learning rate η: shrinks every tree's contribution. */
  eta?: number;
  maxDepth?: number;
  /** L2 penalty λ on the leaf weights. */
  lambda?: number;
  /** Minimum gain γ a split must achieve (otherwise pruned). */
  gamma?: number;
  minChildWeight?: number;
  /** Starting prediction (probability for classification). */
  baseScore?: number;
}

export interface XgbNode {
  feature: number;
  threshold: number;
  left: XgbNode | null;
  right: XgbNode | null;
  /** Leaf weight (already multiplied by η). */
  value: number;
  /** Gain of the split (before subtracting γ), 0 for leaves. */
  gain: number;
  /** Sum of the Hessians of the rows in the node. */
  cover: number;
  depth: number;
}

function buildTree(
  x: Matrix,
  g: Float64Array,
  h: Float64Array,
  rows: number[],
  opts: Required<Pick<XgbOptions, 'eta' | 'maxDepth' | 'lambda' | 'gamma' | 'minChildWeight'>>,
  depth: number,
): XgbNode {
  let G = 0;
  let H = 0;
  for (const i of rows) {
    G += g[i] as number;
    H += h[i] as number;
  }
  const leaf: XgbNode = {
    feature: -1,
    threshold: 0,
    left: null,
    right: null,
    value: (-G / (H + opts.lambda)) * opts.eta,
    gain: 0,
    cover: H,
    depth,
  };
  if (depth >= opts.maxDepth || rows.length < 2) return leaf;
  const parent = (G * G) / (H + opts.lambda);
  let best: {
    gain: number;
    feature: number;
    threshold: number;
    position: number;
    order: number[];
  } | null = null;
  for (let f = 0; f < x.cols; f++) {
    const order = [...rows].sort((a, b) => x.get(a, f) - x.get(b, f) || a - b);
    let gl = 0;
    let hl = 0;
    for (let p = 1; p < order.length; p++) {
      const i = order[p - 1] as number;
      gl += g[i] as number;
      hl += h[i] as number;
      const a = x.get(i, f);
      const b = x.get(order[p] as number, f);
      if (b <= a) continue;
      const hr = H - hl;
      if (hl < opts.minChildWeight || hr < opts.minChildWeight) continue;
      const gr = G - gl;
      // xgboost's loss_chg: twice the ½-scaled gain of the paper, and the quantity γ is compared with
      const gain = (gl * gl) / (hl + opts.lambda) + (gr * gr) / (hr + opts.lambda) - parent;
      if (!best || gain > best.gain)
        best = { gain, feature: f, threshold: (a + b) / 2, position: p, order };
    }
  }
  if (!best || best.gain <= 1e-6) return leaf;
  const left = buildTree(x, g, h, best.order.slice(0, best.position), opts, depth + 1);
  const right = buildTree(x, g, h, best.order.slice(best.position), opts, depth + 1);
  // bottom-up pruning: a split whose gain is below γ disappears unless it protects a worthwhile subtree
  if (!left.left && !right.left && best.gain < opts.gamma) return leaf;
  return {
    ...leaf,
    feature: best.feature,
    threshold: best.threshold,
    left,
    right,
    gain: best.gain,
  };
}

function leafValue(node: XgbNode, x: Matrix, i: number): number {
  let n = node;
  while (n.left && n.right) n = x.get(i, n.feature) < n.threshold ? n.left : n.right;
  return n.value;
}

const withDefaults = (o: XgbOptions) => ({
  eta: o.eta ?? 0.3,
  maxDepth: o.maxDepth ?? 6,
  lambda: o.lambda ?? 1,
  gamma: o.gamma ?? 0,
  minChildWeight: o.minChildWeight ?? 1,
});

const sigmoid = (z: number) => 1 / (1 + Math.exp(-z));

export class XgbRegressor implements Regressor {
  trees: XgbNode[] = [];
  base = 0.5;
  /** Training RMSE after each round (the first entry is the constant model). */
  trainRmse: number[] = [];

  constructor(readonly options: XgbOptions = {}) {}

  fit(x: Matrix, y: ArrayLike<number>): this {
    const opts = withDefaults(this.options);
    const n = x.rows;
    this.base = this.options.baseScore ?? 0.5;
    const f = new Float64Array(n).fill(this.base);
    const rows = Array.from({ length: n }, (_, i) => i);
    this.trees = [];
    const rmse = () => Math.sqrt(f.reduce((s, v, i) => s + ((y[i] as number) - v) ** 2, 0) / n);
    this.trainRmse = [rmse()];
    for (let m = 0; m < (this.options.rounds ?? 100); m++) {
      const g = Float64Array.from(f, (v, i) => v - (y[i] as number));
      const h = new Float64Array(n).fill(1);
      const tree = buildTree(x, g, h, rows, opts, 0);
      for (let i = 0; i < n; i++) f[i] = (f[i] as number) + leafValue(tree, x, i);
      this.trees.push(tree);
      this.trainRmse.push(rmse());
    }
    return this;
  }

  predict(x: Matrix, count = this.trees.length): Float64Array {
    const out = new Float64Array(x.rows).fill(this.base);
    for (const t of this.trees.slice(0, count))
      for (let i = 0; i < x.rows; i++) out[i] = (out[i] as number) + leafValue(t, x, i);
    return out;
  }
}

export class XgbClassifier implements Classifier {
  classes = 2;
  /** `trees[m][k]` for round m and class k (binary: a single tree per round). */
  trees: XgbNode[][] = [];
  base = 0.5;
  trainLoss: number[] = [];

  constructor(readonly options: XgbOptions = {}) {}

  fit(x: Matrix, y: ArrayLike<number>): this {
    const opts = withDefaults(this.options);
    const n = x.rows;
    let k = 0;
    for (let i = 0; i < n; i++) k = Math.max(k, (y[i] as number) + 1);
    this.classes = Math.max(2, k);
    const binary = this.classes === 2;
    const width = binary ? 1 : this.classes;
    this.base = this.options.baseScore ?? 0.5;
    const start = binary ? Math.log(this.base / (1 - this.base)) : 0;
    const f = Array.from({ length: width }, () => new Float64Array(n).fill(start));
    const rows = Array.from({ length: n }, (_, i) => i);
    this.trees = [];
    const probs = (): Float64Array[] => {
      if (binary) return [Float64Array.from(f[0] as Float64Array, sigmoid)];
      const out = Array.from({ length: width }, () => new Float64Array(n));
      for (let i = 0; i < n; i++) {
        const z = f.map((a) => a[i] as number);
        const max = Math.max(...z);
        const e = z.map((v) => Math.exp(v - max));
        const s = e.reduce((a, b) => a + b, 0);
        e.forEach((v, c) => ((out[c] as Float64Array)[i] = v / s));
      }
      return out;
    };
    const loss = () => {
      const p = probs();
      let s = 0;
      for (let i = 0; i < n; i++) {
        const pr = binary
          ? y[i] === 1
            ? (p[0] as Float64Array)[i]
            : 1 - ((p[0] as Float64Array)[i] as number)
          : ((p[y[i] as number] as Float64Array)[i] as number);
        s -= Math.log(Math.max(pr as number, 1e-300));
      }
      return s / n;
    };
    this.trainLoss = [loss()];
    for (let m = 0; m < (this.options.rounds ?? 100); m++) {
      const p = probs();
      const round: XgbNode[] = [];
      for (let c = 0; c < width; c++) {
        const target = binary ? 1 : c;
        const g = Float64Array.from(
          { length: n },
          (_, i) => (p[c] as Float64Array)[i]! - (y[i] === target ? 1 : 0),
        );
        const h = Float64Array.from({ length: n }, (_, i) => {
          const pr = (p[c] as Float64Array)[i] as number;
          return Math.max(binary ? pr * (1 - pr) : 2 * pr * (1 - pr), 1e-16);
        });
        const tree = buildTree(x, g, h, rows, opts, 0);
        for (let i = 0; i < n; i++)
          (f[c] as Float64Array)[i] = ((f[c] as Float64Array)[i] as number) + leafValue(tree, x, i);
        round.push(tree);
      }
      this.trees.push(round);
      this.trainLoss.push(loss());
    }
    return this;
  }

  predictProba(x: Matrix, count = this.trees.length): Matrix {
    const binary = this.classes === 2;
    const width = binary ? 1 : this.classes;
    const start = binary ? Math.log(this.base / (1 - this.base)) : 0;
    const z = Array.from({ length: width }, () => new Float64Array(x.rows).fill(start));
    for (const round of this.trees.slice(0, count))
      round.forEach((t, c) => {
        for (let i = 0; i < x.rows; i++)
          (z[c] as Float64Array)[i] = ((z[c] as Float64Array)[i] as number) + leafValue(t, x, i);
      });
    const out = new Matrix(x.rows, this.classes);
    for (let i = 0; i < x.rows; i++) {
      if (binary) {
        const p = sigmoid((z[0] as Float64Array)[i] as number);
        out.set(i, 0, 1 - p);
        out.set(i, 1, p);
      } else {
        const v = z.map((a) => a[i] as number);
        const max = Math.max(...v);
        const e = v.map((q) => Math.exp(q - max));
        const s = e.reduce((a, b) => a + b, 0);
        e.forEach((q, c) => out.set(i, c, q / s));
      }
    }
    return out;
  }

  predict(x: Matrix, count = this.trees.length): Int32Array {
    const p = this.predictProba(x, count);
    return Int32Array.from({ length: x.rows }, (_, i) => {
      let best = 0;
      for (let c = 1; c < this.classes; c++) if (p.get(i, c) > p.get(i, best)) best = c;
      return best;
    });
  }
}

/** Total gain each feature contributed across all trees, as a share of the total ("gain" importance). */
export function gainImportance(trees: XgbNode[], features: number): number[] {
  const total = new Array<number>(features).fill(0);
  const walk = (n: XgbNode) => {
    if (n.left && n.right) {
      total[n.feature] = (total[n.feature] as number) + n.gain;
      walk(n.left);
      walk(n.right);
    }
  };
  trees.forEach(walk);
  const s = total.reduce((a, b) => a + b, 0);
  return s > 0 ? total.map((v) => v / s) : total;
}
