import { Matrix, createRng, type Classifier, type Regressor, type Rng } from '@sim/core';
import { DecisionTree, DecisionTreeRegressor, type TreeNode, type TreeOptions } from './tree';

/**
 * Ensembles of trees. Bagging (random forests) averages many deep trees grown on bootstrap
 * samples; boosting (AdaBoost, gradient boosting) adds shallow trees one after another, each one
 * correcting what the ensemble so far gets wrong.
 */

const argmax = (row: ArrayLike<number>): number => {
  let best = 0;
  for (let c = 1; c < row.length; c++) if ((row[c] as number) > (row[best] as number)) best = c;
  return best;
};

const classCount = (y: ArrayLike<number>): number => {
  let k = 0;
  for (let i = 0; i < y.length; i++) k = Math.max(k, (y[i] as number) + 1);
  return Math.max(2, k);
};

// ---- random forest ---------------------------------------------------------------------------

export interface ForestOptions extends Omit<TreeOptions, 'rng' | 'maxFeatures'> {
  trees?: number;
  /** Features tried per split; default √d for classification and all for regression. */
  maxFeatures?: number;
  bootstrap?: boolean;
  seed?: number;
}

function bootstrapIndices(n: number, rng: Rng, bootstrap: boolean): number[] {
  return Array.from({ length: n }, (_, i) => (bootstrap ? Math.floor(rng.next() * n) : i));
}

const takeRows = (x: Matrix, idx: readonly number[]): Matrix => {
  const out = new Matrix(idx.length, x.cols);
  idx.forEach((r, i) => out.data.set(x.data.subarray(r * x.cols, (r + 1) * x.cols), i * x.cols));
  return out;
};

export class RandomForestClassifier implements Classifier {
  classes = 2;
  trees: DecisionTree[] = [];
  /** `inBag[t][i]` = how many times row i was drawn for tree t. */
  inBag: Uint16Array[] = [];
  importances: number[] = [];
  /** Out-of-bag accuracy: every row is judged only by the trees that never saw it. */
  oobAccuracy: number | null = null;
  oobProba: Matrix | null = null;

  constructor(readonly options: ForestOptions = {}) {}

  fit(x: Matrix, y: ArrayLike<number>): this {
    const { trees = 100, bootstrap = true, seed = 1 } = this.options;
    const n = x.rows;
    this.classes = classCount(y);
    const rng = createRng(seed);
    const maxFeatures = this.options.maxFeatures ?? Math.max(1, Math.round(Math.sqrt(x.cols)));
    this.trees = [];
    this.inBag = [];
    for (let t = 0; t < trees; t++) {
      const idx = bootstrapIndices(n, rng, bootstrap);
      const counts = new Uint16Array(n);
      for (const i of idx) counts[i] = (counts[i] as number) + 1;
      const yy = idx.map((i) => y[i] as number);
      const tree = new DecisionTree({ ...this.options, maxFeatures, rng });
      // make sure every class exists in the sample so the probability vectors line up
      tree.fit(takeRows(x, idx), yy);
      tree.classes = this.classes;
      this.trees.push(tree);
      this.inBag.push(counts);
    }
    this.importances = Array.from(
      { length: x.cols },
      (_, j) => this.trees.reduce((s, t) => s + (t.importances[j] ?? 0), 0) / trees,
    );
    this.computeOob(x, y);
    return this;
  }

  private computeOob(x: Matrix, y: ArrayLike<number>): void {
    if (!(this.options.bootstrap ?? true)) {
      this.oobAccuracy = null;
      this.oobProba = null;
      return;
    }
    const n = x.rows;
    const sum = new Matrix(n, this.classes);
    const votes = new Uint16Array(n);
    this.trees.forEach((tree, t) => {
      const bag = this.inBag[t] as Uint16Array;
      const out: number[] = [];
      for (let i = 0; i < n; i++) if (bag[i] === 0) out.push(i);
      if (out.length === 0) return;
      const p = this.treeProba(tree, takeRows(x, out));
      out.forEach((row, r) => {
        votes[row] = (votes[row] as number) + 1;
        for (let c = 0; c < this.classes; c++) sum.set(row, c, sum.get(row, c) + p.get(r, c));
      });
    });
    let correct = 0;
    let counted = 0;
    for (let i = 0; i < n; i++) {
      if (!votes[i]) continue;
      counted++;
      const row = Array.from({ length: this.classes }, (_, c) => sum.get(i, c));
      if (argmax(row) === y[i]) correct++;
      for (let c = 0; c < this.classes; c++) sum.set(i, c, sum.get(i, c) / (votes[i] as number));
    }
    this.oobAccuracy = counted ? correct / counted : null;
    this.oobProba = sum;
  }

  /**
   * Out-of-bag accuracy using only the first k trees, for k = 1 … T: how the forest's honest error
   * estimate improves as trees are added.
   */
  oobCurve(x: Matrix, y: ArrayLike<number>): number[] {
    const n = x.rows;
    const sum = new Matrix(n, this.classes);
    const votes = new Uint16Array(n);
    const curve: number[] = [];
    this.trees.forEach((tree, t) => {
      const bag = this.inBag[t] as Uint16Array;
      const out: number[] = [];
      for (let i = 0; i < n; i++) if (bag[i] === 0) out.push(i);
      if (out.length) {
        const p = this.treeProba(tree, takeRows(x, out));
        out.forEach((row, r) => {
          votes[row] = (votes[row] as number) + 1;
          for (let c = 0; c < this.classes; c++) sum.set(row, c, sum.get(row, c) + p.get(r, c));
        });
      }
      let correct = 0;
      let counted = 0;
      for (let i = 0; i < n; i++) {
        if (!votes[i]) continue;
        counted++;
        if (argmax(sum.data.subarray(i * this.classes, (i + 1) * this.classes)) === y[i]) correct++;
      }
      curve.push(counted ? correct / counted : NaN);
    });
    return curve;
  }

  /** A tree's class probabilities padded to the forest's class count. */
  private treeProba(tree: DecisionTree, x: Matrix): Matrix {
    const p = tree.predictProba(x);
    if (p.cols === this.classes) return p;
    const out = new Matrix(x.rows, this.classes);
    for (let i = 0; i < x.rows; i++) for (let c = 0; c < p.cols; c++) out.set(i, c, p.get(i, c));
    return out;
  }

  predictProba(x: Matrix): Matrix {
    const out = new Matrix(x.rows, this.classes);
    for (const tree of this.trees) {
      const p = this.treeProba(tree, x);
      for (let i = 0; i < out.data.length; i++)
        out.data[i] = (out.data[i] as number) + (p.data[i] as number);
    }
    for (let i = 0; i < out.data.length; i++)
      out.data[i] = (out.data[i] as number) / this.trees.length;
    return out;
  }

  predict(x: Matrix): Int32Array {
    const p = this.predictProba(x);
    return Int32Array.from({ length: x.rows }, (_, i) =>
      argmax(p.data.subarray(i * p.cols, (i + 1) * p.cols)),
    );
  }
}

export class RandomForestRegressor implements Regressor {
  trees: DecisionTreeRegressor[] = [];
  inBag: Uint16Array[] = [];
  importances: number[] = [];
  /** Out-of-bag R² (1 − residual / total sum of squares over the rows with an OOB prediction). */
  oobR2: number | null = null;
  oobPrediction: Float64Array | null = null;

  constructor(readonly options: ForestOptions = {}) {}

  fit(x: Matrix, y: ArrayLike<number>): this {
    const { trees = 100, bootstrap = true, seed = 1 } = this.options;
    const n = x.rows;
    const rng = createRng(seed);
    const maxFeatures = this.options.maxFeatures ?? x.cols;
    this.trees = [];
    this.inBag = [];
    for (let t = 0; t < trees; t++) {
      const idx = bootstrapIndices(n, rng, bootstrap);
      const counts = new Uint16Array(n);
      for (const i of idx) counts[i] = (counts[i] as number) + 1;
      const tree = new DecisionTreeRegressor({ ...this.options, maxFeatures, rng });
      tree.fit(
        takeRows(x, idx),
        idx.map((i) => y[i] as number),
      );
      this.trees.push(tree);
      this.inBag.push(counts);
    }
    this.importances = Array.from(
      { length: x.cols },
      (_, j) => this.trees.reduce((s, t) => s + (t.importances[j] ?? 0), 0) / trees,
    );
    if (bootstrap) {
      const sum = new Float64Array(n);
      const votes = new Uint16Array(n);
      this.trees.forEach((tree, t) => {
        const bag = this.inBag[t] as Uint16Array;
        const out: number[] = [];
        for (let i = 0; i < n; i++) if (bag[i] === 0) out.push(i);
        if (!out.length) return;
        const p = tree.predict(takeRows(x, out));
        out.forEach((row, r) => {
          sum[row] = (sum[row] as number) + (p[r] as number);
          votes[row] = (votes[row] as number) + 1;
        });
      });
      const pred = new Float64Array(n).fill(NaN);
      let mean = 0;
      let counted = 0;
      for (let i = 0; i < n; i++)
        if (votes[i]) {
          pred[i] = (sum[i] as number) / (votes[i] as number);
          mean += y[i] as number;
          counted++;
        }
      mean /= Math.max(1, counted);
      let rss = 0;
      let tss = 0;
      for (let i = 0; i < n; i++)
        if (votes[i]) {
          rss += ((y[i] as number) - (pred[i] as number)) ** 2;
          tss += ((y[i] as number) - mean) ** 2;
        }
      this.oobPrediction = pred;
      this.oobR2 = counted && tss > 0 ? 1 - rss / tss : null;
    }
    return this;
  }

  predict(x: Matrix): Float64Array {
    const out = new Float64Array(x.rows);
    for (const tree of this.trees) {
      const p = tree.predict(x);
      for (let i = 0; i < out.length; i++) out[i] = (out[i] as number) + (p[i] as number);
    }
    return out.map((v) => v / this.trees.length);
  }
}

// ---- AdaBoost ---------------------------------------------------------------------------------

export interface AdaBoostOptions {
  rounds?: number;
  learningRate?: number;
  /** Depth of the weak learners (1 = decision stumps). */
  depth?: number;
}

export interface AdaBoostRound {
  tree: DecisionTree;
  alpha: number;
  error: number;
  /** Sample weights the tree was trained with (normalised to sum 1). */
  weights: Float64Array;
}

/** Multi-class AdaBoost (SAMME): reweights the points each round so the next learner focuses on the mistakes. */
export class AdaBoost implements Classifier {
  classes = 2;
  rounds: AdaBoostRound[] = [];

  constructor(readonly options: AdaBoostOptions = {}) {}

  fit(x: Matrix, y: ArrayLike<number>): this {
    const { rounds = 50, learningRate = 1, depth = 1 } = this.options;
    const n = x.rows;
    const k = classCount(y);
    this.classes = k;
    let w = new Float64Array(n).fill(1 / n);
    this.rounds = [];
    for (let m = 0; m < rounds; m++) {
      const tree = new DecisionTree({ maxDepth: depth }).fit(x, y, w);
      tree.classes = k;
      const pred = tree.predict(x);
      let error = 0;
      for (let i = 0; i < n; i++) if (pred[i] !== y[i]) error += w[i] as number;
      error /= w.reduce((a, b) => a + b, 0);
      const weights = Float64Array.from(w);
      if (error <= 0) {
        // a perfect learner: scikit-learn gives it weight 1 and stops
        this.rounds.push({ tree, alpha: 1, error, weights });
        break;
      }
      if (error >= 1 - 1 / k) {
        // no better than guessing: stop (keep it only if it is the first round)
        if (this.rounds.length === 0) this.rounds.push({ tree, alpha: 0, error, weights });
        break;
      }
      const alpha = learningRate * (Math.log((1 - error) / error) + Math.log(k - 1));
      this.rounds.push({ tree, alpha, error, weights });
      w = Float64Array.from(w, (v, i) => (pred[i] !== y[i] ? v * Math.exp(alpha) : v));
      const total = w.reduce((a, b) => a + b, 0);
      w = w.map((v) => v / total);
    }
    return this;
  }

  /**
   * Weighted votes using only the first `count` rounds: a learner adds `α` to the class it picks
   * and subtracts `α / (K − 1)` from every other class (scikit-learn's SAMME decision function).
   */
  decision(x: Matrix, count = this.rounds.length): Matrix {
    const out = new Matrix(x.rows, this.classes);
    const k = this.classes;
    for (const r of this.rounds.slice(0, count)) {
      const pred = r.tree.predict(x);
      for (let i = 0; i < x.rows; i++)
        for (let c = 0; c < k; c++)
          out.set(i, c, out.get(i, c) + (pred[i] === c ? r.alpha : -r.alpha / (k - 1)));
    }
    return out;
  }

  predictProba(x: Matrix, count = this.rounds.length): Matrix {
    const d = this.decision(x, count);
    const out = new Matrix(x.rows, this.classes);
    const total = this.rounds.slice(0, count).reduce((s, r) => s + r.alpha, 0) || 1;
    for (let i = 0; i < x.rows; i++) {
      // scikit-learn: softmax of the decision function scaled by 1 / (K − 1), the decision being
      // normalised by the sum of the alphas
      const z = Array.from(
        { length: this.classes },
        (_, c) => d.get(i, c) / total / (this.classes - 1),
      );
      const max = Math.max(...z);
      const e = z.map((v) => Math.exp(v - max));
      const s = e.reduce((a, b) => a + b, 0);
      e.forEach((v, c) => out.set(i, c, v / s));
    }
    return out;
  }

  predict(x: Matrix, count = this.rounds.length): Int32Array {
    const d = this.decision(x, count);
    return Int32Array.from({ length: x.rows }, (_, i) =>
      argmax(d.data.subarray(i * d.cols, (i + 1) * d.cols)),
    );
  }
}

// ---- gradient boosting ------------------------------------------------------------------------

export interface GradientBoostingOptions {
  rounds?: number;
  learningRate?: number;
  depth?: number;
  /** Fraction of the rows each tree sees (1 = all; < 1 is stochastic gradient boosting). */
  subsample?: number;
  minSamplesLeaf?: number;
  seed?: number;
}

const sigmoid = (z: number) => 1 / (1 + Math.exp(-z));

/** Replaces each leaf's value by `fn(rows in the leaf)`. */
function relabelLeaves(
  tree: DecisionTreeRegressor,
  x: Matrix,
  fn: (rows: number[]) => number,
): void {
  const buckets = new Map<TreeNode, number[]>();
  for (let i = 0; i < x.rows; i++) {
    const leaf = tree.leafOf(x, i);
    const list = buckets.get(leaf);
    if (list) list.push(i);
    else buckets.set(leaf, [i]);
  }
  for (const [leaf, rows] of buckets) leaf.value = [fn(rows)];
}

export class GradientBoostingRegressor implements Regressor {
  init = 0;
  trees: DecisionTreeRegressor[] = [];
  /** Training loss (mean squared error) after each round, starting with the constant model. */
  trainLoss: number[] = [];

  constructor(readonly options: GradientBoostingOptions = {}) {}

  fit(x: Matrix, y: ArrayLike<number>): this {
    const {
      rounds = 100,
      learningRate = 0.1,
      depth = 3,
      subsample = 1,
      minSamplesLeaf = 1,
      seed = 1,
    } = this.options;
    const n = x.rows;
    const rng = createRng(seed);
    this.init = Array.from(y).reduce((a, b) => a + b, 0) / n;
    const f = new Float64Array(n).fill(this.init);
    this.trees = [];
    this.trainLoss = [mseOf(y, f)];
    for (let m = 0; m < rounds; m++) {
      const residual = Float64Array.from(f, (v, i) => (y[i] as number) - v);
      const rows =
        subsample < 1
          ? Array.from({ length: n }, (_, i) => i).filter(() => rng.next() < subsample)
          : Array.from({ length: n }, (_, i) => i);
      const use = rows.length >= 2 ? rows : Array.from({ length: n }, (_, i) => i);
      const tree = new DecisionTreeRegressor({ maxDepth: depth, minSamplesLeaf });
      tree.fit(
        takeRows(x, use),
        use.map((i) => residual[i] as number),
      );
      // refresh the leaves with the residual mean over *all* rows of that leaf (a no-op without subsampling)
      if (subsample < 1)
        relabelLeaves(
          tree,
          x,
          (r) => r.reduce((s, i) => s + (residual[i] as number), 0) / r.length,
        );
      const p = tree.predict(x);
      for (let i = 0; i < n; i++) f[i] = (f[i] as number) + learningRate * (p[i] as number);
      this.trees.push(tree);
      this.trainLoss.push(mseOf(y, f));
    }
    return this;
  }

  /** Prediction using the first `count` trees (default all). */
  predict(x: Matrix, count = this.trees.length): Float64Array {
    const lr = this.options.learningRate ?? 0.1;
    const out = new Float64Array(x.rows).fill(this.init);
    for (const tree of this.trees.slice(0, count)) {
      const p = tree.predict(x);
      for (let i = 0; i < out.length; i++) out[i] = (out[i] as number) + lr * (p[i] as number);
    }
    return out;
  }
}

function mseOf(y: ArrayLike<number>, f: ArrayLike<number>): number {
  let s = 0;
  for (let i = 0; i < y.length; i++) s += ((y[i] as number) - (f[i] as number)) ** 2;
  return s / y.length;
}

/** Gradient boosting for classification with the log-loss (binary) or the softmax cross-entropy. */
export class GradientBoostingClassifier implements Classifier {
  classes = 2;
  /** Initial raw scores, one per class (binary: a single log-odds). */
  init: number[] = [];
  /** `trees[m][k]`: the tree of round m for class k (binary: one tree per round). */
  trees: DecisionTreeRegressor[][] = [];
  trainLoss: number[] = [];

  constructor(readonly options: GradientBoostingOptions = {}) {}

  fit(x: Matrix, y: ArrayLike<number>): this {
    const {
      rounds = 100,
      learningRate = 0.1,
      depth = 3,
      subsample = 1,
      minSamplesLeaf = 1,
      seed = 1,
    } = this.options;
    const n = x.rows;
    const k = classCount(y);
    this.classes = k;
    const rng = createRng(seed);
    const binary = k === 2;
    const width = binary ? 1 : k;
    const counts = new Array<number>(k).fill(0);
    for (let i = 0; i < n; i++) counts[y[i] as number] = (counts[y[i] as number] as number) + 1;
    this.init = binary
      ? [Math.log((counts[1] as number) / (counts[0] as number))]
      : counts.map((c) => Math.log(c / n));
    const f = Array.from({ length: width }, (_, c) =>
      new Float64Array(n).fill(this.init[c] as number),
    );
    this.trees = [];
    this.trainLoss = [this.loss(y, f)];
    for (let m = 0; m < rounds; m++) {
      const probs = this.probabilities(f);
      const round: DecisionTreeRegressor[] = [];
      const rows =
        subsample < 1
          ? Array.from({ length: n }, (_, i) => i).filter(() => rng.next() < subsample)
          : Array.from({ length: n }, (_, i) => i);
      const use = rows.length >= 2 ? rows : Array.from({ length: n }, (_, i) => i);
      for (let c = 0; c < width; c++) {
        const target = binary ? 1 : c;
        const residual = Float64Array.from(
          { length: n },
          (_, i) => (y[i] === target ? 1 : 0) - (probs[c]![i] as number),
        );
        const tree = new DecisionTreeRegressor({ maxDepth: depth, minSamplesLeaf });
        tree.fit(
          takeRows(x, use),
          use.map((i) => residual[i] as number),
        );
        // Newton step per leaf, as in Friedman's paper and scikit-learn
        relabelLeaves(tree, takeRows(x, use), (r) => {
          const rr = r.map((i) => residual[use[i] as number] as number);
          const num = rr.reduce((s, v) => s + v, 0);
          const den = rr.reduce((s, v) => s + Math.abs(v) * (1 - Math.abs(v)), 0);
          const scale = binary ? 1 : (k - 1) / k;
          return den > 1e-150 ? (scale * num) / den : 0;
        });
        round.push(tree);
        const p = tree.predict(x);
        for (let i = 0; i < n; i++)
          (f[c] as Float64Array)[i] =
            ((f[c] as Float64Array)[i] as number) + learningRate * (p[i] as number);
      }
      this.trees.push(round);
      this.trainLoss.push(this.loss(y, f));
    }
    return this;
  }

  private probabilities(f: Float64Array[]): Float64Array[] {
    const n = (f[0] as Float64Array).length;
    if (this.classes === 2) return [Float64Array.from(f[0] as Float64Array, sigmoid)];
    const out = Array.from({ length: this.classes }, () => new Float64Array(n));
    for (let i = 0; i < n; i++) {
      const z = f.map((a) => a[i] as number);
      const max = Math.max(...z);
      const e = z.map((v) => Math.exp(v - max));
      const s = e.reduce((a, b) => a + b, 0);
      e.forEach((v, c) => ((out[c] as Float64Array)[i] = v / s));
    }
    return out;
  }

  private loss(y: ArrayLike<number>, f: Float64Array[]): number {
    const p = this.probabilities(f);
    let s = 0;
    for (let i = 0; i < y.length; i++) {
      const pr =
        this.classes === 2
          ? y[i] === 1
            ? (p[0] as Float64Array)[i]
            : 1 - ((p[0] as Float64Array)[i] as number)
          : (p[y[i] as number] as Float64Array)[i];
      s -= Math.log(Math.max(pr as number, 1e-300));
    }
    return s / y.length;
  }

  /** Raw scores after the first `count` rounds (binary: one column of log-odds). */
  decision(x: Matrix, count = this.trees.length): Matrix {
    const lr = this.options.learningRate ?? 0.1;
    const width = this.classes === 2 ? 1 : this.classes;
    const out = new Matrix(x.rows, width);
    for (let c = 0; c < width; c++)
      for (let i = 0; i < x.rows; i++) out.set(i, c, this.init[c] as number);
    for (const round of this.trees.slice(0, count))
      round.forEach((tree, c) => {
        const p = tree.predict(x);
        for (let i = 0; i < x.rows; i++) out.set(i, c, out.get(i, c) + lr * (p[i] as number));
      });
    return out;
  }

  predictProba(x: Matrix, count = this.trees.length): Matrix {
    const d = this.decision(x, count);
    const out = new Matrix(x.rows, this.classes);
    for (let i = 0; i < x.rows; i++) {
      if (this.classes === 2) {
        const p = sigmoid(d.get(i, 0));
        out.set(i, 0, 1 - p);
        out.set(i, 1, p);
      } else {
        const z = Array.from({ length: this.classes }, (_, c) => d.get(i, c));
        const max = Math.max(...z);
        const e = z.map((v) => Math.exp(v - max));
        const s = e.reduce((a, b) => a + b, 0);
        e.forEach((v, c) => out.set(i, c, v / s));
      }
    }
    return out;
  }

  predict(x: Matrix, count = this.trees.length): Int32Array {
    const p = this.predictProba(x, count);
    return Int32Array.from({ length: x.rows }, (_, i) =>
      argmax(p.data.subarray(i * p.cols, (i + 1) * p.cols)),
    );
  }
}
