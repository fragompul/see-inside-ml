import { Matrix, type Classifier, type Regressor, type Rng } from '@sim/core';

/**
 * CART decision trees (classification and regression). The tree greedily picks, at every node, the
 * feature and threshold that most reduce impurity, then recurses on both halves. Conventions follow
 * scikit-learn: thresholds are midpoints between consecutive sorted values, entropy uses log₂, and
 * pruning is minimal cost-complexity pruning.
 */
export type Criterion = 'gini' | 'entropy' | 'mse';

export interface TreeOptions {
  criterion?: Criterion;
  maxDepth?: number;
  minSamplesSplit?: number;
  minSamplesLeaf?: number;
  /** A split must reduce the (weighted) impurity by at least this much. */
  minImpurityDecrease?: number;
  /** Cost-complexity pruning strength (0 = grow fully). */
  ccpAlpha?: number;
  /** Features examined at each split (for random forests); default all. */
  maxFeatures?: number;
  rng?: Rng;
}

export interface TreeNode {
  /** Position in breadth-first order: the order in which a learner would watch the tree grow. */
  order: number;
  depth: number;
  /** Rows reaching the node. */
  samples: number;
  /** Total sample weight reaching the node. */
  weight: number;
  impurity: number;
  /** Class weights (classification) or `[mean]` (regression). */
  value: number[];
  feature: number;
  threshold: number;
  left: TreeNode | null;
  right: TreeNode | null;
}

/** Values closer than this cannot be split (scikit-learn's FEATURE_THRESHOLD). */
const EPS = 1e-7;
const PURE = Number.EPSILON;

const isLeaf = (n: TreeNode) => n.left === null;

function impurityOf(criterion: Criterion, counts: number[], total: number): number {
  if (total <= 0) return 0;
  if (criterion === 'gini') {
    let s = 0;
    for (const c of counts) s += (c / total) ** 2;
    return 1 - s;
  }
  let h = 0;
  for (const c of counts) if (c > 0) h -= (c / total) * Math.log2(c / total);
  return h;
}

interface Grown {
  root: TreeNode;
  importances: number[];
}

/** Builds a tree on `x` (rows) with targets `y` (class indices or reals) and optional sample weights. */
function grow(
  x: Matrix,
  y: ArrayLike<number>,
  weights: ArrayLike<number> | undefined,
  classes: number,
  options: TreeOptions,
): Grown {
  const criterion = options.criterion ?? (classes > 0 ? 'gini' : 'mse');
  const {
    maxDepth = Infinity,
    minSamplesSplit = 2,
    minSamplesLeaf = 1,
    minImpurityDecrease = 0,
    maxFeatures = x.cols,
  } = options;
  const n = x.rows;
  const w = weights ?? new Float64Array(n).fill(1);
  let totalWeight = 0;
  for (let i = 0; i < n; i++) totalWeight += w[i] as number;
  const importances = new Array<number>(x.cols).fill(0);

  const statsOf = (idx: number[]) => {
    let weight = 0;
    if (classes > 0) {
      const counts = new Array<number>(classes).fill(0);
      for (const i of idx) {
        counts[y[i] as number] = (counts[y[i] as number] as number) + (w[i] as number);
        weight += w[i] as number;
      }
      return { weight, value: counts, impurity: impurityOf(criterion, counts, weight) };
    }
    let sum = 0;
    for (const i of idx) {
      sum += (w[i] as number) * (y[i] as number);
      weight += w[i] as number;
    }
    const mean = weight > 0 ? sum / weight : 0;
    let sq = 0;
    for (const i of idx) sq += (w[i] as number) * ((y[i] as number) - mean) ** 2;
    return { weight, value: [mean], impurity: weight > 0 ? sq / weight : 0 };
  };

  const build = (idx: number[], depth: number): TreeNode => {
    const s = statsOf(idx);
    const node: TreeNode = {
      order: -1,
      depth,
      samples: idx.length,
      weight: s.weight,
      impurity: s.impurity,
      value: s.value,
      feature: -1,
      threshold: 0,
      left: null,
      right: null,
    };
    if (
      depth >= maxDepth ||
      idx.length < minSamplesSplit ||
      idx.length < 2 * minSamplesLeaf ||
      s.impurity <= PURE
    )
      return node;

    // candidate features (a random subset for forests)
    let features = Array.from({ length: x.cols }, (_, j) => j);
    if (maxFeatures < x.cols && options.rng) {
      for (let k = features.length - 1; k > 0; k--) {
        const r = Math.floor(options.rng.next() * (k + 1));
        [features[k], features[r]] = [features[r] as number, features[k] as number];
      }
      features = features.slice(0, maxFeatures);
    }

    let best: {
      gain: number;
      feature: number;
      threshold: number;
      position: number;
      order: number[];
    } | null = null;
    for (const f of features) {
      const order = [...idx].sort((a, b) => x.get(a, f) - x.get(b, f) || a - b);
      const leftCounts = classes > 0 ? new Array<number>(classes).fill(0) : [];
      let leftW = 0;
      let leftSum = 0;
      let leftSq = 0;
      let totalSum = 0;
      let totalSq = 0;
      if (classes === 0)
        for (const i of order) {
          totalSum += (w[i] as number) * (y[i] as number);
          totalSq += (w[i] as number) * (y[i] as number) ** 2;
        }
      for (let p = 1; p < order.length; p++) {
        const i = order[p - 1] as number;
        const wi = w[i] as number;
        if (classes > 0) leftCounts[y[i] as number] = (leftCounts[y[i] as number] as number) + wi;
        else {
          leftSum += wi * (y[i] as number);
          leftSq += wi * (y[i] as number) ** 2;
        }
        leftW += wi;
        const a = x.get(i, f);
        const b = x.get(order[p] as number, f);
        if (b - a <= EPS) continue; // identical values cannot be separated
        if (p < minSamplesLeaf || order.length - p < minSamplesLeaf) continue;
        const rightW = s.weight - leftW;
        let child: number;
        if (classes > 0) {
          const rightCounts = s.value.map((c, k) => c - (leftCounts[k] as number));
          child =
            (leftW * impurityOf(criterion, leftCounts, leftW) +
              rightW * impurityOf(criterion, rightCounts, rightW)) /
            s.weight;
        } else {
          const rightSum = totalSum - leftSum;
          const rightSq = totalSq - leftSq;
          const lVar = leftW > 0 ? Math.max(0, leftSq / leftW - (leftSum / leftW) ** 2) : 0;
          const rVar = rightW > 0 ? Math.max(0, rightSq / rightW - (rightSum / rightW) ** 2) : 0;
          child = (leftW * lVar + rightW * rVar) / s.weight;
        }
        const gain = s.impurity - child;
        if (!best || gain > best.gain + 1e-15) {
          let threshold = a / 2 + b / 2;
          if (threshold === b || !Number.isFinite(threshold)) threshold = a;
          best = { gain, feature: f, threshold, position: p, order };
        }
      }
    }
    // weighted impurity decrease relative to the whole dataset, as scikit-learn defines it
    if (!best || (s.weight / totalWeight) * best.gain < minImpurityDecrease) return node;

    node.feature = best.feature;
    node.threshold = best.threshold;
    importances[best.feature] =
      (importances[best.feature] as number) + (s.weight / totalWeight) * best.gain;
    const leftIdx = best.order.slice(0, best.position);
    const rightIdx = best.order.slice(best.position);
    node.left = build(leftIdx, depth + 1);
    node.right = build(rightIdx, depth + 1);
    return node;
  };

  const root = build(
    Array.from({ length: n }, (_, i) => i),
    0,
  );
  numberBreadthFirst(root);
  const total = importances.reduce((a, b) => a + b, 0);
  return { root, importances: total > 0 ? importances.map((v) => v / total) : importances };
}

function numberBreadthFirst(root: TreeNode): void {
  const queue: TreeNode[] = [root];
  let k = 0;
  while (queue.length) {
    const node = queue.shift() as TreeNode;
    node.order = k++;
    if (node.left) queue.push(node.left, node.right as TreeNode);
  }
}

function descend(root: TreeNode, x: Matrix, i: number): TreeNode {
  let node = root;
  while (node.left && node.right)
    node = x.get(i, node.feature) <= node.threshold ? node.left : node.right;
  return node;
}

export const treeDepth = (n: TreeNode): number =>
  n.left && n.right ? 1 + Math.max(treeDepth(n.left), treeDepth(n.right)) : 0;
export const leafCount = (n: TreeNode): number =>
  n.left && n.right ? leafCount(n.left) + leafCount(n.right) : 1;
export const nodeCount = (n: TreeNode): number =>
  n.left && n.right ? 1 + nodeCount(n.left) + nodeCount(n.right) : 1;

/** All nodes in breadth-first order. */
export function nodesOf(root: TreeNode): TreeNode[] {
  const out: TreeNode[] = [];
  const queue = [root];
  while (queue.length) {
    const n = queue.shift() as TreeNode;
    out.push(n);
    if (n.left && n.right) queue.push(n.left, n.right);
  }
  return out;
}

/** A copy of the tree keeping only the first `splits` splits in breadth-first order. */
export function truncateTree(root: TreeNode, splits: number): TreeNode {
  const copy = (n: TreeNode): TreeNode => {
    const splitsHere = n.left !== null && n.order < splits;
    return {
      ...n,
      left: splitsHere ? copy(n.left as TreeNode) : null,
      right: splitsHere ? copy(n.right as TreeNode) : null,
    };
  };
  return copy(root);
}

export interface PruningPath {
  /** Effective α at each pruning step (the first is 0: the full tree). */
  alphas: number[];
  /** Total leaf impurity of the tree after each step. */
  impurities: number[];
  leaves: number[];
}

/** Minimal cost-complexity pruning path (scikit-learn's `cost_complexity_pruning_path`). */
export function pruningPath(root: TreeNode): PruningPath {
  const work = cloneTree(root);
  const total = work.weight;
  const cost = (n: TreeNode) => (n.weight / total) * n.impurity;
  const leavesCost = (n: TreeNode): number =>
    n.left && n.right ? leavesCost(n.left) + leavesCost(n.right) : cost(n);
  const alphas = [0];
  const impurities = [leavesCost(work)];
  const leaves = [leafCount(work)];
  while (!isLeaf(work)) {
    let weakest: TreeNode | null = null;
    let weakestAlpha = Infinity;
    const visit = (n: TreeNode) => {
      if (!n.left || !n.right) return;
      const alpha = (cost(n) - leavesCost(n)) / (leafCount(n) - 1);
      if (alpha < weakestAlpha - 1e-15) {
        weakestAlpha = alpha;
        weakest = n;
      }
      visit(n.left);
      visit(n.right);
    };
    visit(work);
    const target: TreeNode = weakest as unknown as TreeNode;
    target.left = null;
    target.right = null;
    alphas.push(Math.max(weakestAlpha, 0));
    impurities.push(leavesCost(work));
    leaves.push(leafCount(work));
  }
  return { alphas, impurities, leaves };
}

function cloneTree(n: TreeNode): TreeNode {
  return {
    ...n,
    value: [...n.value],
    left: n.left && cloneTree(n.left),
    right: n.right && cloneTree(n.right),
  };
}

/** Prunes every subtree whose effective α does not exceed `alpha`. */
export function pruneTree(root: TreeNode, alpha: number): TreeNode {
  const work = cloneTree(root);
  const total = work.weight;
  const cost = (n: TreeNode) => (n.weight / total) * n.impurity;
  const leavesCost = (n: TreeNode): number =>
    n.left && n.right ? leavesCost(n.left) + leavesCost(n.right) : cost(n);
  for (;;) {
    let weakest: TreeNode | null = null;
    let weakestAlpha = Infinity;
    const visit = (n: TreeNode) => {
      if (!n.left || !n.right) return;
      const a = (cost(n) - leavesCost(n)) / (leafCount(n) - 1);
      if (a < weakestAlpha - 1e-15) {
        weakestAlpha = a;
        weakest = n;
      }
      visit(n.left);
      visit(n.right);
    };
    visit(work);
    if (!weakest || weakestAlpha > alpha) break;
    const target: TreeNode = weakest;
    target.left = null;
    target.right = null;
  }
  numberBreadthFirst(work);
  return work;
}

export class DecisionTree implements Classifier {
  classes = 2;
  root: TreeNode = {
    order: 0,
    depth: 0,
    samples: 0,
    weight: 0,
    impurity: 0,
    value: [],
    feature: -1,
    threshold: 0,
    left: null,
    right: null,
  };
  importances: number[] = [];

  constructor(readonly options: TreeOptions = {}) {}

  fit(x: Matrix, y: ArrayLike<number>, sampleWeight?: ArrayLike<number>): this {
    let k = 0;
    for (let i = 0; i < y.length; i++) k = Math.max(k, (y[i] as number) + 1);
    this.classes = Math.max(2, k);
    const grown = grow(x, y, sampleWeight, this.classes, this.options);
    this.importances = grown.importances;
    this.root =
      (this.options.ccpAlpha ?? 0) > 0
        ? pruneTree(grown.root, this.options.ccpAlpha as number)
        : grown.root;
    return this;
  }

  predictProba(x: Matrix): Matrix {
    const out = new Matrix(x.rows, this.classes);
    for (let i = 0; i < x.rows; i++) {
      const leaf = descend(this.root, x, i);
      const total = leaf.value.reduce((a, b) => a + b, 0) || 1;
      for (let c = 0; c < this.classes; c++) out.set(i, c, (leaf.value[c] ?? 0) / total);
    }
    return out;
  }

  predict(x: Matrix): Int32Array {
    const p = this.predictProba(x);
    return Int32Array.from({ length: x.rows }, (_, i) => {
      let best = 0;
      for (let c = 1; c < this.classes; c++) if (p.get(i, c) > p.get(i, best)) best = c;
      return best;
    });
  }

  /** The leaf a row falls in, to highlight the decision path. */
  leafOf(x: Matrix, i: number): TreeNode {
    return descend(this.root, x, i);
  }
}

export class DecisionTreeRegressor implements Regressor {
  root: TreeNode = {
    order: 0,
    depth: 0,
    samples: 0,
    weight: 0,
    impurity: 0,
    value: [0],
    feature: -1,
    threshold: 0,
    left: null,
    right: null,
  };
  importances: number[] = [];

  constructor(readonly options: TreeOptions = {}) {}

  fit(x: Matrix, y: ArrayLike<number>, sampleWeight?: ArrayLike<number>): this {
    const grown = grow(x, y, sampleWeight, 0, { ...this.options, criterion: 'mse' });
    this.importances = grown.importances;
    this.root =
      (this.options.ccpAlpha ?? 0) > 0
        ? pruneTree(grown.root, this.options.ccpAlpha as number)
        : grown.root;
    return this;
  }

  predict(x: Matrix): Float64Array {
    return Float64Array.from(
      { length: x.rows },
      (_, i) => descend(this.root, x, i).value[0] as number,
    );
  }

  leafOf(x: Matrix, i: number): TreeNode {
    return descend(this.root, x, i);
  }
}
