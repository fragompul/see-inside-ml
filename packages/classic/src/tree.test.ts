import { Matrix, createRng } from '@sim/core';
import { describe, expect, it } from 'vitest';
import fixture from '../fixtures/tree.json';
import {
  DecisionTree,
  DecisionTreeRegressor,
  leafCount,
  nodeCount,
  pruneTree,
  pruningPath,
  treeDepth,
  truncateTree,
  type TreeNode,
  type TreeOptions,
} from './tree';

interface RefNode {
  depth: number;
  feature: number;
  threshold: number;
  samples: number;
  weight: number;
  impurity: number;
  value: number[];
  leaf: boolean;
}

const preorder = (n: TreeNode, out: TreeNode[] = []): TreeNode[] => {
  out.push(n);
  if (n.left && n.right) {
    preorder(n.left, out);
    preorder(n.right, out);
  }
  return out;
};

const optionsOf = (cfg: Record<string, number | string | undefined>): TreeOptions => ({
  ...(cfg.criterion ? { criterion: cfg.criterion as 'gini' | 'entropy' } : {}),
  ...(cfg.max_depth ? { maxDepth: cfg.max_depth as number } : {}),
  ...(cfg.min_samples_leaf ? { minSamplesLeaf: cfg.min_samples_leaf as number } : {}),
  ...(cfg.min_samples_split ? { minSamplesSplit: cfg.min_samples_split as number } : {}),
  ...(cfg.min_impurity_decrease
    ? { minImpurityDecrease: cfg.min_impurity_decrease as number }
    : {}),
  ...(cfg.ccp_alpha ? { ccpAlpha: cfg.ccp_alpha as number } : {}),
});

/**
 * Compares both trees node by node in pre-order. Where several splits tie exactly (deep inside a
 * fully grown tree, or a node of two points) scikit-learn breaks the tie with a random feature
 * order, so those trees are compared only down to `maxDepth`.
 */
function sameTree(root: TreeNode, ref: RefNode[], normalise: boolean, maxDepth = Infinity) {
  const mine = preorder(root).filter((n) => n.depth <= maxDepth);
  const theirs = ref.filter((r) => r.depth <= maxDepth);
  if (maxDepth === Infinity) expect(mine.length).toBe(theirs.length);
  theirs.forEach((r, i) => {
    const n = mine[i] as TreeNode;
    expect(n.depth).toBe(r.depth);
    if (n.depth === maxDepth) return; // the subtree below may differ only in tie-breaking
    expect(n.left === null).toBe(r.leaf);
    expect(n.samples).toBe(r.samples);
    expect(Math.abs(n.weight - r.weight)).toBeLessThan(1e-9);
    expect(Math.abs(n.impurity - r.impurity)).toBeLessThan(1e-9);
    if (!r.leaf) {
      expect(n.feature).toBe(r.feature);
      expect(Math.abs(n.threshold - r.threshold)).toBeLessThan(1e-9);
    }
    const total = normalise ? n.value.reduce((a, b) => a + b, 0) : 1;
    n.value.forEach((v, k) =>
      expect(Math.abs(v / total - (r.value[k] as number))).toBeLessThan(1e-9),
    );
  });
}

/** Configurations that stop growing before exact ties can appear: their trees match node for node. */
const exact = (cfg: Record<string, unknown>) =>
  ['max_depth', 'min_samples_leaf', 'min_samples_split', 'min_impurity_decrease', 'ccp_alpha'].some(
    (k) => k in cfg,
  );

const x = Matrix.from(fixture.x);
const grid = Matrix.from(fixture.grid);

describe('classification trees agree with scikit-learn', () => {
  for (const ref of fixture.classifier) {
    it(`structure, predictions and importances: ${JSON.stringify(ref.config)}`, () => {
      const m = new DecisionTree(optionsOf(ref.config)).fit(x, fixture.y);
      sameTree(m.root, ref.nodes, true, exact(ref.config) ? Infinity : 3);
      const pred = Array.from(m.predict(grid));
      const agree = pred.filter((c, i) => c === ref.pred[i]).length / pred.length;
      // identical trees predict identically; fully grown ones may differ where exact ties were broken
      expect(agree).toBeGreaterThanOrEqual(exact(ref.config) ? 1 : 0.9);
      if (exact(ref.config) && !('ccp_alpha' in ref.config))
        ref.importances.forEach((v, j) =>
          expect(Math.abs((m.importances[j] as number) - v)).toBeLessThan(1e-9),
        );
    });
  }

  it('sample weights', () => {
    const m = new DecisionTree({ maxDepth: 4 }).fit(x, fixture.y, fixture.weighted.sw);
    sameTree(m.root, fixture.weighted.nodes, true);
    expect(Array.from(m.predict(grid))).toEqual(fixture.weighted.pred);
  });

  it('the cost-complexity pruning path', () => {
    const full = new DecisionTree().fit(x, fixture.y);
    const path = pruningPath(full.root);
    const unique = (a: number[]) =>
      a.filter((v, i) => i === 0 || Math.abs(v - (a[i - 1] as number)) > 1e-12);
    const mine = unique(path.alphas);
    const ref = unique(fixture.path.alphas);
    expect(mine.length).toBe(ref.length);
    ref.forEach((v, i) => expect(Math.abs((mine[i] as number) - v)).toBeLessThan(1e-9));
  });
});

describe('regression trees agree with scikit-learn', () => {
  const xr = Matrix.from(fixture.regressor.x);
  const gr = Matrix.from(fixture.regressor.grid);
  for (const ref of fixture.regressor.fits) {
    it(`structure and predictions: ${JSON.stringify(ref.config)}`, () => {
      const m = new DecisionTreeRegressor(optionsOf(ref.config)).fit(xr, fixture.regressor.y);
      sameTree(m.root, ref.nodes, false, Object.keys(ref.config).length ? Infinity : 3);
      const pred = m.predict(gr);
      const agree =
        ref.pred.filter((v, i) => Math.abs((pred[i] as number) - v) < 1e-9).length /
        ref.pred.length;
      expect(agree).toBeGreaterThanOrEqual(Object.keys(ref.config).length ? 1 : 0.6);
      if (Object.keys(ref.config).length && !('ccp_alpha' in ref.config))
        ref.importances.forEach((v, j) =>
          expect(Math.abs((m.importances[j] as number) - v)).toBeLessThan(1e-9),
        );
    });
  }
});

describe('properties', () => {
  it('an unrestricted tree memorises distinct training points; depth limits the fit', () => {
    const full = new DecisionTree().fit(x, fixture.y);
    expect(Array.from(full.predict(x))).toEqual(fixture.y);
    const shallow = new DecisionTree({ maxDepth: 2 }).fit(x, fixture.y);
    expect(treeDepth(shallow.root)).toBeLessThanOrEqual(2);
    expect(leafCount(shallow.root)).toBeLessThanOrEqual(4);
  });

  it('pruning only ever shrinks the tree, monotonically in α', () => {
    const full = new DecisionTree().fit(x, fixture.y).root;
    let previous = nodeCount(full);
    for (const alpha of [0.001, 0.005, 0.01, 0.05, 0.2]) {
      const size = nodeCount(pruneTree(full, alpha));
      expect(size).toBeLessThanOrEqual(previous);
      previous = size;
    }
    expect(nodeCount(pruneTree(full, 10))).toBe(1);
  });

  it('truncating to k splits keeps exactly k splits, in breadth-first order', () => {
    const root = new DecisionTree().fit(x, fixture.y).root;
    for (const k of [0, 1, 3, 6]) {
      const t = truncateTree(root, k);
      expect(nodeCount(t)).toBe(1 + 2 * k);
    }
  });

  it('XOR: no single split helps, yet a deeper tree carves the four quadrants', () => {
    const rng = createRng(1);
    const rows: number[][] = [];
    const y: number[] = [];
    for (let i = 0; i < 200; i++) {
      const a = rng.next() * 2 - 1;
      const b = rng.next() * 2 - 1;
      rows.push([a, b]);
      y.push(a * b > 0 ? 1 : 0);
    }
    const t = new DecisionTree({ maxDepth: 6 }).fit(Matrix.from(rows), y);
    const acc = Array.from(t.predict(Matrix.from(rows))).filter((p, i) => p === y[i]).length / 200;
    expect(acc).toBeGreaterThan(0.9);
  });

  it('feature importances sum to one', () => {
    const m = new DecisionTree().fit(x, fixture.y);
    expect(m.importances.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 12);
  });
});
