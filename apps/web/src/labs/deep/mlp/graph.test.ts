import { describe, expect, it } from 'vitest';
import { buildEdges, computeLayout, edgePath, maxAbsWeight } from './graph';

describe('computeLayout', () => {
  const layout = computeLayout({ sizes: [2, 4, 3, 2], width: 800, height: 460, headerHeight: 60 });

  it('spreads columns evenly inside the padding', () => {
    expect(layout.columnX[0]).toBe(56);
    expect(layout.columnX[3]).toBe(744);
    const gaps = layout.columnX.slice(1).map((x, i) => x - (layout.columnX[i] as number));
    expect(new Set(gaps.map((g) => Math.round(g))).size).toBe(1);
  });

  it('centres every column vertically and keeps nodes inside the free area', () => {
    const mids = layout.nodeY.map((ys) => ((ys[0] as number) + (ys[ys.length - 1] as number)) / 2);
    expect(new Set(mids.map((m) => Math.round(m))).size).toBe(1);
    for (const ys of layout.nodeY) {
      for (const y of ys) {
        expect(y).toBeGreaterThan(60);
        expect(y).toBeLessThan(460);
      }
    }
  });

  it('shrinks nodes when a column gets crowded', () => {
    const crowded = computeLayout({ sizes: [2, 16, 2], width: 800, height: 460, headerHeight: 60 });
    expect(crowded.diameter).toBeLessThan(layout.diameter);
    expect(crowded.diameter).toBeGreaterThanOrEqual(18);
  });
});

describe('buildEdges', () => {
  const weights = [
    {
      W: [
        [1, -1, 0.5],
        [0.2, 0.3, -0.4],
      ],
      b: [0, 0, 0],
    }, // 2 → 3
    { W: [[1], [2], [3]], b: [0] }, // 3 → 1
  ];

  it('creates one edge per weight, wired between adjacent layers', () => {
    const edges = buildEdges(weights, [2, 3, 1], false);
    expect(edges).toHaveLength(6 + 3);
    expect(edges.every((e) => e.toLayer === e.fromLayer + 1 && !e.skip)).toBe(true);
    expect(edges[1]).toMatchObject({
      fromLayer: 0,
      fromIndex: 0,
      toLayer: 1,
      toIndex: 1,
      weight: -1,
    });
  });

  it('records where each weight lives so gradients can be looked up', () => {
    const edges = buildEdges(weights, [2, 3, 1], false);
    for (const e of edges) {
      const w = weights[e.layerIndex]!.W[e.row]![e.col];
      expect(e.weight).toBe(w);
    }
  });

  it('routes skip-connection rows to the input column', () => {
    // second layer sees [3 hidden | 2 inputs] = 5 rows
    const skipWeights = [weights[0]!, { W: [[1], [1], [1], [7], [9]], b: [0] }];
    const edges = buildEdges(skipWeights, [2, 3, 1], true);
    const skipEdges = edges.filter((e) => e.skip);
    expect(skipEdges).toHaveLength(2);
    expect(skipEdges[0]).toMatchObject({ fromLayer: 0, fromIndex: 0, toLayer: 2, weight: 7 });
    expect(skipEdges[1]).toMatchObject({ fromLayer: 0, fromIndex: 1, toLayer: 2, weight: 9 });
  });

  it('finds the largest magnitude', () => {
    expect(maxAbsWeight(buildEdges(weights, [2, 3, 1], false))).toBe(3);
    expect(maxAbsWeight([])).toBe(0);
  });
});

describe('edgePath', () => {
  it('draws a horizontal-tangent cubic Bézier', () => {
    expect(edgePath(0, 0, 100, 50)).toBe('M0,0 C50,0 50,50 100,50');
  });
});
