import { DEFAULT_CONFIG, PlaygroundSession } from '@sim/deep';
import { describe, expect, it } from 'vitest';
import { buildEdges } from './graph';
import {
  STAGES,
  biggestUpdate,
  edgeValue,
  formatValue,
  maxEdgeValue,
  nodeValue,
} from './inspectView';

const session = new PlaygroundSession({ ...DEFAULT_CONFIG, classes: 3 });
session.step(3);
const snapshot = session.snapshot();
const inspection = session.inspect(4);
const edges = buildEdges(snapshot.weights, session.scene().layerSizes, false);

describe('inspector view model', () => {
  it('has the four stages in order', () => {
    expect(STAGES.map((s) => s.id)).toEqual(['forward', 'loss', 'backward', 'update']);
  });

  it('forward edge values are weight × input, and sum (with bias) to the pre-activation', () => {
    // First hidden neuron: z = b + Σ w·x over its incoming edges.
    const incoming = edges.filter((e) => e.layerIndex === 0 && e.col === 0);
    const sum = incoming.reduce((s, e) => s + edgeValue(e, inspection, 0), 0);
    const bias = snapshot.weights[0]?.b[0] as number;
    expect(sum + bias).toBeCloseTo(inspection.preActivations[0]?.[0] as number, 10);
  });

  it('backward values are the gradients; update values are −lr × gradient', () => {
    const e = edges[5] as (typeof edges)[number];
    const grad = inspection.weightGrads[e.layerIndex]?.[e.row]?.[e.col] as number;
    expect(edgeValue(e, inspection, 2)).toBe(grad);
    expect(edgeValue(e, inspection, 3)).toBeCloseTo(-inspection.lr * grad, 14);
  });

  it('normalisers are positive and bound every edge value', () => {
    for (const stage of [0, 1, 2, 3]) {
      const max = maxEdgeValue(edges, inspection, stage);
      expect(max).toBeGreaterThan(0);
      for (const e of edges)
        expect(Math.abs(edgeValue(e, inspection, stage))).toBeLessThanOrEqual(max);
    }
  });

  it('neurons show activations going forward and deltas going back; inputs have no delta', () => {
    expect(nodeValue(0, 1, inspection, 0)).toBe(inspection.activations[0]?.[1]);
    expect(nodeValue(2, 1, inspection, 1)).toBe(inspection.activations[2]?.[1]);
    expect(nodeValue(2, 1, inspection, 2)).toBe(inspection.deltas[1]?.[1]);
    expect(nodeValue(0, 0, inspection, 2)).toBeNull();
    expect(nodeValue(9, 0, inspection, 0)).toBeNull();
  });

  it('finds the weight that moves the most', () => {
    const best = biggestUpdate(edges, inspection);
    const max = Math.max(...edges.map((e) => Math.abs(edgeValue(e, inspection, 3))));
    expect(Math.abs(best?.delta as number)).toBeCloseTo(max, 14);
    expect(biggestUpdate([], inspection)).toBeNull();
  });

  it('formats numbers compactly', () => {
    expect(formatValue(0)).toBe('0');
    expect(formatValue(0.5)).toBe('0.50');
    expect(formatValue(-12.34)).toBe('−12.3');
    expect(formatValue(1234)).toBe('1234');
    expect(formatValue(0.0004)).toBe('4e-4');
    expect(formatValue(Number.NaN)).toBe('–');
  });
});
