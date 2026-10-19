import { type StepInspection } from '@sim/deep';
import type { GraphEdge } from './graph';

/** The four moments of a training step, in order. */
export const STAGES = [
  { id: 'forward', title: 'Forward pass', short: 'Forward' },
  { id: 'loss', title: 'Loss', short: 'Loss' },
  { id: 'backward', title: 'Backward pass', short: 'Backward' },
  { id: 'update', title: 'Weight update', short: 'Update' },
] as const;

export type StageIndex = 0 | 1 | 2 | 3;

/**
 * Signed quantity that an edge should display in a given stage:
 * - forward / loss: the edge's contribution to the next neuron, w · input;
 * - backward: its gradient ∂L/∂w = input · δ;
 * - update: the weight change Δw = −η · ∂L/∂w.
 */
export function edgeValue(edge: GraphEdge, inspection: StepInspection, stage: number): number {
  if (stage <= 1) {
    const input = inspection.layerInputs[edge.layerIndex]?.[edge.row] ?? 0;
    return edge.weight * input;
  }
  const gradient = inspection.weightGrads[edge.layerIndex]?.[edge.row]?.[edge.col] ?? 0;
  return stage === 2 ? gradient : -inspection.lr * gradient;
}

/** Largest |edgeValue| over all edges, used to normalise widths (never 0). */
export function maxEdgeValue(
  edges: readonly GraphEdge[],
  inspection: StepInspection,
  stage: number,
): number {
  let max = 0;
  for (const edge of edges) max = Math.max(max, Math.abs(edgeValue(edge, inspection, stage)));
  return max || 1;
}

/**
 * Number to print on a neuron: its activation while the signal flows forward, its error term δ
 * while it flows back. Input features have no δ (nothing is learned upstream of them).
 */
export function nodeValue(
  column: number,
  index: number,
  inspection: StepInspection,
  stage: number,
): number | null {
  if (stage <= 1) return inspection.activations[column]?.[index] ?? null;
  if (column === 0) return null;
  return inspection.deltas[column - 1]?.[index] ?? null;
}

/** Compact signed number for small labels. */
export function formatValue(value: number): string {
  if (!Number.isFinite(value)) return '–';
  const abs = Math.abs(value);
  const sign = value < 0 ? '−' : '';
  if (abs === 0) return '0';
  if (abs >= 100) return `${sign}${abs.toFixed(0)}`;
  if (abs >= 10) return `${sign}${abs.toFixed(1)}`;
  if (abs >= 0.01) return `${sign}${abs.toFixed(2)}`;
  return `${sign}${abs.toExponential(0)}`;
}

/** The edge whose weight changes the most in this step (a nice thing to point out). */
export function biggestUpdate(
  edges: readonly GraphEdge[],
  inspection: StepInspection,
): { edge: GraphEdge; delta: number } | null {
  let best: { edge: GraphEdge; delta: number } | null = null;
  for (const edge of edges) {
    const delta = edgeValue(edge, inspection, 3);
    if (!best || Math.abs(delta) > Math.abs(best.delta)) best = { edge, delta };
  }
  return best;
}
