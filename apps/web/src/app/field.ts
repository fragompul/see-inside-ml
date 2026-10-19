import { createRng } from '@sim/core';

export interface FieldNode {
  layer: number;
  /** Normalised position inside the field (0..1). */
  x: number;
  y: number;
  phase: number;
}

export interface FieldEdge {
  from: number;
  to: number;
  /** Signed weight in [-1, 1]; sign picks the colour (neg / pos). */
  weight: number;
}

export interface Field {
  nodes: FieldNode[];
  edges: FieldEdge[];
  layers: number[][];
  /** edge indices leaving / entering each node, for forward / backward propagation. */
  outgoing: number[][];
  incoming: number[][];
}

/** Fully-connected feed-forward layout used as the hero's living background. */
export function buildField(layerSizes: number[], seed = 7): Field {
  const rng = createRng(seed);
  const nodes: FieldNode[] = [];
  const layers: number[][] = [];

  layerSizes.forEach((size, layer) => {
    const ids: number[] = [];
    for (let i = 0; i < size; i++) {
      ids.push(nodes.length);
      nodes.push({
        layer,
        x: layerSizes.length === 1 ? 0.5 : layer / (layerSizes.length - 1),
        y: size === 1 ? 0.5 : i / (size - 1),
        phase: rng.uniform(0, Math.PI * 2),
      });
    }
    layers.push(ids);
  });

  const edges: FieldEdge[] = [];
  for (let l = 0; l < layers.length - 1; l++) {
    for (const from of layers[l] ?? []) {
      for (const to of layers[l + 1] ?? []) {
        // tanh squashes a normal sample into a nicely spread (-1, 1) range.
        edges.push({ from, to, weight: Math.tanh(rng.normal() * 0.9) });
      }
    }
  }

  const outgoing: number[][] = nodes.map(() => []);
  const incoming: number[][] = nodes.map(() => []);
  edges.forEach((edge, index) => {
    outgoing[edge.from]?.push(index);
    incoming[edge.to]?.push(index);
  });

  return { nodes, edges, layers, outgoing, incoming };
}
