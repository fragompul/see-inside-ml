import { type LayerWeights } from '@sim/deep';

export interface GraphLayout {
  /** x of each column (0 = inputs … last = outputs). */
  columnX: number[];
  /** y of every node, per column. */
  nodeY: number[][];
  /** Node diameter in px. */
  diameter: number;
}

export interface LayoutInput {
  sizes: number[];
  width: number;
  height: number;
  /** Space reserved above the nodes for column headers. */
  headerHeight: number;
  paddingX?: number;
  paddingBottom?: number;
  maxDiameter?: number;
}

/** Positions columns evenly and centres each column's nodes in the free vertical space. */
export function computeLayout({
  sizes,
  width,
  height,
  headerHeight,
  paddingX = 56,
  paddingBottom = 20,
  maxDiameter = 46,
}: LayoutInput): GraphLayout {
  const columns = sizes.length;
  const usable = Math.max(1, width - 2 * paddingX);
  const columnX = sizes.map((_, c) =>
    columns === 1 ? width / 2 : paddingX + (c * usable) / (columns - 1),
  );

  const top = headerHeight;
  const available = Math.max(40, height - headerHeight - paddingBottom);
  const widest = Math.max(...sizes, 1);
  const gap = Math.min(64, available / widest);
  const diameter = Math.max(18, Math.min(maxDiameter, gap * 0.8));
  const middle = top + available / 2;

  const nodeY = sizes.map((count) =>
    Array.from({ length: count }, (_, i) => middle + (i - (count - 1) / 2) * gap),
  );
  return { columnX, nodeY, diameter };
}

export interface GraphEdge {
  fromLayer: number;
  fromIndex: number;
  toLayer: number;
  toIndex: number;
  weight: number;
  /** Edge fed by the raw input features into a deeper layer (skip connection). */
  skip: boolean;
  /** Position of the weight in the network: layer `layerIndex`, W[row][col]. */
  layerIndex: number;
  row: number;
  col: number;
}

/**
 * Turns the network's weight matrices into drawable edges. With skip connections every layer
 * after the first sees [previous activations | raw inputs], so its extra rows originate from the
 * input column.
 */
export function buildEdges(
  weights: readonly LayerWeights[],
  sizes: readonly number[],
  skip: boolean,
): GraphEdge[] {
  const edges: GraphEdge[] = [];
  weights.forEach((layer, l) => {
    const previous = sizes[l] as number;
    layer.W.forEach((row, i) => {
      const fromInput = skip && l > 0 && i >= previous;
      row.forEach((weight, j) => {
        edges.push({
          fromLayer: fromInput ? 0 : l,
          fromIndex: fromInput ? i - previous : i,
          toLayer: l + 1,
          toIndex: j,
          weight,
          skip: fromInput,
          layerIndex: l,
          row: i,
          col: j,
        });
      });
    });
  });
  return edges;
}

export function maxAbsWeight(edges: readonly GraphEdge[]): number {
  let max = 0;
  for (const e of edges) max = Math.max(max, Math.abs(e.weight));
  return max;
}

/** Horizontal cubic Bézier between two node centres. */
export function edgePath(x1: number, y1: number, x2: number, y2: number): string {
  const dx = (x2 - x1) * 0.5;
  return `M${x1},${y1} C${x1 + dx},${y1} ${x2 - dx},${y2} ${x2},${y2}`;
}
