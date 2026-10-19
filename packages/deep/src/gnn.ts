import {
  Matrix,
  Optimizer,
  Parameter,
  createRng,
  matmul,
  matmulAT,
  matmulBT,
  type Rng,
} from '@sim/core';

/**
 * Graph convolutional networks (Kipf and Welling) for classifying the nodes of a graph. Every layer replaces a node's
 * features by an average of its own and its neighbours', then transforms them:
 *     H' = relu(A_hat H W),    A_hat = D^(-1/2) (A + I) D^(-1/2)
 * so that after k layers a node knows about everything within k steps. The layers are trained on a handful of labelled
 * nodes; the graph carries the labels to the rest. With `mix = 0` the averaging is skipped and the network is an
 * ordinary MLP that ignores the graph.
 */
export interface Graph {
  nodes: number;
  /** Undirected edges as pairs of node indices (each once). */
  edges: [number, number][];
}

export const normalisedAdjacency = (g: Graph): Matrix => {
  const n = g.nodes;
  const a = new Matrix(n, n);
  for (let i = 0; i < n; i++) a.data[i * n + i] = 1;
  for (const [u, v] of g.edges) {
    a.data[u * n + v] = 1;
    a.data[v * n + u] = 1;
  }
  const degree = new Float64Array(n);
  for (let i = 0; i < n; i++)
    for (let j = 0; j < n; j++) degree[i] = (degree[i] as number) + (a.data[i * n + j] as number);
  for (let i = 0; i < n; i++)
    for (let j = 0; j < n; j++)
      a.data[i * n + j] =
        (a.data[i * n + j] as number) / Math.sqrt((degree[i] as number) * (degree[j] as number));
  return a;
};

export interface PlantedGraph {
  graph: Graph;
  labels: number[];
  features: Matrix;
}

/** A graph with `classes` communities: edges are likely inside a community and unlikely between them, and node features only weakly hint at the community. */
export function plantedPartition(
  classes: number,
  perClass: number,
  pIn: number,
  pOut: number,
  featureDim: number,
  signal: number,
  rng: Rng,
): PlantedGraph {
  const n = classes * perClass;
  const labels = Array.from({ length: n }, (_, i) => Math.floor(i / perClass));
  const edges: [number, number][] = [];
  for (let i = 0; i < n; i++)
    for (let j = i + 1; j < n; j++)
      if (rng.next() < (labels[i] === labels[j] ? pIn : pOut)) edges.push([i, j]);
  const centres = Array.from({ length: classes }, () =>
    Float64Array.from({ length: featureDim }, () => rng.normal()),
  );
  const features = new Matrix(n, featureDim);
  for (let i = 0; i < n; i++)
    for (let d = 0; d < featureDim; d++)
      features.data[i * featureDim + d] =
        signal * ((centres[labels[i] as number] as Float64Array)[d] as number) + rng.normal();
  return { graph: { nodes: n, edges }, labels, features };
}

export interface GnnConfig {
  /** Number of layers (each averages over neighbours, except in an MLP). */
  layers: number;
  hidden: number;
  /** False gives an ordinary network that never looks at the graph. */
  useGraph: boolean;
}

export class Gcn {
  readonly weights: Matrix[] = [];
  readonly biases: Matrix[] = [];
  readonly params: Parameter[] = [];
  private cache: { inputs: Matrix[]; mixed: Matrix[]; pre: Matrix[]; probs: Matrix } | null = null;

  constructor(
    readonly config: GnnConfig,
    inputDim: number,
    readonly classes: number,
    readonly adjacency: Matrix,
    rng: Rng,
  ) {
    const dims = [
      inputDim,
      ...Array.from({ length: config.layers - 1 }, () => config.hidden),
      classes,
    ];
    for (let l = 0; l < config.layers; l++) {
      const fanIn = dims[l] as number;
      const fanOut = dims[l + 1] as number;
      const w = new Matrix(fanIn, fanOut);
      const std = Math.sqrt(2 / (fanIn + fanOut));
      for (let i = 0; i < w.data.length; i++) w.data[i] = rng.normal() * std;
      this.weights.push(w);
      this.biases.push(new Matrix(1, fanOut));
      this.params.push(
        new Parameter(w.data, true),
        new Parameter((this.biases[l] as Matrix).data, false),
      );
    }
  }

  get parameters(): Parameter[] {
    return this.params;
  }

  /** Class probabilities of every node, and the hidden representation after each layer (before the output). */
  forward(x: Matrix, cache = true): { probs: Matrix; hidden: Matrix[] } {
    const inputs: Matrix[] = [];
    const mixed: Matrix[] = [];
    const pre: Matrix[] = [];
    const hidden: Matrix[] = [];
    let h = x;
    for (let l = 0; l < this.config.layers; l++) {
      inputs.push(h);
      const m = this.config.useGraph ? matmul(this.adjacency, h) : h;
      mixed.push(m);
      const z = matmul(m, this.weights[l] as Matrix);
      const b = (this.biases[l] as Matrix).data;
      for (let r = 0; r < z.rows; r++)
        for (let c = 0; c < z.cols; c++)
          z.data[r * z.cols + c] = (z.data[r * z.cols + c] as number) + (b[c] as number);
      pre.push(z);
      if (l < this.config.layers - 1) {
        h = z.map((v) => (v > 0 ? v : 0));
        hidden.push(h);
      } else h = z;
    }
    const probs = new Matrix(h.rows, h.cols);
    for (let r = 0; r < h.rows; r++) {
      let top = -Infinity;
      for (let c = 0; c < h.cols; c++) top = Math.max(top, h.data[r * h.cols + c] as number);
      let sum = 0;
      for (let c = 0; c < h.cols; c++) {
        const e = Math.exp((h.data[r * h.cols + c] as number) - top);
        probs.data[r * h.cols + c] = e;
        sum += e;
      }
      for (let c = 0; c < h.cols; c++)
        probs.data[r * h.cols + c] = (probs.data[r * h.cols + c] as number) / sum;
    }
    if (cache) this.cache = { inputs, mixed, pre, probs };
    return { probs, hidden };
  }

  /** Mean cross-entropy over the nodes in `labelled`, with gradients filled in when `backprop` is set. */
  loss(
    x: Matrix,
    labels: ArrayLike<number>,
    labelled: readonly number[],
    backprop: boolean,
  ): number {
    const { probs } = this.forward(x, backprop);
    let total = 0;
    for (const i of labelled)
      total -=
        Math.log(Math.max(1e-12, probs.data[i * this.classes + (labels[i] as number)] as number)) /
        labelled.length;
    if (!backprop) return total;
    const c = this.cache as NonNullable<typeof this.cache>;
    let d = new Matrix(probs.rows, this.classes);
    for (const i of labelled) {
      for (let k = 0; k < this.classes; k++)
        d.data[i * this.classes + k] =
          ((probs.data[i * this.classes + k] as number) - (labels[i] === k ? 1 : 0)) /
          labelled.length;
    }
    for (let l = this.config.layers - 1; l >= 0; l--) {
      const dW = matmulAT(c.mixed[l] as Matrix, d);
      const p = this.params[2 * l] as Parameter;
      const pb = this.params[2 * l + 1] as Parameter;
      p.grad.set(dW.data);
      pb.grad.fill(0);
      for (let r = 0; r < d.rows; r++)
        for (let k = 0; k < d.cols; k++)
          pb.grad[k] = (pb.grad[k] as number) + (d.data[r * d.cols + k] as number);
      if (l > 0) {
        let dm = matmulBT(d, this.weights[l] as Matrix);
        if (this.config.useGraph) dm = matmul(this.adjacency, dm); // the adjacency is symmetric
        // through the relu of the previous layer
        const prev = c.pre[l - 1] as Matrix;
        d = new Matrix(dm.rows, dm.cols);
        for (let i = 0; i < dm.data.length; i++)
          d.data[i] = (prev.data[i] as number) > 0 ? (dm.data[i] as number) : 0;
      }
    }
    return total;
  }
}

export interface GnnRun {
  model: Gcn;
  trainAccuracy: number;
  testAccuracy: number;
  /** Cosine similarity of node representations after the last hidden layer: average over pairs in the same class and in different classes. */
  sameClass: number;
  otherClass: number;
}

export function accuracy(
  probs: Matrix,
  labels: ArrayLike<number>,
  nodes: readonly number[],
): number {
  let right = 0;
  for (const i of nodes) {
    let best = 0;
    for (let k = 1; k < probs.cols; k++)
      if (
        (probs.data[i * probs.cols + k] as number) > (probs.data[i * probs.cols + best] as number)
      )
        best = k;
    if (best === labels[i]) right++;
  }
  return right / nodes.length;
}

/** Average cosine similarity of the rows of `h` within classes and between them. */
export function similarity(h: Matrix, labels: ArrayLike<number>): { same: number; other: number } {
  const n = h.rows;
  const norm = Float64Array.from({ length: n }, (_, i) => {
    let s = 0;
    for (let d = 0; d < h.cols; d++) s += (h.data[i * h.cols + d] as number) ** 2;
    return Math.sqrt(s) + 1e-12;
  });
  let same = 0;
  let other = 0;
  let ns = 0;
  let no = 0;
  for (let i = 0; i < n; i += 3)
    for (let j = i + 1; j < n; j += 3) {
      let dot = 0;
      for (let d = 0; d < h.cols; d++)
        dot += (h.data[i * h.cols + d] as number) * (h.data[j * h.cols + d] as number);
      const cos = dot / ((norm[i] as number) * (norm[j] as number));
      if (labels[i] === labels[j]) {
        same += cos;
        ns++;
      } else {
        other += cos;
        no++;
      }
    }
  return { same: same / ns, other: other / no };
}

export function trainGcn(
  data: PlantedGraph,
  config: GnnConfig,
  options: { epochs: number; learningRate: number; labelledPerClass: number; seed: number },
): GnnRun {
  const rng = createRng(options.seed);
  const classes = Math.max(...data.labels) + 1;
  const n = data.graph.nodes;
  const labelled: number[] = [];
  for (let c = 0; c < classes; c++) {
    const members = data.labels.map((l, i) => (l === c ? i : -1)).filter((i) => i >= 0);
    for (let k = 0; k < options.labelledPerClass; k++)
      labelled.push(members[Math.floor(rng.next() * members.length)] as number);
  }
  const unlabelled = Array.from({ length: n }, (_, i) => i).filter((i) => !labelled.includes(i));
  const model = new Gcn(
    config,
    data.features.cols,
    classes,
    normalisedAdjacency(data.graph),
    createRng(options.seed + 1),
  );
  const optimizer = new Optimizer('adam');
  for (let e = 0; e < options.epochs; e++) {
    model.loss(data.features, data.labels, labelled, true);
    optimizer.step(model.parameters, options.learningRate);
  }
  const { probs, hidden } = model.forward(data.features, false);
  const last = hidden[hidden.length - 1] ?? data.features;
  const sim = similarity(last, data.labels);
  return {
    model,
    trainAccuracy: accuracy(probs, data.labels, labelled),
    testAccuracy: accuracy(probs, data.labels, unlabelled),
    sameClass: sim.same,
    otherClass: sim.other,
  };
}
