import { createRng } from '@sim/core';
import { accuracy, plantedPartition, trainGcn } from '@sim/deep';

export interface GnnScenario {
  /** Probability of an edge between nodes of different communities (inside it is 0.15). */
  pOut: number;
  /** How strongly the node features hint at the community. */
  signal: number;
  /** Labelled nodes per community. */
  labelled: number;
  /** The depth and kind of network shown in the picture. */
  layers: number;
  useGraph: boolean;
}

export const DEFAULT_GNN: GnnScenario = {
  pOut: 0.01,
  signal: 0.7,
  labelled: 4,
  layers: 2,
  useGraph: true,
};

export const DEPTHS = [1, 2, 3, 4, 6, 8];
const SEEDS = [1, 2, 3];
const CLASSES = 3;
const PER_CLASS = 40;

export interface GnnResult {
  scenario: GnnScenario;
  depths: number[];
  /** Mean test accuracy over three random graphs, per depth, with and without using the graph. */
  graph: number[];
  plain: number[];
  /** Mean cosine similarity of representations of nodes in different communities, per depth (graph network); 1 means indistinguishable. */
  otherClass: number[];
  /** The graph of the first random draw, laid out with one cluster per community, and what the shown network predicts. */
  nodes: { x: number; y: number; label: number; predicted: number; labelled: boolean }[];
  edges: [number, number][];
  shownAccuracy: number;
}

const cache = new Map<string, GnnResult>();
const sweeps = new Map<string, { graph: number[]; plain: number[]; otherClass: number[] }>();

function sweep(s: GnnScenario) {
  const key = `${s.pOut}|${s.signal}|${s.labelled}`;
  const hit = sweeps.get(key);
  if (hit) return hit;
  const graph: number[] = [];
  const plain: number[] = [];
  const otherClass: number[] = [];
  for (const layers of DEPTHS) {
    let g = 0;
    let p = 0;
    let o = 0;
    for (const seed of SEEDS) {
      const data = plantedPartition(
        CLASSES,
        PER_CLASS,
        0.15,
        s.pOut,
        6,
        s.signal,
        createRng(seed + 10),
      );
      const options = { epochs: 200, learningRate: 0.05, labelledPerClass: s.labelled, seed };
      const withGraph = trainGcn(data, { layers, hidden: 16, useGraph: true }, options);
      g += withGraph.testAccuracy / SEEDS.length;
      o += withGraph.otherClass / SEEDS.length;
      p +=
        trainGcn(data, { layers, hidden: 16, useGraph: false }, options).testAccuracy /
        SEEDS.length;
    }
    graph.push(g);
    plain.push(p);
    otherClass.push(o);
  }
  const out = { graph, plain, otherClass };
  sweeps.set(key, out);
  return out;
}

export function runGnnLab(s: GnnScenario): GnnResult {
  const key = JSON.stringify(s);
  const hit = cache.get(key);
  if (hit) return hit;
  const { graph, plain, otherClass } = sweep(s);
  const data = plantedPartition(CLASSES, PER_CLASS, 0.15, s.pOut, 6, s.signal, createRng(11));
  const options = { epochs: 200, learningRate: 0.05, labelledPerClass: s.labelled, seed: 1 };
  const run = trainGcn(data, { layers: s.layers, hidden: 16, useGraph: s.useGraph }, options);
  const { probs } = run.model.forward(data.features, false);
  const positions = createRng(5);
  const labelled = new Set<number>();
  // the labelled nodes are the ones trainGcn chose with this seed: recompute them the same way
  const pick = createRng(options.seed);
  for (let c = 0; c < CLASSES; c++) {
    const members = data.labels.map((l, i) => (l === c ? i : -1)).filter((i) => i >= 0);
    for (let k = 0; k < options.labelledPerClass; k++)
      labelled.add(members[Math.floor(pick.next() * members.length)] as number);
  }
  const nodes = data.labels.map((label, i) => {
    const angle = (2 * Math.PI * label) / CLASSES - Math.PI / 2;
    let predicted = 0;
    for (let k = 1; k < CLASSES; k++)
      if ((probs.data[i * CLASSES + k] as number) > (probs.data[i * CLASSES + predicted] as number))
        predicted = k;
    return {
      x: 0.62 * Math.cos(angle) + 0.28 * positions.normal() * 0.7,
      y: 0.62 * Math.sin(angle) + 0.28 * positions.normal() * 0.7,
      label,
      predicted,
      labelled: labelled.has(i),
    };
  });
  const unlabelled = nodes.map((_, i) => i).filter((i) => !labelled.has(i));
  const out: GnnResult = {
    scenario: s,
    depths: DEPTHS,
    graph,
    plain,
    otherClass,
    nodes,
    edges: data.graph.edges,
    shownAccuracy: accuracy(probs, data.labels, unlabelled),
  };
  cache.set(key, out);
  return out;
}
