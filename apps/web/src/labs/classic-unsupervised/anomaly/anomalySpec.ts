import {
  IsolationForest,
  MahalanobisDetector,
  localOutlierFactor,
  rocAuc,
  thresholdFor,
} from '@sim/classic';
import type { Clusterer, Matrix } from '@sim/core';
import {
  num,
  str,
  type ClusterExperiment,
  type ClusterLabSpec,
  type ParamSpec,
} from '@/shared/lab-kit';

const params: ParamSpec[] = [
  {
    key: 'method',
    label: 'Detector',
    kind: 'segmented',
    options: [
      { value: 'lof', label: 'Local outlier factor' },
      { value: 'iforest', label: 'Isolation forest' },
      { value: 'mahalanobis', label: 'Mahalanobis' },
    ],
  },
  {
    key: 'neighbors',
    label: 'Neighbours k',
    kind: 'slider',
    min: 3,
    max: 60,
    step: 1,
    hint: 'How many neighbours define “the local crowd”.',
    showIf: (v) => v.method === 'lof',
  },
  {
    key: 'trees',
    label: 'Trees',
    kind: 'slider',
    min: 10,
    max: 300,
    step: 10,
    hint: 'More trees = steadier scores.',
    showIf: (v) => v.method === 'iforest',
  },
  {
    key: 'contamination',
    label: 'Share flagged',
    kind: 'slider',
    min: 0.01,
    max: 0.3,
    step: 0.01,
    format: (v) => `${Math.round(v * 100)}%`,
    hint: 'The detector ranks points; this sets how many of the top-ranked are called anomalies.',
  },
];

/** A detector wearing the `Clusterer` interface: label 1 = flagged, 0 = normal. */
export class AnomalyDetector implements Clusterer {
  labels = new Int32Array(0);
  scores: Float64Array<ArrayBufferLike> = new Float64Array(0);
  threshold = 0;

  constructor(
    readonly method: string,
    readonly neighbors: number,
    readonly trees: number,
    readonly contamination: number,
  ) {}

  fit(x: Matrix): this {
    if (this.method === 'lof') this.scores = localOutlierFactor(x, this.neighbors).scores;
    else if (this.method === 'iforest')
      this.scores = new IsolationForest({ trees: this.trees, seed: 1 }).fit(x).scores(x);
    else this.scores = new MahalanobisDetector().fit(x).scores(x);
    this.threshold = thresholdFor(this.scores, this.contamination);
    this.labels = Int32Array.from(this.scores, (s) => (s >= this.threshold ? 1 : 0));
    return this;
  }
}

export const detectorOf = (m: Clusterer): AnomalyDetector => m as AnomalyDetector;

/** The strays are the points whose hidden label is -1. */
export const strayMask = (y: Float64Array): Uint8Array =>
  Uint8Array.from(y, (v) => (v < 0 ? 1 : 0));

export const anomalyReport = (m: Clusterer, y: Float64Array) => {
  const d = detectorOf(m);
  const stray = strayMask(y);
  let tp = 0;
  let flagged = 0;
  let strays = 0;
  d.labels.forEach((l, i) => {
    if (l === 1) flagged++;
    if (stray[i]) strays++;
    if (l === 1 && stray[i]) tp++;
  });
  return {
    auc: rocAuc(d.scores, stray),
    flagged,
    strays,
    precision: flagged ? tp / flagged : 0,
    recall: strays ? tp / strays : 0,
  };
};

const exp = (e: ClusterExperiment): ClusterExperiment => e;

export const anomalySpec: ClusterLabSpec = {
  id: 'anomaly',
  create: (p) =>
    new AnomalyDetector(
      str(p, 'method'),
      num(p, 'neighbors'),
      num(p, 'trees'),
      num(p, 'contamination'),
    ),
  params,
  defaults: { method: 'lof', neighbors: 20, trees: 100, contamination: 0.05 },
  dataset: 'blobs',
  data: { n: 240, noise: 0.1, outliers: 12 },
  experiments: [
    exp({
      id: 'local-crowd',
      title: 'Local means local',
      text: 'On two crescents a point is odd if it is sparser than its own neighbours. With a small crowd (k = 5) LOF ranks almost every stray above every real point.',
      setup: { dataset: 'moons', noise: 0.1, outliers: 12 },
      params: { method: 'lof', neighbors: 5, contamination: 0.05 },
      claim: (run) => anomalyReport(run.model, run.data.y).auc > 0.98,
    }),
    exp({
      id: 'crowd-too-big',
      title: 'A crowd that is too big',
      text: 'Ask for the 50 nearest neighbours and “local” stops being local: points near a crescent are compared with the other crescent and the ranking degrades.',
      setup: { dataset: 'moons', noise: 0.1, outliers: 12 },
      params: { method: 'lof', neighbors: 50, contamination: 0.05 },
      claim: (run) => anomalyReport(run.model, run.data.y).auc < 0.92,
    }),
    exp({
      id: 'one-ellipse',
      title: 'Mahalanobis sees one ellipse',
      text: 'Mahalanobis distance assumes the normal data form a single elliptical cloud. Crescents are not, so the strays tucked into the bend look normal while real points at the tips look odd.',
      setup: { dataset: 'moons', noise: 0.1, outliers: 12 },
      params: { method: 'mahalanobis', contamination: 0.05 },
      claim: (run) => anomalyReport(run.model, run.data.y).auc < 0.93,
    }),
    exp({
      id: 'isolation',
      title: 'Isolation needs few cuts',
      text: 'The isolation forest never measures a distance: it cuts the plane at random and counts how few cuts it takes to fence a point in. Strays are fenced in quickly, even on the spirals.',
      setup: { dataset: 'spirals', noise: 0.1, outliers: 12 },
      params: { method: 'iforest', trees: 100, contamination: 0.05 },
      claim: (run) => anomalyReport(run.model, run.data.y).auc > 0.88,
    }),
    exp({
      id: 'always-finds',
      title: 'It always finds something',
      text: 'There are no strays here at all, yet the detector still flags 5% of the points: it ranks, and the “share flagged” setting decides where to cut. Scores need a human to decide what is truly odd.',
      setup: { dataset: 'blobs', outliers: 0 },
      params: { method: 'lof', neighbors: 20, contamination: 0.05 },
      claim: (run) => {
        const r = anomalyReport(run.model, run.data.y);
        return r.strays === 0 && r.flagged >= 10;
      },
    }),
    exp({
      id: 'flag-too-many',
      title: 'Flagging too many',
      text: 'Flag 30% of the points and nearly everything flagged is a perfectly normal point: recall is high but precision collapses.',
      setup: { dataset: 'blobs', outliers: 12 },
      params: { method: 'lof', neighbors: 20, contamination: 0.3 },
      claim: (run) => {
        const r = anomalyReport(run.model, run.data.y);
        return r.recall > 0.9 && r.precision < 0.2;
      },
    }),
  ],
};
