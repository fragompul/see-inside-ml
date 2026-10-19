import { SOM, somSteps, topographicError, type SomOptions } from '@sim/classic';
import { Matrix } from '@sim/core';
import {
  num,
  type EmbedModel,
  type ParamSpec,
  type ParamValues,
  type ProjectorExperiment,
  type ProjectorLabSpec,
} from '@/shared/lab-kit';

const params: ParamSpec[] = [
  {
    key: 'side',
    label: 'Grid size',
    kind: 'ladder',
    values: [2, 4, 6, 8, 12],
    format: (v) => `${v} × ${v}`,
    hint: 'Units along each side of the square map.',
  },
  {
    key: 'epochs',
    label: 'Epochs',
    kind: 'slider',
    min: 1,
    max: 100,
    step: 1,
    hint: 'Passes over the data.',
  },
  {
    key: 'learningRate',
    label: 'Starting learning rate',
    kind: 'slider',
    min: 0.05,
    max: 1,
    step: 0.05,
    format: (v) => v.toFixed(2),
  },
  {
    key: 'radius',
    label: 'Starting neighbourhood radius',
    kind: 'slider',
    min: 0.5,
    max: 10,
    step: 0.5,
    format: (v) => v.toFixed(1),
    hint: 'How far (in grid cells) the pull on a winner spreads; it shrinks during training.',
  },
];

export const somOptions = (p: ParamValues): SomOptions => ({
  rows: num(p, 'side'),
  cols: num(p, 'side'),
  epochs: num(p, 'epochs'),
  learningRate: num(p, 'learningRate'),
  radius: num(p, 'radius'),
  seed: 3,
});

/** Where every sample sits on the map: its best-matching cell, nudged by a fixed jitter so that samples sharing a cell fan out. */
export function mapPositions(weights: Matrix, x: Matrix, side: number): Matrix {
  const cells = new SOM({ rows: side, cols: side }); // only used for assign()
  cells.weights = weights;
  const out = new Matrix(x.rows, 2);
  cells.assign(x).forEach(({ row, col }, i) => {
    const a = Math.sin(i * 12.9898) * 43758.5453;
    const b = Math.sin(i * 78.233) * 12543.2231;
    out.set(i, 0, col + 0.5 + 0.7 * (a - Math.floor(a) - 0.5));
    out.set(i, 1, row + 0.5 + 0.7 * (b - Math.floor(b) - 0.5));
  });
  return out;
}

export const fitSom = (p: ParamValues, x: Matrix) => new SOM(somOptions(p)).fit(x);

export const somModel = (p: ParamValues): EmbedModel => ({
  *run(x) {
    const opts = somOptions(p);
    const epochs = opts.epochs ?? 40;
    for (const s of somSteps(x, opts)) {
      yield {
        embedding: mapPositions(s.weights, x, opts.rows),
        info: `epoch ${s.epoch} of ${epochs}, quantization error ${s.quantizationError.toFixed(3)}, topographic error ${(topographicError(s.weights, x, opts.cols) * 100).toFixed(0)}%`,
        progress: s.epoch / epochs,
      };
    }
  },
});

export const somSpec: ProjectorLabSpec = {
  id: 'som',
  create: somModel,
  params,
  defaults: { side: 8, epochs: 40, learningRate: 0.5, radius: 4 },
  dataset: 'swiss',
  datasets: ['swiss', 'scurve', 'sphere', 'clusters', 'iris', 'wine', 'digits'],
  data: { n: 300, noise: 0.05 },
  experiments: [
    {
      id: 'unroll',
      title: 'Unrolling the swiss roll',
      text: 'A flat 8 × 8 sheet of units starts tangled up in the data and, as the neighbourhood shrinks, spreads over the rolled surface; the picture on the left keeps nearly all true neighbours.',
      setup: { dataset: 'swiss', n: 300, noise: 0.05 },
      params: { side: 8, epochs: 40, radius: 4 },
      claim: (run) => run.scores.trustworthiness > 0.97 && topographic(run, 8, 40, 4) < 0.1,
    },
    {
      id: 'iris-map',
      title: 'Iris on a 12 × 12 map',
      text: 'With a roomy grid almost every flower gets its own corner of the map and the three species settle into neighbouring regions: a large share of each flower’s nearest neighbours survives.',
      setup: { dataset: 'iris' },
      params: { side: 12, epochs: 40, radius: 4 },
      claim: (run) => run.scores.preservation > 0.55 && run.scores.trustworthiness > 0.97,
    },
    {
      id: 'tiny-map',
      title: 'A map that is too small',
      text: 'Four units cannot describe a surface: every unit has to speak for a whole region, so neighbouring points of the roll are lumped together or split apart and the layout is nearly useless.',
      setup: { dataset: 'swiss', n: 300, noise: 0.05 },
      params: { side: 2, epochs: 40, radius: 4 },
      claim: (run) => run.scores.trustworthiness < 0.85 && run.scores.preservation < 0.15,
    },
    {
      id: 'small-iris',
      title: 'Lumping flowers together',
      text: 'On a 4 × 4 map each unit stands for around nine flowers, so the fine neighbourhoods inside a species are lost even though the species themselves stay apart.',
      setup: { dataset: 'iris' },
      params: { side: 4, epochs: 40, radius: 2 },
      claim: (run) => run.scores.preservation < 0.35 && run.scores.trustworthiness > 0.9,
    },
    {
      id: 'no-neighbourhood',
      title: 'No neighbourhood, no order',
      text: 'With a radius of half a cell only the winner moves. Units become independent prototypes (like k-means), fit the data about as well, but the grid stops meaning anything: neighbouring units are no longer similar (high topographic error).',
      setup: { dataset: 'swiss', n: 300, noise: 0.05 },
      params: { side: 8, epochs: 40, radius: 0.5 },
      claim: (run) => topographic(run, 8, 40, 0.5) > 0.25,
    },
    {
      id: 'one-epoch',
      title: 'One pass is not enough',
      text: 'After a single epoch the units are still far from the data: the quantization error is about twice what a trained map reaches.',
      setup: { dataset: 'swiss', n: 300, noise: 0.05 },
      params: { side: 8, epochs: 1, radius: 4 },
      claim: (run) =>
        fitSom({ ...somSpec.defaults, side: 8, epochs: 1, radius: 4 }, run.data.x).history.at(-1)! >
        3.5,
    },
    {
      id: 'digits-map',
      title: 'Digits on a sheet',
      text: 'Pictures of handwritten digits live in 256 dimensions; the map flattens them to a grid where similar shapes sit together.',
      setup: { dataset: 'digits', n: 200 },
      params: { side: 8, epochs: 40, radius: 4 },
      claim: (run) => run.scores.trustworthiness > 0.93,
    },
  ] satisfies ProjectorExperiment[],
};

function topographic(
  run: { data: { x: Matrix } },
  side: number,
  epochs: number,
  radius: number,
): number {
  const m = fitSom({ ...somSpec.defaults, side, epochs, radius }, run.data.x);
  return topographicError(m.weights, run.data.x, side);
}
