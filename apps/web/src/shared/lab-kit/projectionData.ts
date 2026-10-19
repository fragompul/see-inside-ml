import { CLASSIC_DATASETS } from '@sim/core/datasets';
import { generateDigits } from '@sim/deep';
import {
  MANIFOLD_SETS,
  Matrix,
  Standardizer,
  createRng,
  makeManifold,
  manifoldSetInfo,
  type ManifoldSet,
} from '@sim/core';

/** Everything the dimensionality-reduction labs need to know about a dataset. */
export interface ProjectionData {
  x: Matrix;
  /** Class index, or the position along the surface (0 … 1) for `continuous` colouring. */
  y: Float64Array;
  colour: 'continuous' | 'classes';
  classes: string[] | null;
  /** Number of original dimensions. */
  dimensions: number;
}

export type ProjectionDatasetId = ManifoldSet | 'iris' | 'wine' | 'digits';

export const PROJECTION_DATASETS: readonly ProjectionDatasetId[] = [
  ...MANIFOLD_SETS,
  'iris',
  'wine',
  'digits',
];

export const projectionDatasetInfo: Record<
  ProjectionDatasetId,
  { label: string; note: string; real: boolean }
> = {
  ...(Object.fromEntries(
    MANIFOLD_SETS.map((k) => [
      k,
      { label: manifoldSetInfo[k].label, note: manifoldSetInfo[k].note, real: false },
    ]),
  ) as Record<ManifoldSet, { label: string; note: string; real: boolean }>),
  iris: {
    label: 'Iris',
    note: 'Fisher’s irises: 150 flowers, 4 measurements, 3 species (one is easy to tell apart, two overlap).',
    real: true,
  },
  wine: {
    label: 'Wine',
    note: 'Chemical analysis of 178 Italian wines from 3 growers, 13 measurements (standardised).',
    real: true,
  },
  digits: {
    label: 'Digits',
    note: 'Little 16 × 16 pictures of handwritten digits: every point lives in 256 dimensions.',
    real: false,
  },
};

export interface ProjectionSetup {
  dataset: ProjectionDatasetId;
  n: number;
  noise: number;
  seed: number;
}

export const DEFAULT_PROJECTION_SETUP: ProjectionSetup = {
  dataset: 'swiss',
  n: 150,
  noise: 0.1,
  seed: 4,
};

export function buildProjectionData(setup: ProjectionSetup): ProjectionData {
  const id = setup.dataset;
  if (id === 'iris' || id === 'wine') {
    const ds = CLASSIC_DATASETS[id];
    const raw = Matrix.from(ds.rows);
    const x = id === 'wine' ? new Standardizer().fit(raw).transform(raw) : raw;
    return {
      x,
      y: Float64Array.from(ds.target),
      colour: 'classes',
      classes: ds.classes,
      dimensions: x.cols,
    };
  }
  if (id === 'digits') {
    const perClass = Math.max(3, Math.round(setup.n / 10));
    const set = generateDigits(perClass, createRng(setup.seed));
    const n = set.count;
    return {
      x: new Matrix(n, 256, Float64Array.from(set.images)),
      y: Float64Array.from(set.labels),
      colour: 'classes',
      classes: Array.from({ length: 10 }, (_, d) => String(d)),
      dimensions: 256,
    };
  }
  const t = makeManifold(id, setup.n, setup.noise, createRng(setup.seed));
  return {
    x: t.x,
    y: t.y,
    colour: manifoldSetInfo[id].colour,
    classes: t.classes ?? null,
    dimensions: t.x.cols,
  };
}
