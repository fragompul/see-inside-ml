import { CLASSIC_DATASETS } from '@sim/core/datasets';
import { Standardizer, createRng, makeMultiple, makeTable, type Table } from '@sim/core';
import { splitData, type DataSetup, type SplitData } from './pipeline';

/**
 * Multi-feature regression data for the labs with named features: a synthetic table with
 * correlated features and a known truth, or the (standardised) Diabetes dataset.
 */
export type TabularSource = 'synthetic' | 'diabetes';

export interface TabularSetup extends DataSetup {
  source: TabularSource;
  /** Correlation between the first two synthetic features. */
  correlation: number;
}

export const TABULAR_SOURCES: { value: TabularSource; label: string; note: string }[] = [
  {
    value: 'synthetic',
    label: 'Synthetic',
    note: 'y = 1.5 + 2·x₁ − 1·x₂ + 0.4·x₃ + noise, with x₂ correlated to x₁. The truth is known, so you can see how close each fit gets.',
  },
  {
    value: 'diabetes',
    label: 'Diabetes',
    note: 'Disease progression of 442 patients from ten baseline measurements (features standardised).',
  },
];

/** The coefficients that generated the synthetic table (intercept first). */
export const SYNTHETIC_TRUTH = { intercept: 1.5, coef: [2, -1, 0.4] } as const;

export const DEFAULT_TABULAR: TabularSetup = {
  source: 'synthetic',
  n: 120,
  noise: 0.5,
  correlation: 0.3,
  seed: 4,
  testFraction: 0.3,
};

export function tabularTable(setup: TabularSetup): Table {
  if (setup.source === 'diabetes') {
    const d = CLASSIC_DATASETS.diabetes;
    const raw = makeTable(d.rows, d.target, d.features);
    const z = new Standardizer().fit(raw.x);
    return { ...raw, x: z.transform(raw.x) };
  }
  return makeMultiple(setup.n, setup.noise, setup.correlation, createRng(setup.seed));
}

export function buildTabular(setup: TabularSetup): SplitData {
  return splitData(tabularTable(setup), setup, false);
}
