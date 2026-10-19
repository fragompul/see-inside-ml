import { fitPowerLaw } from '@sim/core';
import type { ParamSpec } from '@/shared/lab-kit';
import { DATA, FEATURES, type ScalingRun, type ScalingScenario } from './scalingCore';

export const scalingParams: ParamSpec[] = [
  {
    key: 'noise',
    label: 'Noise in the labels',
    kind: 'ladder',
    values: [0.1, 0.3, 0.5],
    format: (v) => String(v),
    hint: 'Standard deviation. Noise sets a floor no model can pass.',
  },
  {
    key: 'seed',
    label: 'Random target',
    kind: 'ladder',
    values: [6, 7, 8],
    format: (v) => `#${v - 5}`,
    hint: 'A different random function to learn.',
  },
];

export interface ScalingExperiment {
  id: string;
  title: string;
  text: string;
  scenario: Partial<ScalingScenario>;
  claim: (run: ScalingRun) => boolean;
}

const nonDecreasing = (v: number[]) => v.every((x, i) => i === 0 || x >= (v[i - 1] as number));

export const scalingExperiments: ScalingExperiment[] = [
  {
    id: 'data-law',
    title: 'More data: a straight line on log-log paper',
    text: 'For the largest model the loss above the noise floor falls as a power of the number of examples, with an exponent of about 0.3: ten times the data cuts the excess loss roughly in half. Plotted on logarithmic axes it is a straight line.',
    scenario: { noise: 0.1, seed: 6 },
    claim: (r) => r.dataLaw.r2 > 0.9 && r.dataLaw.alpha > 0.2 && r.dataLaw.alpha < 0.4,
  },
  {
    id: 'size-law',
    title: 'Bigger models: another straight line',
    text: 'With the most data, the excess loss also falls as a power of the number of features, with an exponent of about 0.4.',
    scenario: { noise: 0.1, seed: 6 },
    claim: (r) => r.featureLaw.r2 > 0.95 && r.featureLaw.alpha > 0.3 && r.featureLaw.alpha < 0.6,
  },
  {
    id: 'both',
    title: 'A small model cannot use more data',
    text: 'From 64 to 4,096 examples the loss of the largest model falls to a third; the loss of the smallest falls by only 10%. Past a point the model, not the data, is the limit, and the other way round.',
    scenario: { noise: 0.1, seed: 6 },
    claim: (r) => {
      const big = r.loss[FEATURES.length - 1] as number[];
      const small = r.loss[0] as number[];
      const ratio = (row: number[]) => (row[0] as number) / (row[row.length - 1] as number);
      return ratio(big) > 2.5 && ratio(small) < 1.3;
    },
  },
  {
    id: 'budget',
    title: 'Spending a budget: grow both together',
    text: 'If the cost is features times examples, the best way to spend a larger budget is never to grow only one: the winning model gets bigger and the winning data set gets bigger at every step, from 32 features and 128 examples up to 256 and 4,096.',
    scenario: { noise: 0.1, seed: 6 },
    claim: (r) =>
      nonDecreasing(r.frontier.map((f) => f.features)) &&
      nonDecreasing(r.frontier.map((f) => f.data)) &&
      (r.frontier[r.frontier.length - 1] as { features: number }).features >=
        4 * (r.frontier[0] as { features: number }).features,
  },
  {
    id: 'extrapolate',
    title: 'Predicting a bigger run from small ones',
    text: 'Fit the law to the first four data sizes (64 to 512) and use it to predict the loss at 4,096. The prediction lands within a third of the real value, though the flattening of the real curve makes it too optimistic: scaling laws are a guide for planning, not a promise.',
    scenario: { noise: 0.1, seed: 6 },
    claim: (r) => {
      const row = r.loss[FEATURES.length - 1] as number[];
      const fit = fitPowerLaw(DATA.slice(0, 4), row.slice(0, 4), r.floor);
      const actual = row[row.length - 1] as number;
      const guess = fit.predict(DATA[DATA.length - 1] as number);
      return guess < actual && Math.abs(guess / actual - 1) < 0.35;
    },
  },
  {
    id: 'floor',
    title: 'Noise is a floor',
    text: 'Labels with noise of 0.5 cannot be predicted better than a loss of 0.25 however large the model or the data set. No cell here gets below it, and the biggest model, at 0.42, is still two thirds above it.',
    scenario: { noise: 0.5, seed: 6 },
    claim: (r) => {
      const all = r.loss.flat();
      return Math.min(...all) >= r.floor && Math.min(...all) > 1.5 * r.floor;
    },
  },
];
