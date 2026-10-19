import {
  DEFAULT_GAN_VARIANT,
  MODE_CENTERS,
  MODE_STD,
  sampleReal,
  trainGanVariant,
  type GanKind,
  type GanVariantPoint,
} from '@sim/deep';
import { createRng } from '@sim/core';

export interface GanScenario {
  kind: GanKind;
  steps: number;
  seed: number;
}

export const DEFAULT_GAN_LAB: GanScenario = { kind: 'gan', steps: 1500, seed: 1 };

export const KIND_LABEL: Record<GanKind, string> = {
  gan: 'Standard GAN',
  'wgan-gp': 'WGAN-GP',
  cgan: 'Conditional GAN',
};

export interface GanLabRun {
  scenario: GanScenario;
  history: GanVariantPoint[];
  /** Generated points (n x 2) and the cluster each was asked for (conditional GAN). */
  samples: number[];
  labels: number[];
  /** A fresh sample of the real data, for drawing. */
  real: number[];
  modes: number;
  distance: number;
  nearData: number;
  classAccuracy: number | null;
  criticUpdates: number;
}

const cache = new Map<string, GanLabRun>();

export function runGanLab(s: GanScenario): GanLabRun {
  const key = JSON.stringify(s);
  const hit = cache.get(key);
  if (hit) return hit;
  const run = trainGanVariant(
    {
      ...DEFAULT_GAN_VARIANT,
      kind: s.kind,
      steps: s.steps,
      seed: s.seed,
      criticSteps: s.kind === 'wgan-gp' ? 5 : 1,
    },
    100,
  );
  const out: GanLabRun = {
    scenario: s,
    history: run.history,
    samples: Array.from(run.samples),
    labels: run.labels,
    real: Array.from(sampleReal('gaussians8', 400, createRng(7))),
    modes: run.modes,
    distance: run.distance,
    nearData: run.nearData,
    classAccuracy: Number.isNaN(run.classAccuracy) ? null : run.classAccuracy,
    criticUpdates: run.criticUpdates,
  };
  cache.set(key, out);
  return out;
}

export { MODE_CENTERS, MODE_STD };
