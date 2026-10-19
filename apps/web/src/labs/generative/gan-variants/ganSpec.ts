import type { GanKind } from '@sim/deep';
import type { ParamSpec } from '@/shared/lab-kit';
import { KIND_LABEL, runGanLab, type GanLabRun, type GanScenario } from './ganCore';

export const ganParams: ParamSpec[] = [
  {
    key: 'kind',
    label: 'Recipe',
    kind: 'select',
    options: (Object.keys(KIND_LABEL) as GanKind[]).map((id) => ({
      value: id,
      label: KIND_LABEL[id],
    })),
  },
  {
    key: 'steps',
    label: 'Generator updates',
    kind: 'ladder',
    values: [500, 1500, 3000],
    format: (v) => String(v),
  },
  {
    key: 'seed',
    label: 'Random start',
    kind: 'ladder',
    values: [1, 2, 3],
    format: (v) => `#${v}`,
    hint: 'GANs are sensitive to where they start.',
  },
];

export interface GanExperiment {
  id: string;
  title: string;
  text: string;
  scenario: Partial<GanScenario>;
  claim: (run: GanLabRun) => boolean;
}

const sibling = (r: GanLabRun, change: Partial<GanScenario>) =>
  runGanLab({ ...r.scenario, ...change });

export const ganExperiments: GanExperiment[] = [
  {
    id: 'hopping',
    title: 'The standard GAN hops between modes',
    text: 'Early on the generator covers only a few of the eight clusters, then another few: each time the discriminator catches up with where it is, it moves on. After 1,500 updates it has found all eight.',
    scenario: { kind: 'gan', steps: 1500, seed: 1 },
    claim: (r) => {
      const modes = r.history.map((h) => h.modes);
      return (
        (modes[2] as number) <= 7 &&
        Math.min(...modes.slice(0, 5)) <= 4 &&
        r.modes === 8 &&
        modes.some((m, i) => i > 0 && m < (modes[i - 1] as number))
      );
    },
  },
  {
    id: 'conditional',
    title: 'Tell it the cluster, and it goes there',
    text: 'Given which cluster to draw as well as the noise, the conditional GAN puts every point in the requested cluster (100% here) and every point lands on real data, while the standard GAN at the same stage leaves more than half of its points between clusters.',
    scenario: { kind: 'cgan', steps: 1500, seed: 1 },
    claim: (r) =>
      (r.classAccuracy as number) >= 0.95 &&
      r.nearData >= 0.95 &&
      sibling(r, { kind: 'gan' }).nearData <= 0.6,
  },
  {
    id: 'wgan',
    title: 'WGAN-GP can match the data better',
    text: 'The critic scores points instead of labelling them, and a gradient penalty keeps its slope near 1. On two of three starting points it ends much closer to the real distribution than the standard GAN (sliced Wasserstein distance 0.17 against 0.31, and 0.13 against 0.35).',
    scenario: { kind: 'wgan-gp', steps: 1500, seed: 1 },
    claim: (r) =>
      r.distance < 0.6 * sibling(r, { kind: 'gan' }).distance &&
      sibling(r, { seed: 3 }).distance < 0.6 * sibling(r, { kind: 'gan', seed: 3 }).distance,
  },
  {
    id: 'unlucky',
    title: 'But not from every start',
    text: 'From the second starting point the WGAN-GP covers only four of the eight clusters after the same 1,500 updates, where the standard GAN covered all eight. Neither recipe removes the dependence on the random start.',
    scenario: { kind: 'wgan-gp', steps: 1500, seed: 2 },
    claim: (r) => r.modes <= 5 && sibling(r, { kind: 'gan' }).modes === 8,
  },
  {
    id: 'cost',
    title: 'The critic needs five updates per step',
    text: 'Each generator update is preceded by five critic updates, so the same 1,500 steps cost five times as many critic updates (7,500 against 1,500) and about four times as long.',
    scenario: { kind: 'wgan-gp', steps: 1500, seed: 1 },
    claim: (r) => r.criticUpdates === 7500 && sibling(r, { kind: 'gan' }).criticUpdates === 1500,
  },
];
