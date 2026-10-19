import type { ParamSpec } from '@/shared/lab-kit';
import {
  AUGMENTATIONS,
  runAugment,
  type AugmentId,
  type AugmentRun,
  type AugmentScenario,
} from './augmentCore';

export const augmentParams: ParamSpec[] = [
  {
    key: 'augmentation',
    label: 'Augmentation',
    kind: 'select',
    options: (Object.keys(AUGMENTATIONS) as AugmentId[]).map((id) => ({
      value: id,
      label: (AUGMENTATIONS[id] as { label: string }).label,
    })),
  },
  {
    key: 'perClass',
    label: 'Real examples per digit',
    kind: 'ladder',
    values: [3, 10],
    format: (v) => String(v),
  },
  {
    key: 'copies',
    label: 'Random copies of each image',
    kind: 'ladder',
    values: [4, 8],
    format: (v) => String(v),
  },
];

export interface AugmentExperiment {
  id: string;
  title: string;
  text: string;
  scenario: Partial<AugmentScenario>;
  claim: (run: AugmentRun) => boolean;
}

const same = (r: AugmentRun, change: Partial<AugmentScenario>) =>
  runAugment({ ...r.scenario, ...change });

export const augmentExperiments: AugmentExperiment[] = [
  {
    id: 'memorise',
    title: 'Thirty pictures are learned by heart',
    text: 'With three examples of each digit the network gets every training picture right and still misses a quarter of new, similar ones; on new digits that are shifted, tilted and noisy it gets fewer than one in three.',
    scenario: { augmentation: 'none', perClass: 3 },
    claim: (r) => r.train === 1 && r.clean < 0.8 && r.wild < 0.35,
  },
  {
    id: 'shift',
    title: 'Shifted copies teach what a shift means',
    text: 'Adding eight randomly shifted copies of every picture, with no new real data, lifts accuracy on ordinary new digits from about 75% to 87%, and on shifted ones from 29% to 49%.',
    scenario: { augmentation: 'shift', perClass: 3 },
    claim: (r) => {
      const base = same(r, { augmentation: 'none' });
      return r.clean > base.clean + 0.1 && r.wild > base.wild + 0.18;
    },
  },
  {
    id: 'match',
    title: 'Augment the changes you will meet',
    text: 'A network that reads raw pixels has no built-in idea of shift, so only shifted copies help it with shifted digits: with ten examples per digit it goes from 32% to 68%, while rotated or noisy copies barely move it (under 2 points).',
    scenario: { augmentation: 'shift', perClass: 10 },
    claim: (r) => {
      const base = same(r, { augmentation: 'none' });
      return (
        r.wild > base.wild + 0.3 &&
        same(r, { augmentation: 'rotate' }).wild < base.wild + 0.05 &&
        same(r, { augmentation: 'noise' }).wild < base.wild + 0.05
      );
    },
  },
  {
    id: 'rotate',
    title: 'Rotations help with the variety you already have',
    text: 'Handwriting really does come tilted, so rotated copies make the network better at ordinary new digits (from 75% to 81% here) without helping it with a shifted camera.',
    scenario: { augmentation: 'rotate', perClass: 3 },
    claim: (r) => {
      const base = same(r, { augmentation: 'none' });
      return r.clean > base.clean + 0.04 && r.wild < base.wild + 0.05;
    },
  },
  {
    id: 'not-enough',
    title: 'Copies are not a substitute for data',
    text: 'Three real examples with shifted copies reach 87% on new digits; ten real examples with no augmentation reach 96%. Augmentation adds variety around what you have, never information about what you have not seen.',
    scenario: { augmentation: 'shift', perClass: 3 },
    claim: (r) => r.clean < same(r, { augmentation: 'none', perClass: 10 }).clean - 0.05,
  },
];
