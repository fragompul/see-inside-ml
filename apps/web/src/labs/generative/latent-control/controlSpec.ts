import type { ParamSpec } from '@/shared/lab-kit';
import {
  runControl,
  type ConditionalResult,
  type ControlResult,
  type ControlScenario,
  type VqResult,
} from './controlCore';

export const controlParams: ParamSpec[] = [
  {
    key: 'mode',
    label: 'Model',
    kind: 'select',
    options: [
      { value: 'conditional', label: 'Conditional VAE (tell it the digit)' },
      { value: 'vq', label: 'VQ-VAE (a codebook of prototypes)' },
    ],
  },
  {
    key: 'conditioned',
    label: 'Given the digit class',
    kind: 'switch',
    showIf: (v) => v.mode === 'conditional',
  },
  {
    key: 'beta',
    label: 'KL weight β',
    kind: 'ladder',
    values: [1, 4],
    format: (v) => String(v),
    showIf: (v) => v.mode === 'conditional',
  },
  {
    key: 'entries',
    label: 'Codebook entries',
    kind: 'ladder',
    values: [4, 10, 20],
    format: (v) => String(v),
    showIf: (v) => v.mode === 'vq',
  },
  {
    key: 'revive',
    label: 'Revive unused entries',
    kind: 'switch',
    showIf: (v) => v.mode === 'vq',
  },
];

export interface ControlExperiment {
  id: string;
  title: string;
  text: string;
  scenario: Partial<ControlScenario>;
  claim: (run: ControlResult) => boolean;
}

const cond = (r: ControlResult): ConditionalResult => {
  if (r.kind !== 'conditional') throw new Error('not a conditional run');
  return r;
};
const vq = (r: ControlResult): VqResult => {
  if (r.kind !== 'vq') throw new Error('not a VQ run');
  return r;
};
const sibling = (r: ControlResult, change: Partial<ControlScenario>) =>
  runControl({ ...r.scenario, ...change });
const lastRecon = (v: VqResult) => v.testRecon[v.testRecon.length - 1] as number;

export const controlExperiments: ControlExperiment[] = [
  {
    id: 'control',
    title: 'Tell it the digit, and it draws that digit',
    text: 'Given the digit class as well as the code, the model draws the requested digit every time (an independent judge reads 100% of them right). A model that was never told the class draws whatever the code suggests: asked for a given digit it is right about one time in ten.',
    scenario: { mode: 'conditional', conditioned: true, beta: 1 },
    claim: (r) =>
      cond(r).control >= 0.95 && cond(sibling(r, { conditioned: false })).control <= 0.15,
  },
  {
    id: 'free-code',
    title: 'The code is freed from carrying the digit',
    text: 'When the class is supplied, the two numbers of the code no longer have to say which digit this is. They are spent on style, so images are rebuilt better (a loss of 64 against 72) with less information in the code (KL 2.0 against 2.9).',
    scenario: { mode: 'conditional', conditioned: true, beta: 1 },
    claim: (r) => {
      const other = cond(sibling(r, { conditioned: false }));
      return cond(r).testRecon + 5 < other.testRecon && cond(r).testKl < other.testKl;
    },
  },
  {
    id: 'beta',
    title: 'A heavy KL penalty costs the class nothing',
    text: 'Raising the weight of the KL term from 1 to 4 squeezes the information in the code from 2.0 to 0.4, yet the model still draws the requested digit every time: the class never went through the code.',
    scenario: { mode: 'conditional', conditioned: true, beta: 4 },
    claim: (r) =>
      cond(r).testKl < 0.3 * cond(sibling(r, { beta: 1 })).testKl && cond(r).control >= 0.95,
  },
  {
    id: 'collapse',
    title: 'Codebook entries die',
    text: 'Left alone, a codebook loses entries: with ten, three are never chosen by any image, and with four, two of them are almost never used and one takes nearly every image. Moving unused entries onto real codes during training brings every entry back into use.',
    scenario: { mode: 'vq', entries: 10, revive: false },
    claim: (r) =>
      vq(r).deadEntries >= 3 &&
      vq(sibling(r, { revive: true })).deadEntries === 0 &&
      vq(sibling(r, { entries: 4 })).deadEntries >= 2,
  },
  {
    id: 'size',
    title: 'More entries, finer prototypes',
    text: 'With 4 entries an image is rebuilt from one of four prototypes and the groups mix many digits (28% pure); with 20 the groups are 63% pure and the error is lower.',
    scenario: { mode: 'vq', entries: 20, revive: true },
    claim: (r) => {
      const small = vq(sibling(r, { entries: 4 }));
      const mid = vq(sibling(r, { entries: 10 }));
      return (
        lastRecon(vq(r)) < lastRecon(mid) &&
        lastRecon(mid) < lastRecon(small) &&
        vq(r).purity > mid.purity &&
        mid.purity > small.purity
      );
    },
  },
  {
    id: 'prototypes',
    title: 'Each entry is a prototype of a digit',
    text: 'The picture decoded from a codebook entry, with no image to copy from, is read by the judge as the digit most of that entry’s images show: 92% of the time with ten entries. The codebook has learned something like ten typical digits.',
    scenario: { mode: 'vq', entries: 10, revive: true },
    claim: (r) => vq(r).prototypeAgreement >= 0.9,
  },
];
