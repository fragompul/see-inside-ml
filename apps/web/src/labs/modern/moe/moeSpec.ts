import type { ParamSpec } from '@/shared/lab-kit';
import { DESIGN_LABEL, runMoeLab, type Design, type MoeResult, type MoeScenario } from './moeCore';

export const moeParams: ParamSpec[] = [
  {
    key: 'design',
    label: 'Model shown',
    kind: 'select',
    options: (Object.keys(DESIGN_LABEL) as Design[]).map((id) => ({
      value: id,
      label: DESIGN_LABEL[id],
    })),
  },
  {
    key: 'seed',
    label: 'Random start',
    kind: 'ladder',
    values: [1, 2, 3],
    format: (v) => `#${v}`,
  },
];

export interface MoeExperiment {
  id: string;
  title: string;
  text: string;
  scenario: Partial<MoeScenario>;
  claim: (run: MoeResult) => boolean;
}

const SEEDS = [1, 2, 3];
const mean = (v: number[]) => v.reduce((a, b) => a + b, 0) / v.length;
const across = (r: MoeResult, design: Design, pick: (x: MoeResult) => number) =>
  mean(SEEDS.map((seed) => pick(runMoeLab({ ...r.scenario, design, seed }))));
const sibling = (r: MoeResult, change: Partial<MoeScenario>) =>
  runMoeLab({ ...r.scenario, ...change });

export const moeExperiments: MoeExperiment[] = [
  {
    id: 'sparse',
    title: 'Four experts, but one input uses a third of the weights',
    text: 'The mixture holds 112 parameters, but a single input passes through the gate and one expert only: 37 of them. Capacity grows with the number of experts while the work per input does not.',
    scenario: { design: 'moe-1' },
    claim: (r) =>
      r.total['moe-1'] === 112 &&
      r.active['moe-1'] === 37 &&
      r.active['moe-1'] < r.total['moe-1'] / 2,
  },
  {
    id: 'collapse',
    title: 'Without a balance term the gate sends everything to one expert',
    text: 'From the second starting point, with nothing asking the gate to share, 94% of the inputs go to one expert and the other three are idle. Test error is 0.17 against 0.09 for the same model with the balance term.',
    scenario: { design: 'moe-unbalanced', seed: 2 },
    claim: (r) => {
      const balanced = sibling(r, { design: 'moe-1' });
      return (
        Math.max(...r.share) >= 0.9 &&
        Math.max(...balanced.share) < 0.4 &&
        r.mse['moe-unbalanced'] > 1.5 * balanced.mse['moe-1']
      );
    },
  },
  {
    id: 'balance',
    title: 'The balance term spreads the load',
    text: 'Averaged over three starting points the entropy of the load, which is 1.39 when the four experts are used equally, is 1.35 with the balance term and 0.47 without it.',
    scenario: { design: 'moe-1' },
    claim: (r) =>
      across(r, 'moe-1', (x) => x.entropy) > 1.25 &&
      across(r, 'moe-unbalanced', (x) => x.entropy) < 0.9,
  },
  {
    id: 'top-two',
    title: 'Two experts per input beat a bigger dense network',
    text: 'Letting the two best experts answer, and weighting them by the gate, gives the lowest error of all the designs: 0.077 on average against 0.098 for a single network with 24 hidden units, which uses 100 weights for every input where the mixture uses 62.',
    scenario: { design: 'moe-2' },
    claim: (r) =>
      across(r, 'moe-2', (x) => x.mse['moe-2']) <
        across(r, 'dense-big', (x) => x.mse['dense-big']) &&
      r.active['moe-2'] < r.active['dense-big'],
  },
  {
    id: 'specialise',
    title: 'Experts take regions',
    text: 'The four quadrants of the input have four different functions. With two experts running, more than 80% of every quadrant is routed to one favourite expert (purity 84%, 83%, 100% and 88%): the experts have divided the space.',
    scenario: { design: 'moe-2', seed: 1 },
    claim: (r) => r.purity.every((p) => p > 0.8) && mean(r.purity) > 0.85,
  },
];
