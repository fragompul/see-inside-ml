import type { ParamSpec } from '@/shared/lab-kit';
import { BUDGETS, runRobustness, type RobustScenario, type RobustnessResult } from './robustCore';

export const robustParams: ParamSpec[] = [
  {
    key: 'trainEpsilon',
    label: 'Attack budget used in training',
    kind: 'ladder',
    values: [0, 0.1, 0.2],
    format: (v) => (v === 0 ? 'none' : String(v)),
    hint: 'Per-pixel change allowed, on a 0 to 1 scale.',
  },
  {
    key: 'trainMethod',
    label: 'Attack used in training',
    kind: 'segmented',
    options: [
      { value: 'fgsm', label: 'FGSM (one step)' },
      { value: 'pgd', label: 'PGD (iterated)' },
    ],
    showIf: (v) => Number(v.trainEpsilon) > 0,
  },
  {
    key: 'testMethod',
    label: 'Attack used to test',
    kind: 'segmented',
    options: [
      { value: 'fgsm', label: 'FGSM (one step)' },
      { value: 'pgd', label: 'PGD (iterated)' },
    ],
  },
  {
    key: 'shownEpsilon',
    label: 'Budget of the pictures shown',
    kind: 'ladder',
    values: BUDGETS.slice(1),
    format: (v) => String(v),
  },
];

export interface RobustExperiment {
  id: string;
  title: string;
  text: string;
  scenario: Partial<RobustScenario>;
  claim: (run: RobustnessResult) => boolean;
}

const at = (curve: number[], budget: number) => curve[BUDGETS.indexOf(budget)] as number;
const sibling = (r: RobustnessResult, change: Partial<RobustScenario>) =>
  runRobustness({ ...r.scenario, ...change });

export const robustExperiments: RobustExperiment[] = [
  {
    id: 'fragile',
    title: 'An ordinary network is fragile',
    text: 'Changing every pixel by at most 0.1 (a tenth of the range) with a single gradient step lowers the accuracy of an ordinary digit network from 98% to 58%. Allowing 0.2, and iterating, drives it to zero.',
    scenario: { trainEpsilon: 0, testMethod: 'fgsm' },
    claim: (r) =>
      r.chosen.clean > 0.95 && at(r.chosen.fgsm, 0.1) < 0.65 && at(r.chosen.pgd, 0.2) < 0.02,
  },
  {
    id: 'iterate',
    title: 'Iterating makes the attack stronger',
    text: 'At the same budget of 0.1, repeated small steps (PGD) leave the ordinary network at 38% where a single step (FGSM) left it at 58%. A defence tested only against the weaker attack looks better than it is.',
    scenario: { trainEpsilon: 0 },
    claim: (r) => at(r.chosen.pgd, 0.1) < at(r.chosen.fgsm, 0.1) - 0.1,
  },
  {
    id: 'train',
    title: 'Training against attacks helps',
    text: 'Showing the network attacked copies of its training digits (budget 0.2, one step) raises its accuracy under a PGD attack of 0.1 from 38% to 73%, and under FGSM from 58% to 81%.',
    scenario: { trainEpsilon: 0.2, trainMethod: 'fgsm', testMethod: 'pgd' },
    claim: (r) =>
      at(r.chosen.pgd, 0.1) > at(r.ordinary.pgd, 0.1) + 0.25 &&
      at(r.chosen.fgsm, 0.1) > at(r.ordinary.fgsm, 0.1) + 0.15,
  },
  {
    id: 'cost',
    title: 'The cost here is small',
    text: 'On untouched digits the adversarially trained network is right 97% of the time against 98% for the ordinary one. (On harder problems the gap is much larger; this one is easy.)',
    scenario: { trainEpsilon: 0.2, trainMethod: 'fgsm' },
    claim: (r) => r.ordinary.clean - r.chosen.clean < 0.03 && r.chosen.clean > 0.9,
  },
  {
    id: 'budget',
    title: 'Robustness has a limit',
    text: 'Trained for a budget of 0.2, the network still holds 40% under an FGSM attack of 0.2 (the ordinary one: 5%), but at 0.3 it is down to 3%. The protection reaches roughly as far as the attacks it saw.',
    scenario: { trainEpsilon: 0.2, trainMethod: 'fgsm', testMethod: 'fgsm' },
    claim: (r) =>
      at(r.chosen.fgsm, 0.2) > at(r.ordinary.fgsm, 0.2) + 0.25 && at(r.chosen.fgsm, 0.3) < 0.1,
  },
  {
    id: 'stronger-training',
    title: 'Iterated training is the strongest defence',
    text: 'Training against PGD at budget 0.1 gives 93% under an FGSM attack of 0.05 and 76% under PGD at 0.1, a little better than training on single steps at the same budget (91% and 72%).',
    scenario: { trainEpsilon: 0.1, trainMethod: 'pgd', testMethod: 'pgd' },
    claim: (r) => {
      const single = sibling(r, { trainMethod: 'fgsm' });
      return at(r.chosen.pgd, 0.1) > at(single.chosen.pgd, 0.1);
    },
  },
];
