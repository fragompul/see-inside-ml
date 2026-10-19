import type { ParamSpec } from '@/shared/lab-kit';
import { ITEMS, PLANTED, type AssociationRun, type AssociationScenario } from './associationCore';

export const associationParams: ParamSpec[] = [
  {
    key: 'baskets',
    label: 'Shopping baskets',
    kind: 'ladder',
    values: [60, 200, 600, 2000, 5000],
    format: (v) => String(v),
  },
  {
    key: 'minSupport',
    label: 'Minimum support',
    kind: 'ladder',
    values: [0.01, 0.02, 0.05, 0.1, 0.2, 0.3],
    format: (v) => `${Math.round(v * 100)}%`,
    hint: 'A combination must appear in at least this share of the baskets.',
  },
  {
    key: 'minConfidence',
    label: 'Minimum confidence',
    kind: 'slider',
    min: 0,
    max: 0.95,
    step: 0.05,
    format: (v) => `${Math.round(v * 100)}%`,
    hint: 'Of the baskets with the left side, the share that also has the right side.',
  },
  {
    key: 'maxSize',
    label: 'Largest combination',
    kind: 'segmented',
    options: [
      { value: '2', label: 'Pairs' },
      { value: '3', label: 'Up to 3 items' },
      { value: '4', label: 'Up to 4 items' },
    ],
  },
  {
    key: 'sortBy',
    label: 'Rank the rules by',
    kind: 'segmented',
    options: [
      { value: 'lift', label: 'Lift' },
      { value: 'confidence', label: 'Confidence' },
      { value: 'support', label: 'Support' },
    ],
  },
];

export interface AssociationExperiment {
  id: string;
  title: string;
  text: string;
  scenario: Partial<AssociationScenario>;
  claim: (run: AssociationRun) => boolean;
}

const isPlanted = (r: { antecedent: number[]; consequent: number[] }) =>
  r.antecedent.length === 1 &&
  r.consequent.length === 1 &&
  PLANTED.some(
    ([a, c]) =>
      (r.antecedent[0] === a && r.consequent[0] === c) ||
      (r.antecedent[0] === c && r.consequent[0] === a),
  );

export const associationExperiments: AssociationExperiment[] = [
  {
    id: 'planted',
    title: 'Finding the real pairs',
    text: 'The baskets hide four true habits: bread with butter, beer with diapers, chips with salsa, wine with cheese. Ranked by lift, the strongest rules are exactly those pairs, and all four are found.',
    scenario: {},
    claim: (r) => r.recovered === 4 && r.ranked.slice(0, 4).every(isPlanted) && r.bestLift > 3,
  },
  {
    id: 'popular-trap',
    title: 'Confidence can mislead',
    text: 'Milk is in 60% of baskets, so “bread → milk” has a respectable confidence above 55%. But bread buyers are not more likely to buy milk than anyone else: the lift is below 1. Confidence alone rewards popular items; lift corrects for them.',
    scenario: { sortBy: 'support' },
    claim: (r) =>
      r.frequency[2]! > 0.55 &&
      r.rules.some(
        (x) =>
          x.consequent.length === 1 && x.consequent[0] === 2 && x.confidence > 0.55 && x.lift < 1,
      ),
  },
  {
    id: 'strict-support',
    title: 'Support set too high',
    text: 'Requiring a combination to appear in 30% of baskets throws away every habit that is real but not mainstream: only bread and butter survive, and the other three habits are lost.',
    scenario: { minSupport: 0.3 },
    claim: (r) => r.rules.length <= 2 && r.recovered === 1,
  },
  {
    id: 'lenient',
    title: 'Everything is a rule',
    text: 'With low support and low confidence the table fills with over a hundred rules, and fewer than one in ten shows a real association (lift above 1.5). Thresholds are how you separate habits from coincidences.',
    scenario: { minSupport: 0.01, minConfidence: 0.1 },
    claim: (r) => r.rules.length > 100 && r.strong / r.rules.length < 0.1,
  },
  {
    id: 'strict-confidence',
    title: 'Confidence set too high',
    text: 'No habit in these baskets is a certainty: even the strongest pair reaches about 80% confidence. Asking for 90% returns nothing.',
    scenario: { minConfidence: 0.9 },
    claim: (r) => r.rules.length === 0,
  },
  {
    id: 'few-baskets',
    title: 'Too few baskets',
    text: 'With 60 baskets, chance fluctuations look like strong associations: the best lift is well above the true value seen with thousands of baskets.',
    scenario: { baskets: 60 },
    claim: (r) => r.bestLift > 4.5,
  },
  {
    id: 'explosion',
    title: 'Bigger combinations',
    text: 'Allowing combinations of up to four items makes the number of rules explode, and most describe the same few habits in different combinations. Search cost and reading effort both grow fast.',
    scenario: { maxSize: 4, minSupport: 0.02 },
    claim: (r) => r.rules.length > 200,
  },
];

export const itemName = (i: number): string => ITEMS[i] as string;
