import { apriori, associationRules, type AssociationRule, type Itemset } from '@sim/classic';
import { createRng } from '@sim/core';

export const ITEMS = [
  'bread',
  'butter',
  'milk',
  'eggs',
  'beer',
  'diapers',
  'chips',
  'salsa',
  'wine',
  'cheese',
  'apples',
  'coffee',
] as const;

export const name = (set: readonly number[]): string => set.map((i) => ITEMS[i]).join(' + ');

/** The pairs where one item truly makes the other more likely (the structure hidden in the baskets). */
export const PLANTED: readonly [number, number][] = [
  [0, 1],
  [4, 5],
  [6, 7],
  [8, 9],
];

export type RuleSort = 'lift' | 'confidence' | 'support';

export interface AssociationScenario {
  baskets: number;
  minSupport: number;
  minConfidence: number;
  sortBy: RuleSort;
  /** Largest itemset considered (2 = pairs only). */
  maxSize: number;
  seed: number;
}

export const DEFAULT_ASSOCIATION: AssociationScenario = {
  baskets: 600,
  minSupport: 0.05,
  minConfidence: 0.5,
  sortBy: 'lift',
  maxSize: 2,
  seed: 8,
};

export function makeBaskets(n: number, seed: number): number[][] {
  const rng = createRng(seed);
  const out: number[][] = [];
  for (let i = 0; i < n; i++) {
    const b = new Set<number>();
    const add = (item: number, p: number) => {
      if (rng.next() < p) b.add(item);
    };
    add(2, 0.6); // milk: popular, unrelated to anything
    add(3, 0.3);
    add(10, 0.2);
    add(11, 0.25);
    add(0, 0.45);
    add(1, b.has(0) ? 0.75 : 0.1);
    add(4, 0.2);
    add(5, b.has(4) ? 0.5 : 0.08);
    add(6, 0.25);
    add(7, b.has(6) ? 0.7 : 0.05);
    add(8, 0.15);
    add(9, b.has(8) ? 0.65 : 0.1);
    if (b.size === 0) b.add(2);
    out.push([...b].sort((a, c) => a - c));
  }
  return out;
}

export interface AssociationRun {
  scenario: AssociationScenario;
  baskets: number[][];
  itemsets: Itemset[];
  rules: AssociationRule[];
  /** Rules sorted by the chosen measure. */
  ranked: AssociationRule[];
  bestLift: number;
  strong: number;
  /** Planted pairs that appear (in either direction) among the rules. */
  recovered: number;
  /** Frequency of every single item. */
  frequency: number[];
}

export function runAssociation(s: AssociationScenario): AssociationRun {
  const baskets = makeBaskets(s.baskets, s.seed);
  const itemsets = apriori(baskets, s.minSupport, s.maxSize);
  const rules = associationRules(itemsets, s.minConfidence);
  const ranked = [...rules].sort((a, b) => b[s.sortBy] - a[s.sortBy] || b.lift - a.lift);
  const has = (a: number, c: number) =>
    rules.some(
      (r) =>
        r.antecedent.length === 1 &&
        r.consequent.length === 1 &&
        ((r.antecedent[0] === a && r.consequent[0] === c) ||
          (r.antecedent[0] === c && r.consequent[0] === a)),
    );
  const frequency = ITEMS.map(
    (_, i) => baskets.filter((b) => b.includes(i)).length / baskets.length,
  );
  return {
    scenario: s,
    baskets,
    itemsets,
    rules,
    ranked,
    bestLift: rules.length ? Math.max(...rules.map((r) => r.lift)) : 0,
    strong: rules.filter((r) => r.lift > 1.5).length,
    recovered: PLANTED.filter(([a, c]) => has(a, c)).length,
    frequency,
  };
}
