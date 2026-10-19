import { Matrix, createRng, solve } from '@sim/core';
import { describe, expect, it } from 'vitest';
import {
  apriori,
  associationRules,
  factorize,
  itemMeanPredictor,
  predict,
  rmseOf,
  type Rating,
} from './recommend';

describe('Apriori against brute force', () => {
  const rng = createRng(5);
  const items = 7;
  const baskets = Array.from({ length: 120 }, () => {
    const b: number[] = [];
    for (let i = 0; i < items; i++) if (rng.next() < 0.25 + 0.08 * i) b.push(i);
    if (rng.next() < 0.4) b.push(0, 1);
    return [...new Set(b)].sort((a, c) => a - c);
  });

  it('finds exactly the itemsets whose support reaches the threshold', () => {
    const minSupport = 0.1;
    const found = apriori(baskets, minSupport, items);
    const expected: Record<string, number> = {};
    for (let mask = 1; mask < 1 << items; mask++) {
      const set = Array.from({ length: items }, (_, i) => i).filter((i) => mask & (1 << i));
      const count = baskets.filter((b) => set.every((i) => b.includes(i))).length;
      if (count >= Math.ceil(minSupport * baskets.length - 1e-9)) expected[set.join(',')] = count;
    }
    expect(found).toHaveLength(Object.keys(expected).length);
    for (const f of found) expect(f.count).toBe(expected[f.items.join(',')]);
  });

  it('supports are anti-monotone: a bigger set is never more frequent than its subsets', () => {
    const found = apriori(baskets, 0.05, 4);
    const sup = new Map(found.map((f) => [f.items.join(','), f.support]));
    for (const f of found)
      if (f.items.length > 1)
        for (let drop = 0; drop < f.items.length; drop++)
          expect(sup.get(f.items.filter((_, k) => k !== drop).join(','))!).toBeGreaterThanOrEqual(
            f.support - 1e-12,
          );
  });

  it('rule metrics follow their definitions', () => {
    const found = apriori(baskets, 0.1, 3);
    const rules = associationRules(found, 0);
    expect(rules.length).toBeGreaterThan(0);
    const n = baskets.length;
    const freq = (set: number[]) =>
      baskets.filter((b) => set.every((i) => b.includes(i))).length / n;
    for (const r of rules.slice(0, 30)) {
      const both = freq([...r.antecedent, ...r.consequent]);
      expect(r.support).toBeCloseTo(both, 12);
      expect(r.confidence).toBeCloseTo(both / freq(r.antecedent), 12);
      expect(r.lift).toBeCloseTo(both / (freq(r.antecedent) * freq(r.consequent)), 10);
      expect(r.leverage).toBeCloseTo(both - freq(r.antecedent) * freq(r.consequent), 12);
    }
  });

  it('a minimum confidence filters rules', () => {
    const found = apriori(baskets, 0.1, 3);
    const lax = associationRules(found, 0.1).length;
    const strict = associationRules(found, 0.8).length;
    expect(strict).toBeLessThan(lax);
  });
});

function lowRankRatings(users: number, items: number, k: number, seed: number) {
  const rng = createRng(seed);
  const u = Array.from({ length: users }, () => Array.from({ length: k }, () => rng.normal()));
  const v = Array.from({ length: items }, () => Array.from({ length: k }, () => rng.normal()));
  const all: Rating[] = [];
  for (let i = 0; i < users; i++)
    for (let j = 0; j < items; j++)
      all.push({ user: i, item: j, value: 3 + u[i]!.reduce((s, x, f) => s + x * v[j]![f]!, 0) });
  const train: Rating[] = [];
  const test: Rating[] = [];
  for (const r of all) (rng.next() < 0.5 ? train : test).push(r);
  return { train, test };
}

describe('matrix factorisation', () => {
  const users = 30;
  const items = 25;
  const { train, test } = lowRankRatings(users, items, 2, 7);

  it('ALS never increases the regularised objective', () => {
    const states = [
      ...factorize(train, { users, items, factors: 2, lambda: 0.1, iterations: 15, seed: 1 }),
    ];
    for (let i = 1; i < states.length; i++)
      expect(states[i]!.objective).toBeLessThanOrEqual(states[i - 1]!.objective + 1e-9);
  });

  it('recovers a noise-free rank-2 table: held-out ratings are predicted far better than by the item mean', () => {
    const states = [
      ...factorize(train, { users, items, factors: 2, lambda: 0.05, iterations: 40, seed: 1 }),
    ];
    const fit = states[states.length - 1]!;
    const baseline = itemMeanPredictor(train, items);
    const baseRmse = Math.sqrt(
      test.reduce((s, r) => s + (r.value - baseline(r.user, r.item)) ** 2, 0) / test.length,
    );
    expect(rmseOf(fit, test)).toBeLessThan(0.35);
    expect(rmseOf(fit, test)).toBeLessThan(baseRmse / 3);
  });

  it('one ALS update of a user is exactly a ridge regression on the items they rated', () => {
    const lambda = 0.3;
    const [start, one] = factorize(train, {
      users,
      items,
      factors: 2,
      lambda,
      iterations: 1,
      seed: 3,
    });
    // reproduce the user-0 solve by hand from the starting item factors
    const rated = train.filter((r) => r.user === 0);
    const a = new Matrix(2, 2);
    const b = new Matrix(2, 1);
    for (let f = 0; f < 2; f++) a.set(f, f, lambda);
    for (const r of rated)
      for (let f = 0; f < 2; f++) {
        b.set(f, 0, b.get(f, 0) + start!.V.get(r.item, f) * (r.value - start!.mean));
        for (let g = 0; g < 2; g++)
          a.set(f, g, a.get(f, g) + start!.V.get(r.item, f) * start!.V.get(r.item, g));
      }
    const x = solve(a, b);
    expect(one!.U.get(0, 0)).toBeCloseTo(x.get(0, 0), 10);
    expect(one!.U.get(0, 1)).toBeCloseTo(x.get(1, 0), 10);
  });

  it('a huge penalty shrinks every prediction to the global mean', () => {
    const states = [
      ...factorize(train, { users, items, factors: 2, lambda: 1e6, iterations: 5, seed: 1 }),
    ];
    const fit = states[states.length - 1]!;
    expect(predict(fit, 3, 4)).toBeCloseTo(fit.mean, 4);
  });

  it('SGD lowers the training error as well', () => {
    const states = [
      ...factorize(train, {
        users,
        items,
        factors: 2,
        lambda: 0.05,
        iterations: 60,
        method: 'sgd',
        learningRate: 0.03,
        seed: 1,
      }),
    ];
    expect(states[states.length - 1]!.trainRmse).toBeLessThan(states[0]!.trainRmse / 2);
  });
});
