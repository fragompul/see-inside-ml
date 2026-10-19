import { Matrix, createRng, solve } from '@sim/core';

/**
 * Two classic "people who liked this also liked that" tools.
 *   - Association rules (Apriori): which items turn up together in baskets more often than chance?
 *   - Matrix factorisation (ALS, SGD): explain a sparse table of ratings by a few hidden "taste"
 *     dimensions and predict the empty cells.
 */

// ---- association rules ---------------------------------------------------------------------------------

export interface Itemset {
  items: number[];
  count: number;
  /** Share of baskets that contain every item of the set. */
  support: number;
}

export interface AssociationRule {
  antecedent: number[];
  consequent: number[];
  support: number;
  /** P(consequent | antecedent). */
  confidence: number;
  /** Confidence divided by the consequent's own frequency: 1 = no association, above 1 = positive. */
  lift: number;
  /** support(A,C) - support(A) * support(C): the excess over independence. */
  leverage: number;
}

const key = (items: readonly number[]) => items.join(',');

/** Frequent itemsets by the Apriori level-wise search: a set can only be frequent if all its subsets are. */
export function apriori(
  baskets: readonly (readonly number[])[],
  minSupport: number,
  maxSize = 4,
): Itemset[] {
  const n = baskets.length;
  const sets = baskets.map((b) => new Set(b));
  const need = Math.ceil(minSupport * n - 1e-9);
  const countOf = (items: readonly number[]) =>
    sets.reduce((c, s) => c + (items.every((i) => s.has(i)) ? 1 : 0), 0);
  const all: Itemset[] = [];
  const universe = [...new Set(baskets.flat())].sort((a, b) => a - b);
  let level: number[][] = [];
  for (const item of universe) {
    const c = countOf([item]);
    if (c >= need) {
      level.push([item]);
      all.push({ items: [item], count: c, support: c / n });
    }
  }
  for (let size = 2; size <= maxSize && level.length > 1; size++) {
    const known = new Set(level.map(key));
    const candidates = new Map<string, number[]>();
    for (let i = 0; i < level.length; i++)
      for (let j = i + 1; j < level.length; j++) {
        const a = level[i] as number[];
        const b = level[j] as number[];
        // join two sets that agree on all but the last item
        if (a.slice(0, -1).every((v, k) => v === b[k])) {
          const merged = [...a, b[b.length - 1] as number];
          // prune: every (size-1)-subset must itself be frequent
          const ok = merged.every((_, drop) =>
            known.has(key(merged.filter((__, k) => k !== drop))),
          );
          if (ok) candidates.set(key(merged), merged);
        }
      }
    const next: number[][] = [];
    for (const c of candidates.values()) {
      const cnt = countOf(c);
      if (cnt >= need) {
        next.push(c);
        all.push({ items: c, count: cnt, support: cnt / n });
      }
    }
    level = next;
  }
  return all;
}

/** All rules `A -> C` that can be made by splitting a frequent itemset, with confidence at least `minConfidence`. */
export function associationRules(
  itemsets: readonly Itemset[],
  minConfidence: number,
): AssociationRule[] {
  const support = new Map(itemsets.map((s) => [key(s.items), s.support]));
  const rules: AssociationRule[] = [];
  for (const set of itemsets) {
    if (set.items.length < 2) continue;
    const m = set.items.length;
    for (let mask = 1; mask < (1 << m) - 1; mask++) {
      const antecedent = set.items.filter((_, i) => mask & (1 << i));
      const consequent = set.items.filter((_, i) => !(mask & (1 << i)));
      const sa = support.get(key(antecedent));
      const sc = support.get(key(consequent));
      if (sa === undefined || sc === undefined) continue;
      const confidence = set.support / sa;
      if (confidence + 1e-12 < minConfidence) continue;
      rules.push({
        antecedent,
        consequent,
        support: set.support,
        confidence,
        lift: confidence / sc,
        leverage: set.support - sa * sc,
      });
    }
  }
  return rules;
}

// ---- matrix factorisation ----------------------------------------------------------------------------------

export interface Rating {
  user: number;
  item: number;
  value: number;
}

export interface FactorOptions {
  users: number;
  items: number;
  /** Number of hidden taste dimensions. */
  factors: number;
  /** L2 penalty on the factors. */
  lambda: number;
  iterations: number;
  method?: 'als' | 'sgd';
  /** SGD learning rate. */
  learningRate?: number;
  seed?: number;
}

export interface FactorState {
  iteration: number;
  /** `users x factors` and `items x factors`. */
  U: Matrix;
  V: Matrix;
  mean: number;
  /** Regularised objective on the training ratings. */
  objective: number;
  trainRmse: number;
}

export const predict = (
  s: Pick<FactorState, 'U' | 'V' | 'mean'>,
  user: number,
  item: number,
): number => {
  let v = s.mean;
  for (let f = 0; f < s.U.cols; f++) v += s.U.get(user, f) * s.V.get(item, f);
  return v;
};

export function rmseOf(
  s: Pick<FactorState, 'U' | 'V' | 'mean'>,
  ratings: readonly Rating[],
): number {
  if (!ratings.length) return NaN;
  let e = 0;
  for (const r of ratings) e += (r.value - predict(s, r.user, r.item)) ** 2;
  return Math.sqrt(e / ratings.length);
}

const objectiveOf = (s: FactorState, ratings: readonly Rating[], lambda: number): number => {
  let e = 0;
  for (const r of ratings) e += (r.value - predict(s, r.user, r.item)) ** 2;
  let reg = 0;
  for (const v of s.U.data) reg += v * v;
  for (const v of s.V.data) reg += v * v;
  return e + lambda * reg;
};

/** Alternating least squares / stochastic gradient descent as a generator (state 0 is the random start). */
export function* factorize(
  ratings: readonly Rating[],
  o: FactorOptions,
): Generator<FactorState, void, undefined> {
  const { users, items, factors: k, lambda, iterations } = o;
  const rng = createRng(o.seed ?? 1);
  const mean = ratings.reduce((a, r) => a + r.value, 0) / Math.max(ratings.length, 1);
  const U = new Matrix(
    users,
    k,
    Float64Array.from({ length: users * k }, () => rng.normal() * 0.1),
  );
  const V = new Matrix(
    items,
    k,
    Float64Array.from({ length: items * k }, () => rng.normal() * 0.1),
  );
  const byUser: Rating[][] = Array.from({ length: users }, () => []);
  const byItem: Rating[][] = Array.from({ length: items }, () => []);
  for (const r of ratings) {
    (byUser[r.user] as Rating[]).push(r);
    (byItem[r.item] as Rating[]).push(r);
  }
  const snapshot = (iteration: number): FactorState => {
    const s: FactorState = {
      iteration,
      U: new Matrix(users, k, Float64Array.from(U.data)),
      V: new Matrix(items, k, Float64Array.from(V.data)),
      mean,
      objective: 0,
      trainRmse: 0,
    };
    s.objective = objectiveOf(s, ratings, lambda);
    s.trainRmse = rmseOf(s, ratings);
    return s;
  };
  yield snapshot(0);
  // solve (F_obs^T F_obs + lambda I) x = F_obs^T r for one row of the unknown factor matrix
  const solveRow = (
    target: Matrix,
    row: number,
    others: Matrix,
    obs: Rating[],
    otherIndex: 'item' | 'user',
  ) => {
    const a = new Matrix(k, k);
    const b = new Matrix(k, 1);
    for (let f = 0; f < k; f++) a.set(f, f, lambda);
    for (const r of obs) {
      const o2 = otherIndex === 'item' ? r.item : r.user;
      const y = r.value - mean;
      for (let f = 0; f < k; f++) {
        const vf = others.get(o2, f);
        b.set(f, 0, b.get(f, 0) + vf * y);
        for (let g = 0; g < k; g++) a.set(f, g, a.get(f, g) + vf * others.get(o2, g));
      }
    }
    const x = solve(a, b);
    for (let f = 0; f < k; f++) target.set(row, f, x.get(f, 0));
  };
  for (let it = 1; it <= iterations; it++) {
    if ((o.method ?? 'als') === 'als') {
      for (let u = 0; u < users; u++)
        if ((byUser[u] as Rating[]).length) solveRow(U, u, V, byUser[u] as Rating[], 'item');
      for (let i = 0; i < items; i++)
        if ((byItem[i] as Rating[]).length) solveRow(V, i, U, byItem[i] as Rating[], 'user');
    } else {
      const lr = o.learningRate ?? 0.02;
      const order = Array.from({ length: ratings.length }, (_, i) => i);
      for (let i = order.length - 1; i > 0; i--) {
        const j = Math.floor(rng.next() * (i + 1));
        [order[i], order[j]] = [order[j] as number, order[i] as number];
      }
      for (const idx of order) {
        const r = ratings[idx] as Rating;
        const err = r.value - predict({ U, V, mean }, r.user, r.item);
        for (let f = 0; f < k; f++) {
          const uf = U.get(r.user, f);
          const vf = V.get(r.item, f);
          // the penalty is shared out over the ratings of each user / item
          U.set(
            r.user,
            f,
            uf + lr * (err * vf - (lambda / (byUser[r.user] as Rating[]).length) * uf),
          );
          V.set(
            r.item,
            f,
            vf + lr * (err * uf - (lambda / (byItem[r.item] as Rating[]).length) * vf),
          );
        }
      }
    }
    yield snapshot(it);
  }
}

/** Baselines: always the global mean, or the mean of the item. */
export function itemMeanPredictor(
  ratings: readonly Rating[],
  items: number,
): (user: number, item: number) => number {
  const mean = ratings.reduce((a, r) => a + r.value, 0) / Math.max(ratings.length, 1);
  const sum = new Array<number>(items).fill(0);
  const cnt = new Array<number>(items).fill(0);
  for (const r of ratings) {
    sum[r.item] = (sum[r.item] as number) + r.value;
    cnt[r.item] = (cnt[r.item] as number) + 1;
  }
  return (_, item) =>
    (cnt[item] as number) > 0 ? (sum[item] as number) / (cnt[item] as number) : mean;
}
