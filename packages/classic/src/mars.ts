import { Matrix, gram, solve, type Regressor } from '@sim/core';

/**
 * Multivariate adaptive regression splines (Friedman, 1991). The model is a sum of terms, each a product of up to
 * `degree` hinge functions max(0, x_v - t) or max(0, t - x_v), and a constant. A forward pass adds terms greedily, always
 * the pair of mirrored hinges (at the best variable and knot, multiplied by a term already in the model) that most
 * reduces the squared error; a backward pass prunes terms one at a time and keeps the model with the lowest
 * generalised cross-validation score.
 */
export interface Hinge {
  variable: number;
  knot: number;
  /** +1 for max(0, x - knot), -1 for max(0, knot - x). */
  sign: 1 | -1;
}

export type Term = Hinge[];

export interface MarsOptions {
  /** Terms at the end of the forward pass (before pruning). */
  maxTerms: number;
  /** Highest number of hinges multiplied together in one term. */
  degree: number;
  /** Candidate knots per variable, taken at evenly spaced quantiles of the data. */
  knots: number;
  /** Cost of a knot in the GCV criterion (Friedman uses 3 with interactions, 2 for an additive model). */
  penalty: number;
}

export const DEFAULT_MARS: MarsOptions = { maxTerms: 15, degree: 1, knots: 14, penalty: 2 };

export const hingeValue = (h: Hinge, x: number): number =>
  h.sign === 1 ? Math.max(0, x - h.knot) : Math.max(0, h.knot - x);

export const termValue = (t: Term, row: ArrayLike<number>): number => {
  let v = 1;
  for (const h of t) {
    v *= hingeValue(h, row[h.variable] as number);
    if (v === 0) return 0;
  }
  return v;
};

function leastSquares(
  columns: Float64Array[],
  y: ArrayLike<number>,
): { coef: Float64Array; rss: number } {
  const m = columns.length;
  const n = y.length;
  const a = new Matrix(n, m);
  for (let j = 0; j < m; j++)
    for (let i = 0; i < n; i++) a.data[i * m + j] = (columns[j] as Float64Array)[i] as number;
  const g = gram(a);
  for (let j = 0; j < m; j++) g.data[j * m + j] = (g.data[j * m + j] as number) + 1e-9;
  const rhs = new Matrix(m, 1);
  for (let j = 0; j < m; j++) {
    let s = 0;
    for (let i = 0; i < n; i++) s += (a.data[i * m + j] as number) * (y[i] as number);
    rhs.data[j] = s;
  }
  const coef = solve(g, rhs).data;
  let rss = 0;
  for (let i = 0; i < n; i++) {
    let p = 0;
    for (let j = 0; j < m; j++) p += (a.data[i * m + j] as number) * (coef[j] as number);
    rss += ((y[i] as number) - p) ** 2;
  }
  return { coef, rss };
}

export class Mars implements Regressor {
  /** The terms of the final model (the first is the empty product, the constant) and their coefficients. */
  terms: Term[] = [];
  coef: Float64Array = new Float64Array(0);
  /** The model after every addition of the forward pass and after every deletion of the backward pass: terms, RSS, GCV. */
  forwardPath: { terms: number; rss: number }[] = [];
  backwardPath: { terms: number; gcv: number }[] = [];
  gcv = 0;

  constructor(readonly options: MarsOptions = DEFAULT_MARS) {}

  private gcvOf(rss: number, terms: number, n: number): number {
    const knots = (terms - 1) / 2;
    const effective = terms + this.options.penalty * knots;
    const denom = 1 - effective / n;
    return denom <= 0 ? Infinity : rss / n / denom ** 2;
  }

  fit(x: Matrix, y: ArrayLike<number>): this {
    const n = x.rows;
    const d = x.cols;
    const rows = Array.from({ length: n }, (_, i) => x.data.subarray(i * d, (i + 1) * d));
    const column = (t: Term) => Float64Array.from(rows, (r) => termValue(t, r));
    const knotsFor = (v: number) => {
      const values = Array.from(
        new Set(Array.from({ length: n }, (_, i) => x.data[i * d + v] as number)),
      ).sort((a, b) => a - b);
      const count = Math.min(this.options.knots, values.length - 2);
      return Array.from(
        { length: Math.max(0, count) },
        (_, k) => values[Math.round(((k + 1) * (values.length - 1)) / (count + 1))] as number,
      );
    };
    const knots = Array.from({ length: d }, (_, v) => knotsFor(v));

    // forward pass
    let terms: Term[] = [[]];
    let columns: Float64Array[] = [column([])];
    let current = leastSquares(columns, y);
    this.forwardPath = [{ terms: 1, rss: current.rss }];
    while (terms.length + 2 <= this.options.maxTerms) {
      let best: { parent: number; variable: number; knot: number; rss: number } | null = null;
      for (let p = 0; p < terms.length; p++) {
        const parent = terms[p] as Term;
        if (parent.length >= this.options.degree) continue;
        const used = new Set(parent.map((h) => h.variable));
        for (let v = 0; v < d; v++) {
          if (used.has(v)) continue;
          for (const knot of knots[v] as number[]) {
            const up = column([...parent, { variable: v, knot, sign: 1 }]);
            const down = column([...parent, { variable: v, knot, sign: -1 }]);
            if (!up.some((q) => q !== 0) || !down.some((q) => q !== 0)) continue;
            const { rss } = leastSquares([...columns, up, down], y);
            if (!best || rss < best.rss - 1e-12) best = { parent: p, variable: v, knot, rss };
          }
        }
      }
      if (!best || best.rss > current.rss * (1 - 1e-4)) break;
      const parent = terms[best.parent] as Term;
      const up: Term = [...parent, { variable: best.variable, knot: best.knot, sign: 1 }];
      const down: Term = [...parent, { variable: best.variable, knot: best.knot, sign: -1 }];
      terms = [...terms, up, down];
      columns = [...columns, column(up), column(down)];
      current = leastSquares(columns, y);
      this.forwardPath.push({ terms: terms.length, rss: current.rss });
    }

    // backward pass: delete the term whose removal hurts least, remembering the best GCV along the way
    let bestTerms = terms;
    let bestCols = columns;
    let bestGcv = this.gcvOf(current.rss, terms.length, n);
    this.backwardPath = [{ terms: terms.length, gcv: bestGcv }];
    let t = terms;
    let c = columns;
    while (t.length > 1) {
      let drop = -1;
      let dropRss = Infinity;
      for (let k = 1; k < t.length; k++) {
        const rest = c.filter((_, i) => i !== k);
        const { rss } = leastSquares(rest, y);
        if (rss < dropRss) {
          dropRss = rss;
          drop = k;
        }
      }
      t = t.filter((_, i) => i !== drop);
      c = c.filter((_, i) => i !== drop);
      const g = this.gcvOf(dropRss, t.length, n);
      this.backwardPath.push({ terms: t.length, gcv: g });
      if (g < bestGcv) {
        bestGcv = g;
        bestTerms = t;
        bestCols = c;
      }
    }
    this.terms = bestTerms;
    this.coef = leastSquares(bestCols, y).coef;
    this.gcv = bestGcv;
    return this;
  }

  predict(x: Matrix): Float64Array {
    const d = x.cols;
    return Float64Array.from({ length: x.rows }, (_, i) => {
      const row = x.data.subarray(i * d, (i + 1) * d);
      let s = 0;
      this.terms.forEach((term, k) => (s += (this.coef[k] as number) * termValue(term, row)));
      return s;
    });
  }

  /** The knots used for each variable in the final model, sorted. */
  knotsUsed(variable: number): number[] {
    const k = new Set<number>();
    for (const t of this.terms) for (const h of t) if (h.variable === variable) k.add(h.knot);
    return [...k].sort((a, b) => a - b);
  }
}
