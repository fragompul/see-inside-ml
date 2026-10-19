import { Matrix, createRng, kFold, type Classifier } from '@sim/core';
import { LogisticRegression } from './logistic';

/**
 * Combining different models. `VotingClassifier` lets them vote (majority) or average their
 * probabilities; `StackingClassifier` trains a second-level model on what the first-level models
 * predict for rows they were *not* trained on, so it learns how far to trust each of them.
 */
const argmax = (a: ArrayLike<number>) => {
  let best = 0;
  for (let c = 1; c < a.length; c++) if ((a[c] as number) > (a[best] as number)) best = c;
  return best;
};

const takeRows = (x: Matrix, idx: readonly number[]): Matrix => {
  const out = new Matrix(idx.length, x.cols);
  idx.forEach((r, i) => out.data.set(x.data.subarray(r * x.cols, (r + 1) * x.cols), i * x.cols));
  return out;
};

export type Voting = 'hard' | 'soft';

export class VotingClassifier implements Classifier {
  classes = 2;
  members: Classifier[] = [];

  constructor(
    readonly makers: readonly (() => Classifier)[],
    readonly options: { voting?: Voting; weights?: readonly number[] } = {},
  ) {}

  fit(x: Matrix, y: ArrayLike<number>): this {
    this.members = this.makers.map((make) => make().fit(x, y));
    this.classes = Math.max(...this.members.map((m) => m.classes));
    return this;
  }

  private weight(i: number): number {
    return this.options.weights?.[i] ?? 1;
  }

  /** Probabilities: the weighted mean of the members' (soft) or the weighted share of their votes (hard). */
  predictProba(x: Matrix): Matrix {
    const out = new Matrix(x.rows, this.classes);
    let total = 0;
    this.members.forEach((m, k) => {
      const w = this.weight(k);
      total += w;
      if ((this.options.voting ?? 'hard') === 'soft' && m.predictProba) {
        const p = m.predictProba(x);
        for (let i = 0; i < x.rows; i++)
          for (let c = 0; c < p.cols; c++) out.set(i, c, out.get(i, c) + w * p.get(i, c));
      } else {
        const pred = m.predict(x);
        for (let i = 0; i < x.rows; i++)
          out.set(i, pred[i] as number, out.get(i, pred[i] as number) + w);
      }
    });
    for (let i = 0; i < out.data.length; i++) out.data[i] = (out.data[i] as number) / total;
    return out;
  }

  predict(x: Matrix): Int32Array {
    const p = this.predictProba(x);
    return Int32Array.from({ length: x.rows }, (_, i) =>
      argmax(p.data.subarray(i * p.cols, (i + 1) * p.cols)),
    );
  }
}

export class StackingClassifier implements Classifier {
  classes = 2;
  bases: Classifier[] = [];
  meta: LogisticRegression = new LogisticRegression({ C: 10 });

  constructor(
    readonly makers: readonly (() => Classifier)[],
    readonly options: { folds?: number; seed?: number; metaC?: number } = {},
  ) {}

  /** The features the meta-learner sees: every base model's class probabilities (all but the last class). */
  private metaFeatures(probas: Matrix[]): Matrix {
    const k = this.classes;
    const width = probas.length * (k - 1);
    const out = new Matrix(probas[0]!.rows, width);
    for (let i = 0; i < out.rows; i++)
      probas.forEach((p, m) => {
        for (let c = 0; c < k - 1; c++) out.set(i, m * (k - 1) + c, p.get(i, c));
      });
    return out;
  }

  private proba(model: Classifier, x: Matrix): Matrix {
    if (model.predictProba) {
      const p = model.predictProba(x);
      if (p.cols === this.classes) return p;
      const out = new Matrix(x.rows, this.classes);
      for (let i = 0; i < x.rows; i++) for (let c = 0; c < p.cols; c++) out.set(i, c, p.get(i, c));
      return out;
    }
    const pred = model.predict(x);
    const out = new Matrix(x.rows, this.classes);
    for (let i = 0; i < x.rows; i++) out.set(i, pred[i] as number, 1);
    return out;
  }

  fit(x: Matrix, y: ArrayLike<number>): this {
    const n = x.rows;
    let k = 0;
    for (let i = 0; i < n; i++) k = Math.max(k, (y[i] as number) + 1);
    this.classes = Math.max(2, k);
    const folds = kFold(n, Math.min(this.options.folds ?? 5, n), createRng(this.options.seed ?? 1));
    // out-of-fold probabilities of every base model: predictions on rows it never saw
    const oof = this.makers.map(() => new Matrix(n, this.classes));
    for (const fold of folds) {
      const train = takeRows(x, fold.train);
      const yTrain = fold.train.map((i) => y[i] as number);
      const valid = takeRows(x, fold.validation);
      this.makers.forEach((make, m) => {
        const p = this.proba(make().fit(train, yTrain), valid);
        fold.validation.forEach((row, r) => {
          for (let c = 0; c < this.classes; c++) (oof[m] as Matrix).set(row, c, p.get(r, c));
        });
      });
    }
    this.meta = new LogisticRegression({ C: this.options.metaC ?? 10 }).fit(
      this.metaFeatures(oof),
      y,
    );
    // the base models used at prediction time are refit on all the data
    this.bases = this.makers.map((make) => make().fit(x, y));
    return this;
  }

  predictProba(x: Matrix): Matrix {
    return this.meta.predictProba(this.metaFeatures(this.bases.map((b) => this.proba(b, x))));
  }

  predict(x: Matrix): Int32Array {
    const p = this.predictProba(x);
    return Int32Array.from({ length: x.rows }, (_, i) =>
      argmax(p.data.subarray(i * p.cols, (i + 1) * p.cols)),
    );
  }

  /** How much the meta-learner leans on each base model (summed absolute weights over its columns). */
  trust(): number[] {
    const per = this.classes - 1;
    const w = this.meta.weights;
    return this.makers.map((_, m) => {
      let s = 0;
      for (let r = 0; r < w.rows; r++)
        for (let c = 0; c < per; c++) s += Math.abs(w.get(r, m * per + c));
      return s;
    });
  }
}
