import { describe, expect, it } from 'vitest';
import fixture from '../fixtures/core.json';
import {
  cholesky,
  choleskySolve,
  determinant,
  inverse,
  logDetFromCholesky,
  matMul,
  solve,
  symmetricEigen,
  transpose,
  identity,
  gram,
  matVec,
} from './linalg';
import { Matrix } from './matrix';
import {
  accuracy,
  adjustedRandIndex,
  auc,
  classScores,
  confusionMatrix,
  logLoss,
  mae,
  mse,
  r2,
  rmse,
  rocCurve,
  silhouette,
} from './metrics';
import { nelderMead } from './neldermead';
import { createRng } from './rng';
import {
  betaInc,
  chi2Cdf,
  columnMeans,
  correlation,
  covariance,
  fCdf,
  gammaInc,
  lgamma,
  mean,
  median,
  normalCdf,
  normalPdf,
  normalPpf,
  quantile,
  std,
  tCdf,
  tPpf,
  variance,
} from './stats';
import {
  CLASSIFICATION_SETS,
  REGRESSION_SETS,
  Standardizer,
  anscombe,
  kFold,
  makeClassification,
  makeMultiple,
  makeRegression,
  makeManifold,
  MANIFOLD_SETS,
  makeTable,
  plotBounds,
  subset,
  trainTestSplit,
} from './tabular';
import { gridPoints } from './estimator';

const m = (rows: number[][]) => Matrix.from(rows);
const close = (a: number, b: number, tol = 1e-9) =>
  expect(Math.abs(a - b)).toBeLessThanOrEqual(tol * Math.max(1, Math.abs(b)));

describe('distributions agree with SciPy', () => {
  it('normal', () => {
    const f = fixture.normal;
    f.x.forEach((x, i) => {
      close(normalCdf(x), f.cdf[i] as number, 1e-12);
      close(normalPdf(x), f.pdf[i] as number, 1e-12);
    });
    f.p.forEach((p, i) => close(normalPpf(p), f.ppf[i] as number, 1e-9));
    close(normalCdf(5, 3, 2), 0.8413447460685429, 1e-12);
  });

  it('Student t', () => {
    for (const c of fixture.t) {
      close(tCdf(c.t, c.df), c.cdf, 1e-10);
      close(tPpf(c.p, c.df), c.ppf, 1e-7);
    }
  });

  it('F, chi-squared, gamma and beta functions', () => {
    for (const c of fixture.f) close(fCdf(c.f, c.d1, c.d2), c.cdf, 1e-10);
    for (const c of fixture.chi2) close(chi2Cdf(c.x, c.k), c.cdf, 1e-10);
    for (const c of fixture.lgamma) close(lgamma(c.x), c.value, 1e-11);
    for (const c of fixture.betainc) close(betaInc(c.x, c.a, c.b), c.value, 1e-10);
    expect(gammaInc(2, 0)).toBe(0);
    expect(betaInc(0, 2, 3)).toBe(0);
    expect(betaInc(1, 2, 3)).toBe(1);
    expect(fCdf(-1, 3, 4)).toBe(0);
  });
});

describe('linear algebra agrees with NumPy', () => {
  const L = fixture.linalg;
  it('Cholesky, LU, inverse and determinant', () => {
    const spd = m(L.spd);
    const b = m(L.b);
    const factor = cholesky(spd);
    const back = matMul(factor, transpose(factor));
    spd.data.forEach((v, i) => close(back.data[i] as number, v, 1e-12));
    choleskySolve(factor, b).data.forEach((v, i) =>
      close(v, L.solve_spd.flat()[i] as number, 1e-10),
    );
    close(logDetFromCholesky(factor), L.logdet_spd, 1e-11);
    const a = m(L.general);
    solve(a, b).data.forEach((v, i) => close(v, L.solve_general.flat()[i] as number, 1e-9));
    inverse(a).data.forEach((v, i) => close(v, L.inverse.flat()[i] as number, 1e-9));
    close(determinant(a), L.det, 1e-10);
    const prod = matMul(a, inverse(a));
    identity(5).data.forEach((v, i) => close(prod.data[i] as number, v, 1e-10));
  });

  it('symmetric eigen-decomposition', () => {
    const { values, vectors } = symmetricEigen(m(L.sym));
    values.forEach((v, i) => close(v, L.eigenvalues[i] as number, 1e-9));
    vectors.data.forEach((v, i) =>
      close(Math.abs(v), L.eigenvectors_abs.flat()[i] as number, 1e-8),
    );
    // A v = λ v
    const sym = m(L.sym);
    for (let c = 0; c < 5; c++) {
      const v = Array.from({ length: 5 }, (_, r) => vectors.get(r, c));
      const av = matVec(sym, v);
      av.forEach((x, r) => close(x, (values[c] as number) * (v[r] as number), 1e-9));
    }
  });

  it('rejects what cannot be done', () => {
    expect(() =>
      cholesky(
        m([
          [1, 2],
          [2, 1],
        ]),
      ),
    ).toThrow(RangeError); // not positive definite
    expect(() =>
      solve(
        m([
          [1, 2],
          [2, 4],
        ]),
        m([[1], [1]]),
      ),
    ).toThrow(RangeError); // singular
    expect(
      determinant(
        m([
          [1, 2],
          [2, 4],
        ]),
      ),
    ).toBe(0);
    expect(() => cholesky(new Matrix(2, 3))).toThrow(RangeError);
    expect(() => matMul(new Matrix(2, 3), new Matrix(2, 3))).toThrow(RangeError);
    expect(
      gram(
        m([
          [1, 2],
          [3, 4],
          [5, 6],
        ]),
      ).data,
    ).toEqual(
      matMul(
        transpose(
          m([
            [1, 2],
            [3, 4],
            [5, 6],
          ]),
        ),
        m([
          [1, 2],
          [3, 4],
          [5, 6],
        ]),
      ).data,
    );
  });
});

describe('descriptive statistics agree with NumPy', () => {
  const D = fixture.descriptive;
  it('mean, covariance, correlation, quantiles', () => {
    const x = m(D.x);
    columnMeans(x).forEach((v, i) => close(v, D.mean[i] as number, 1e-12));
    covariance(x).data.forEach((v, i) => close(v, D.cov.flat()[i] as number, 1e-10));
    const col = (j: number) => D.x.map((r) => r[j] as number);
    close(correlation(col(0), col(1)), D.corr01, 1e-12);
    D.q.forEach((q, i) => close(quantile(col(0), q), D.quantiles0[i] as number, 1e-12));
    close(median([3, 1, 2]), 2);
    close(variance([1, 2, 3, 4], 1), 5 / 3);
    close(std([2, 2, 2]), 0);
    close(mean([1, 2, 3, 4]), 2.5);
  });
});

describe('metrics agree with scikit-learn', () => {
  const M = fixture.metrics;
  it('regression scores', () => {
    close(mse(M.y_reg, M.p_reg), M.mse, 1e-12);
    close(mae(M.y_reg, M.p_reg), M.mae, 1e-12);
    close(r2(M.y_reg, M.p_reg), M.r2, 1e-12);
    close(rmse(M.y_reg, M.p_reg), Math.sqrt(M.mse), 1e-12);
    expect(r2([1, 1, 1], [1, 1, 1])).toBe(1);
  });

  it('classification scores', () => {
    const k = 3;
    const cm = confusionMatrix(M.labels_a, M.labels_b, k);
    expect(cm).toEqual(M.confusion);
    const scores = classScores(cm);
    scores.forEach((s, c) => {
      close(s.f1, M.f1[c] as number, 1e-12);
      close(s.precision, M.precision[c] as number, 1e-12);
      close(s.recall, M.recall[c] as number, 1e-12);
    });
    close(accuracy([0, 1, 1], [0, 1, 0]), 2 / 3);
    const probs = Matrix.from([
      [0.9, 0.1],
      [0.2, 0.8],
    ]);
    close(logLoss(probs, [0, 1]), -(Math.log(0.9) + Math.log(0.8)) / 2, 1e-12);
  });

  it('ROC curve and AUC', () => {
    const { fpr, tpr } = rocCurve(M.scores, M.y_true);
    close(auc(fpr, tpr), M.auc, 1e-12);
    expect(fpr).toHaveLength(M.roc_points);
    fpr.forEach((v, i) => close(v, M.fpr[i] as number, 1e-12));
    tpr.forEach((v, i) => close(v, M.tpr[i] as number, 1e-12));
  });

  it('clustering scores', () => {
    close(adjustedRandIndex(M.labels_a, M.labels_b), M.ari, 1e-12);
    close(silhouette(m(M.points), M.cluster_labels), M.silhouette, 1e-12);
    expect(adjustedRandIndex([0, 0, 1, 1], [5, 5, 9, 9])).toBe(1); // relabelling does not matter
    expect(
      silhouette(
        m([
          [0, 0],
          [1, 1],
        ]),
        [0, 0],
      ),
    ).toBe(0); // a single cluster has no silhouette
  });
});

describe('tabular datasets', () => {
  it.each(CLASSIFICATION_SETS)(
    '%s: reproducible, finite, labelled 0…k−1 and both/all classes present',
    (kind) => {
      const a = makeClassification(kind, 120, 0.1, createRng(2));
      const b = makeClassification(kind, 120, 0.1, createRng(2));
      expect(a.x.data).toEqual(b.x.data);
      expect(a.y).toEqual(b.y);
      expect(a.x.rows).toBe(120);
      expect(a.x.data.every(Number.isFinite)).toBe(true);
      const labels = new Set(a.y);
      expect(labels.size).toBe(a.classes?.length);
      expect(Math.max(...labels)).toBe((a.classes?.length ?? 0) - 1);
      expect(makeClassification(kind, 120, 0.1, createRng(3)).x.data).not.toEqual(a.x.data);
    },
  );

  it('imbalanced really is imbalanced and XOR really is XOR', () => {
    const imb = makeClassification('imbalanced', 330, 0, createRng(1));
    const rare = imb.y.reduce((s, v) => s + v, 0);
    expect(rare).toBe(30);
    const xor = makeClassification('xor', 200, 0, createRng(1));
    for (let i = 0; i < 200; i++)
      expect(xor.y[i]).toBe(xor.x.get(i, 0) * xor.x.get(i, 1) > 0 ? 1 : 0);
  });

  it.each(REGRESSION_SETS)('%s regression data is reproducible and finite', (kind) => {
    const a = makeRegression(kind, 80, 0.3, createRng(5));
    expect(a.x.cols).toBe(1);
    expect(a.y.every(Number.isFinite)).toBe(true);
    expect(makeRegression(kind, 80, 0.3, createRng(5)).y).toEqual(a.y);
  });

  it("Anscombe's quartet has the same statistics but four different shapes", () => {
    for (const i of [1, 2, 3, 4] as const) {
      const t = anscombe(i);
      close(mean(t.y), 7.5, 0.01);
      close(
        correlation(
          Array.from({ length: 11 }, (_, r) => t.x.get(r, 0)),
          t.y,
        ),
        0.816,
        0.01,
      );
    }
    expect(anscombe(4).x.get(7, 0)).toBe(19);
  });

  it('multiple regression data has the requested correlation between its first two features', () => {
    const t = makeMultiple(4000, 0.1, 0.9, createRng(1));
    const col = (j: number) => Array.from({ length: t.x.rows }, (_, i) => t.x.get(i, j));
    close(correlation(col(0), col(1)), 0.9, 0.03);
  });

  it('splits keep every row exactly once and stratified splits keep class shares', () => {
    const t = makeClassification('imbalanced', 330, 0, createRng(1));
    const { train, test } = trainTestSplit(t, 0.3, createRng(2), true);
    expect(train.x.rows + test.x.rows).toBe(330);
    expect(test.y.reduce((s, v) => s + v, 0)).toBe(9); // 30 rare points · 0.3
    const plain = trainTestSplit(t, 0.25, createRng(2));
    expect(plain.test.x.rows).toBe(83);
    expect(subset(t, [0, 5]).x.rows).toBe(2);
  });

  it('k-fold partitions the data: every row validates exactly once', () => {
    const folds = kFold(23, 5, createRng(1));
    const seen = folds.flatMap((f) => f.validation).sort((a, b) => a - b);
    expect(seen).toEqual(Array.from({ length: 23 }, (_, i) => i));
    folds.forEach((f) => expect(f.train.length + f.validation.length).toBe(23));
    expect(() => kFold(5, 1, createRng(1))).toThrow(RangeError);
  });

  it('the standardizer zero-centres, scales and inverts', () => {
    const x = m([
      [1, 10],
      [2, 20],
      [3, 30],
      [4, 40],
    ]);
    const s = new Standardizer().fit(x);
    const z = s.transform(x);
    columnMeans(z).forEach((v) => close(v, 0, 1e-12));
    close(std(Array.from({ length: 4 }, (_, i) => z.get(i, 1))), 1, 1e-12);
    s.inverse(z).data.forEach((v, i) => close(v, x.data[i] as number, 1e-12));
    const constant = new Standardizer().fit(
      m([
        [1, 5],
        [2, 5],
      ]),
    );
    expect(constant.scale[1]).toBe(1); // a constant column is left alone
  });

  it('plot bounds and grids cover the data', () => {
    const t = makeTable(
      [
        [0, 0],
        [2, 4],
      ],
      [0, 1],
      ['a', 'b'],
    );
    const b = plotBounds(t.x, 0.1);
    expect(b.x0).toBeLessThan(0);
    expect(b.x1).toBeGreaterThan(2);
    expect(b.y1).toBeGreaterThan(4);
    const g = gridPoints(0, 1, 0, 1, 4);
    expect(g.rows).toBe(16);
    expect(g.get(0, 1)).toBeGreaterThan(g.get(12, 1)); // rows run from the top down
  });
});

describe('bundled classic datasets', () => {
  it('have the documented shapes, finite values and consistent labels', async () => {
    const { CLASSIC_DATASETS } = await import('./datasets');
    const shapes = {
      iris: [150, 4],
      wine: [178, 13],
      diabetes: [442, 10],
      breastCancer: [569, 30],
    } as const;
    for (const [id, [n, d]] of Object.entries(shapes)) {
      const ds = CLASSIC_DATASETS[id as keyof typeof CLASSIC_DATASETS];
      expect(ds.rows).toHaveLength(n);
      expect(ds.rows[0]).toHaveLength(d);
      expect(ds.features).toHaveLength(d);
      expect(ds.target).toHaveLength(n);
      expect(ds.rows.every((r) => r.every(Number.isFinite))).toBe(true);
      expect(ds.licence.length).toBeGreaterThan(5);
      if (ds.kind === 'classification') {
        expect(Math.max(...ds.target) + 1).toBe(ds.classes?.length);
      } else {
        expect(ds.classes).toBeNull();
      }
    }
    expect(CLASSIC_DATASETS.iris.classes).toEqual(['setosa', 'versicolor', 'virginica']);
  });
});

describe('datasets built for the Gaussian classifiers', () => {
  const within = (kind: 'tilted' | 'cigars', c: number) => {
    const t = makeClassification(kind, 2000, 0.1, createRng(8));
    const a: number[] = [];
    const b: number[] = [];
    for (let i = 0; i < t.y.length; i++)
      if (t.y[i] === c) {
        a.push(t.x.get(i, 0));
        b.push(t.x.get(i, 1));
      }
    return correlation(a, b);
  };

  it('"tilted" has strongly correlated features inside each class', () => {
    expect(within('tilted', 0)).toBeGreaterThan(0.8);
    expect(within('tilted', 1)).toBeGreaterThan(0.8);
  });

  it('"cigars" classes lean opposite ways: positive versus negative correlation', () => {
    expect(within('cigars', 0)).toBeGreaterThan(0.6);
    expect(within('cigars', 1)).toBeLessThan(-0.6);
  });
});

describe('count data', () => {
  it('"counts" are whole numbers whose mean grows with x', () => {
    const t = makeRegression('counts', 3000, 0, createRng(6));
    expect(Array.from(t.y).every((v) => v >= 0 && Number.isInteger(v))).toBe(true);
    const low: number[] = [];
    const high: number[] = [];
    for (let i = 0; i < t.y.length; i++) (t.x.get(i, 0) < 0 ? low : high).push(t.y[i] as number);
    expect(mean(high)).toBeGreaterThan(2 * mean(low));
  });
});

describe('Nelder–Mead', () => {
  it('finds the minimum of a quadratic and of the Rosenbrock valley', () => {
    const q = nelderMead((x) => (x[0]! - 3) ** 2 + (x[1]! + 1) ** 2, [0, 0], { step: 1 });
    expect(q.x[0]).toBeCloseTo(3, 4);
    expect(q.x[1]).toBeCloseTo(-1, 4);
    const rosen = nelderMead((x) => 100 * (x[1]! - x[0]! ** 2) ** 2 + (1 - x[0]!) ** 2, [-1.2, 1], {
      maxIterations: 2000,
    });
    expect(rosen.x[0]).toBeCloseTo(1, 3);
    expect(rosen.x[1]).toBeCloseTo(1, 3);
    expect(rosen.value).toBeLessThan(1e-8);
  });
});

describe('manifold datasets', () => {
  it('have the advertised dimension, are reproducible and carry a position or a class in y', () => {
    for (const kind of MANIFOLD_SETS) {
      const a = makeManifold(kind, 100, 0.1, createRng(2));
      const b = makeManifold(kind, 100, 0.1, createRng(2));
      expect(a.x.rows).toBe(100);
      expect(a.x.cols).toBe(kind === 'clusters' ? 10 : 3);
      expect(Array.from(a.x.data)).toEqual(Array.from(b.x.data));
      expect(a.y.every((v) => Number.isFinite(v))).toBe(true);
      if (kind !== 'clusters') expect(a.y.every((v) => v >= 0 && v <= 1)).toBe(true);
      else expect(new Set(a.y).size).toBe(5);
    }
  });

  it('the sphere has unit radius up to the noise and the swiss roll spirals outwards', () => {
    const s = makeManifold('sphere', 200, 0, createRng(3));
    for (let i = 0; i < 200; i++)
      expect(Math.hypot(s.x.get(i, 0), s.x.get(i, 1), s.x.get(i, 2))).toBeCloseTo(1, 9);
    const r = makeManifold('swiss', 300, 0, createRng(3));
    // the radius in the x–z plane grows with the position along the roll (y = u)
    const radius = (i: number) => Math.hypot(r.x.get(i, 0), r.x.get(i, 2));
    const order = Array.from({ length: 300 }, (_, i) => i).sort(
      (a, b) => (r.y[a] as number) - (r.y[b] as number),
    );
    expect(radius(order[299] as number)).toBeGreaterThan(radius(order[0] as number));
  });
});
