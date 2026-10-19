import { BayesianRidge, PolynomialRegression, polynomialFeatures } from '@sim/classic';
import { Matrix, Standardizer, cholesky, type Regressor, type Rng } from '@sim/core';
import { num, type ParamSpec, type RegressorLabSpec } from '@/shared/lab-kit';

const params: ParamSpec[] = [
  { key: 'degree', label: 'Polynomial degree', kind: 'slider', min: 1, max: 12, step: 1 },
];

/** Bayesian polynomial regression: powers of x, standardised, then `BayesianRidge`. */
export class BayesianPolynomial implements Regressor {
  readonly model = new BayesianRidge({ tolerance: 1e-8, maxIterations: 500 });
  private scaler = new Standardizer();
  private scale = 1;

  constructor(readonly degree: number) {}

  private features(x: Matrix): Matrix {
    return polynomialFeatures(
      Float64Array.from({ length: x.rows }, (_, i) => x.get(i, 0) / this.scale),
      this.degree,
    );
  }

  fit(x: Matrix, y: ArrayLike<number>): this {
    let m = 0;
    for (let i = 0; i < x.rows; i++) m = Math.max(m, Math.abs(x.get(i, 0)));
    this.scale = m || 1;
    const f = this.features(x);
    this.model.fit(this.scaler.fit(f).transform(f), y);
    return this;
  }

  predict(x: Matrix): Float64Array {
    return this.model.predict(this.scaler.transform(this.features(x)));
  }

  /** Predictive standard deviation (noise + uncertainty about the weights). */
  predictStd(x: Matrix): Float64Array {
    return this.model.predictStd(this.scaler.transform(this.features(x)));
  }

  /** Random curves with weights drawn from the posterior over the coefficients. */
  sampleCurves(x: Matrix, count: number, rng: Rng): Float64Array[] {
    const d = this.degree;
    const cov = new Matrix(d, d, Float64Array.from(this.model.covariance.data));
    for (let j = 0; j < d; j++) cov.set(j, j, cov.get(j, j) + 1e-10);
    const l = cholesky(cov);
    const z = this.scaler.transform(this.features(x));
    return Array.from({ length: count }, () => {
      const g = Float64Array.from({ length: d }, () => rng.normal());
      const w = Float64Array.from({ length: d }, (_, i) => {
        let s = this.model.coef[i] as number;
        for (let j = 0; j <= i; j++) s += l.get(i, j) * (g[j] as number);
        return s;
      });
      return Float64Array.from({ length: x.rows }, (_, r) => {
        let s = this.model.intercept;
        for (let j = 0; j < d; j++) s += z.get(r, j) * (w[j] as number);
        return s;
      });
    });
  }
}

export const bayesOf = (m: Regressor): BayesianPolynomial | null =>
  m instanceof BayesianPolynomial ? m : null;

/** The mean predictive standard deviation across the plotted range. */
export function meanStd(model: BayesianPolynomial, from: number, to: number): number {
  const xs = Matrix.from(Array.from({ length: 60 }, (_, i) => [from + ((to - from) * i) / 59]));
  const s = model.predictStd(xs);
  return s.reduce((a, b) => a + b, 0) / s.length;
}

export const plainPolynomial = (degree: number) => new PolynomialRegression({ degree });

export const bayesSpec: RegressorLabSpec = {
  id: 'bayesian-regression',
  create: (p) => new BayesianPolynomial(num(p, 'degree')),
  params,
  defaults: { degree: 5 },
  dataset: 'sine',
  data: { n: 30, noise: 0.3 },
  experiments: [
    {
      id: 'few',
      title: 'Only six points',
      text: 'With little data many curves fit equally well, so the posterior is wide: the band balloons between and beyond the points.',
      setup: { n: 8, testFraction: 0 },
      params: { degree: 5 },
      claim: (run) => meanStd(run.model as BayesianPolynomial, -3, 3) > 0.45,
    },
    {
      id: 'many',
      title: 'Two hundred points',
      text: 'More data pins the curve down: the band narrows to little more than the observation noise.',
      setup: { n: 200 },
      params: { degree: 5 },
      claim: (run) => meanStd(run.model as BayesianPolynomial, -3, 3) < 0.45,
    },
    {
      id: 'flexible',
      title: 'Degree 12 with few points',
      text: 'A twelve-degree polynomial would overfit, but the model chooses how strongly to shrink the weights from the data (evidence maximisation), so it stays tame.',
      setup: { n: 20 },
      params: { degree: 12 },
      claim: (run) => (run.scores.testR2 ?? 0) > 0.4,
    },
    {
      id: 'wrong-model',
      title: 'A line on a wave',
      text: 'A straight line cannot follow the wave, and the model has no way to know that: it explains the misfit as noise, estimating a large noise level and a wide band.',
      params: { degree: 1 },
      claim: (run) => {
        const m = run.model as BayesianPolynomial;
        return run.scores.trainR2 < 0.4 && 1 / Math.sqrt(m.model.alpha) > 1;
      },
    },
  ],
};
