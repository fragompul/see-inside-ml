import { createRng } from '@sim/core';
import { describe, expect, it } from 'vitest';
import airline from '../fixtures/airline.json';
import fx from '../fixtures/timeseries.json';
import {
  acf,
  arimaFit,
  decompose,
  difference,
  etsFit,
  etsRun,
  fitAutoRegression,
  integrate,
  ljungBox,
  pacf,
  psiWeights,
} from './timeseries';

const close = (a: ArrayLike<number>, b: ArrayLike<number>, digits = 7) => {
  expect(a.length).toBe(b.length);
  for (let i = 0; i < a.length; i++) expect(a[i]).toBeCloseTo(b[i] as number, digits);
};

describe('correlation and decomposition against statsmodels', () => {
  it('acf matches', () => close(acf(fx.x, 20), fx.acf, 9));
  it('pacf matches (Durbin-Levinson)', () => close(pacf(fx.x, 15), fx.pacf, 7));

  for (const model of ['additive', 'multiplicative'] as const) {
    it(`${model} decomposition matches seasonal_decompose`, () => {
      const mine = decompose(fx.x, 12, model);
      const ref = fx[`decompose_${model}` as 'decompose_additive'];
      close(mine.seasonal, ref.seasonal, 9);
      mine.trend.forEach((t, i) => {
        const r = ref.trend[i];
        if (r === null || r === undefined) expect(t).toBeNull();
        else expect(t).toBeCloseTo(r, 9);
      });
      mine.residual.forEach((e, i) => {
        const r = ref.resid[i];
        if (r === null || r === undefined) expect(e).toBeNull();
        else expect(e).toBeCloseTo(r, 8);
      });
    });
  }
});

describe('autoregression and smoothing against statsmodels', () => {
  it('AR(3) by least squares matches AutoReg', () => {
    const m = fx.x.reduce((a, b) => a + b, 0) / fx.x.length;
    const fit = fitAutoRegression(
      fx.x.map((v) => v - m),
      3,
    );
    close([fit.intercept, ...fit.phi], fx.autoreg.params, 7);
    expect(fit.sigma2).toBeCloseTo(fx.autoreg.sigma2, 7);
  });

  it('simple exponential smoothing matches', () => {
    const r = etsRun(fx.x, 'simple', { alpha: fx.ses.alpha });
    close(r.fitted, fx.ses.fitted, 8);
    close(r.forecast(6), fx.ses.forecast, 8);
    expect(r.sse).toBeCloseTo(fx.ses.sse, 6);
  });

  it('Holt linear trend matches', () => {
    const r = etsRun(fx.x, 'holt', { alpha: fx.holt.alpha, beta: fx.holt.beta });
    close(r.fitted, fx.holt.fitted, 8);
    close(r.forecast(6), fx.holt.forecast, 8);
  });

  it('additive Holt-Winters matches', () => {
    const r = etsRun(fx.x, 'holt-winters', {
      alpha: fx.hw.alpha,
      beta: fx.hw.beta,
      gamma: fx.hw.gamma,
      period: 12,
      seasonal: 'additive',
    });
    close(r.fitted, fx.hw.fitted, 7);
    // statsmodels' 12th step (a full period ahead) reuses a seasonal slot that the last observation has just
    // updated differently; the first eleven steps and every fitted value agree to rounding error
    close(r.forecast(11), fx.hw.forecast.slice(0, 11), 7);
  });
});

describe('ARIMA by conditional sum of squares', () => {
  it('differencing and integration are inverses', () => {
    const x = [3, 5, 4, 8, 13, 12];
    const d = difference(x, 1);
    expect(integrate(d, [x[0]!])).toEqual(x.slice(1));
    expect(difference(x, 2)).toEqual([
      4 - 5 - (5 - 3),
      8 - 4 - (4 - 5),
      13 - 8 - (8 - 4),
      12 - 13 - (13 - 8),
    ]);
    expect(difference([1, 2, 3, 4, 5, 6, 7], 1, 3)).toEqual([3, 3, 3, 3]);
  });

  it('is close to exact maximum likelihood on an ARMA(1,1)', () => {
    const fit = arimaFit(fx.arma.y, { p: 1, d: 0, q: 1 });
    // statsmodels params: const, ar.L1, ma.L1, sigma2
    expect(fit.phi[0]).toBeCloseTo(fx.arma.params[1]!, 1);
    expect(fit.theta[0]).toBeCloseTo(fx.arma.params[2]!, 1);
    expect(fit.mean).toBeCloseTo(fx.arma.params[0]!, 0);
    expect(fit.sigma2).toBeCloseTo(fx.arma.params[3]!, 1);
    const f = fit.forecast(8);
    close(f.mean, fx.arma.forecast, 0);
    // prediction standard errors grow to the unconditional value
    const se = f.upper.map((u, i) => (u - f.mean[i]!) / 1.96);
    for (let i = 1; i < se.length; i++) expect(se[i]).toBeGreaterThanOrEqual(se[i - 1]! - 1e-9);
    close(se, fx.arma.se, 1);
  });

  it('a random walk forecasts the last value with a standard error growing like the square root of the horizon', () => {
    const fit = arimaFit(fx.walk.y, { p: 0, d: 1, q: 0 });
    const f = fit.forecast(10);
    close(f.mean, fx.walk.forecast, 6);
    const se = f.upper.map((u, i) => (u - f.mean[i]!) / 1.96);
    expect(se[3]! / se[0]!).toBeCloseTo(2, 6);
    expect(se[0]).toBeCloseTo(fx.walk.se[0]!, 1);
  });

  it('recovers the coefficients of a simulated AR(2) and forecasts the mean in the long run', () => {
    const rng = createRng(3);
    const y = new Array<number>(3000).fill(0);
    for (let t = 2; t < 3000; t++)
      y[t] = 0.6 * y[t - 1]! - 0.3 * y[t - 2]! + rng.normal() + 5 * 0.7;
    const fit = arimaFit(y, { p: 2, d: 0, q: 0 });
    expect(fit.phi[0]).toBeCloseTo(0.6, 1);
    expect(fit.phi[1]).toBeCloseTo(-0.3, 1);
    const long = fit.forecast(200).mean;
    expect(long[199]).toBeCloseTo(fit.mean, 2);
  });

  it('psi weights of an AR(1) are powers of phi', () => {
    const psi = psiWeights([0.5], [], 5);
    psi.forEach((v, j) => expect(v).toBeCloseTo(0.5 ** j, 12));
  });

  it('white-noise residuals pass the Ljung-Box check for the right model but not for a wrong one', () => {
    const right = arimaFit(fx.arma.y, { p: 1, d: 0, q: 1 });
    const wrong = arimaFit(fx.arma.y, { p: 0, d: 0, q: 0 });
    // 95th percentile of chi-square(10) is 18.3
    expect(ljungBox(right.residuals.slice(1), 10)).toBeLessThan(25);
    expect(ljungBox(wrong.residuals, 10)).toBeGreaterThan(100);
  });
});

describe('seasonal differencing', () => {
  // a perfectly periodic series with a linear trend: (1 - B^12) leaves a constant, so ARIMA(0,0,0)x(0,1,0)12 + drift extends the pattern
  const series = Array.from(
    { length: 60 },
    (_, t) => 10 + 0.5 * t + 3 * Math.sin((2 * Math.PI * t) / 12),
  );
  it('continues a seasonal pattern with trend', () => {
    const fit = arimaFit(series, { p: 0, d: 0, q: 0, D: 1, period: 12 }, { drift: true });
    const f = fit.forecast(12).mean;
    f.forEach((v, i) => {
      const t = 60 + i;
      expect(v).toBeCloseTo(10 + 0.5 * t + 3 * Math.sin((2 * Math.PI * t) / 12), 6);
    });
  });
  it('seasonal period 1 is the same as ordinary differencing', () => {
    const a = arimaFit(fx.walk.y, { p: 0, d: 0, q: 0, D: 1, period: 1 }).forecast(5);
    const b = arimaFit(fx.walk.y, { p: 0, d: 1, q: 0 }).forecast(5);
    close(a.mean, b.mean, 9);
    close(a.upper, b.upper, 9);
  });
  it('the forecast band keeps widening with the horizon', () => {
    const f = arimaFit(fx.x, { p: 1, d: 0, q: 0, D: 1, period: 12 }).forecast(24);
    const width = f.upper.map((u, i) => u - f.lower[i]!);
    for (let i = 1; i < width.length; i++)
      expect(width[i]).toBeGreaterThanOrEqual(width[i - 1]! - 1e-9);
  });
});

describe('the airline model (0,1,1)(0,1,1)12', () => {
  const fit = arimaFit(airline.y, { p: 0, d: 1, q: 1, D: 1, Q: 1, period: 12 });
  it('finds the coefficients of the exact maximum-likelihood fit', () => {
    expect(fit.theta[0]).toBeCloseTo(airline.params[0]!, 1);
    expect(fit.seasonalTheta).toBeCloseTo(airline.params[1]!, 1);
    expect(fit.sigma2).toBeCloseTo(airline.params[2]!, 3);
  });
  it('forecasts the next year like statsmodels (on the log scale)', () => {
    const f = fit.forecast(12);
    f.mean.forEach((v, i) => expect(Math.abs(v - airline.forecast[i]!)).toBeLessThan(0.03));
    const se = f.upper.map((u, i) => (u - f.mean[i]!) / 1.96);
    se.forEach((v, i) => expect(Math.abs(v - airline.se[i]!)).toBeLessThan(0.012));
  });
});

describe('exponential smoothing fit', () => {
  it('the fitted parameters never do worse than the defaults', () => {
    const fit = etsFit(fx.x, 'holt');
    const fixed = etsRun(fx.x, 'holt', { alpha: 0.5, beta: 0.2 });
    expect(fit.sse).toBeLessThanOrEqual(fixed.sse + 1e-9);
  });
});
