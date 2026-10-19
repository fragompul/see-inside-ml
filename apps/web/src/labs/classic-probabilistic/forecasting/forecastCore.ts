import {
  acf,
  arimaFit,
  decompose,
  etsFit,
  ljungBox,
  mae,
  mape,
  rmse,
  type Decomposition,
} from '@sim/classic';
import { createRng } from '@sim/core';

/** The classic "airline passengers" series: monthly totals (thousands), January 1949 to December 1960. */
export const AIR_PASSENGERS: readonly number[] = [
  112, 118, 132, 129, 121, 135, 148, 148, 136, 119, 104, 118, 115, 126, 141, 135, 125, 149, 170,
  170, 158, 133, 114, 140, 145, 150, 178, 163, 172, 178, 199, 199, 184, 162, 146, 166, 171, 180,
  193, 181, 183, 218, 230, 242, 209, 191, 172, 194, 196, 196, 236, 235, 229, 243, 264, 272, 237,
  211, 180, 201, 204, 188, 235, 227, 234, 264, 302, 293, 259, 229, 203, 229, 242, 233, 267, 269,
  270, 315, 364, 347, 312, 274, 237, 278, 284, 277, 317, 313, 318, 374, 413, 405, 355, 306, 271,
  306, 315, 301, 356, 348, 355, 422, 465, 467, 404, 347, 305, 336, 340, 318, 362, 348, 363, 435,
  491, 505, 404, 359, 310, 337, 360, 342, 406, 396, 420, 472, 548, 559, 463, 407, 362, 405, 417,
  391, 419, 461, 472, 535, 622, 606, 508, 461, 390, 432,
];

export type SeriesId = 'airline' | 'seasonal' | 'walk' | 'cycle';

export interface SeriesInfo {
  id: SeriesId;
  label: string;
  note: string;
  period: number;
  values: number[];
}

function build(): Record<SeriesId, SeriesInfo> {
  const rng = createRng(12);
  const seasonal = Array.from(
    { length: 120 },
    (_, t) => 100 + 0.8 * t + 15 * Math.sin((2 * Math.PI * t) / 12) + rng.normal() * 3,
  );
  let w = 100;
  const walk = Array.from({ length: 120 }, () => (w += rng.normal() * 3));
  // a stationary cycle (AR(2) with complex roots) around a level
  const cycle: number[] = [];
  let a = 0;
  let b = 0;
  for (let t = 0; t < 150; t++) {
    const next = 1.5 * a - 0.8 * b + rng.normal() * 2;
    b = a;
    a = next;
    cycle.push(60 + next);
  }
  return {
    airline: {
      id: 'airline',
      label: 'Airline passengers',
      note: 'Real monthly totals: a steady climb, a strong yearly rhythm and swings that grow with the level.',
      period: 12,
      values: [...AIR_PASSENGERS],
    },
    seasonal: {
      id: 'seasonal',
      label: 'Trend and seasons',
      note: 'Synthetic: a straight upward trend, a regular yearly wave of constant size and a little noise.',
      period: 12,
      values: seasonal,
    },
    walk: {
      id: 'walk',
      label: 'Random walk',
      note: 'Synthetic: each value is the previous one plus a random step. It wanders, and nothing predicts where.',
      period: 12,
      values: walk,
    },
    cycle: {
      id: 'cycle',
      label: 'Stationary cycle',
      note: 'Synthetic: an oscillation that stays around a fixed level, an AR(2) process.',
      period: 12,
      values: cycle,
    },
  };
}

export const SERIES = build();

export type Method =
  'naive' | 'seasonal-naive' | 'drift' | 'ses' | 'holt' | 'holt-winters' | 'arima';

export const METHOD_LABELS: Record<Method, string> = {
  naive: 'Naive (last value)',
  'seasonal-naive': 'Seasonal naive',
  drift: 'Drift',
  ses: 'Simple smoothing',
  holt: 'Holt trend',
  'holt-winters': 'Holt-Winters',
  arima: 'ARIMA',
};

export interface ForecastScenario {
  series: SeriesId;
  method: Method;
  /** Model the logarithm of the series (makes growing seasonal swings constant). */
  log: boolean;
  holdout: number;
  p: number;
  d: number;
  q: number;
  /** One seasonal difference (lag = period). */
  seasonalDiff: boolean;
  /** One seasonal moving-average term. */
  seasonalMa: boolean;
}

export const DEFAULT_FORECAST: ForecastScenario = {
  series: 'airline',
  method: 'holt-winters',
  log: true,
  holdout: 24,
  p: 1,
  d: 1,
  q: 1,
  seasonalDiff: false,
  seasonalMa: false,
};

export interface Accuracy {
  mae: number;
  rmse: number;
  /** Mean absolute percentage error, as a fraction. */
  mape: number;
}

export interface ForecastRun {
  scenario: ForecastScenario;
  info: SeriesInfo;
  train: number[];
  test: number[];
  fitted: (number | null)[];
  forecast: number[];
  lower: number[];
  upper: number[];
  /** True when the band comes from the model's own formula rather than a rough square-root-of-horizon rule. */
  exactBand: boolean;
  accuracy: Accuracy;
  /** Accuracy of the two benchmarks on the same hold-out. */
  naive: Accuracy;
  seasonalNaive: Accuracy;
  /** One-step in-sample residuals (on the original scale) and their autocorrelation. */
  residuals: number[];
  residualAcf: number[];
  ljungBox: number;
  decomposition: Decomposition;
  aic: number | null;
  /** Share of the actual hold-out values that fall inside the band. */
  coverage: number;
}

const accuracy = (actual: number[], pred: number[]): Accuracy => ({
  mae: mae(actual, pred),
  rmse: rmse(actual, pred),
  mape: mape(actual, pred),
});

const sd = (v: readonly number[]): number => {
  const m = v.reduce((a, b) => a + b, 0) / (v.length || 1);
  return Math.sqrt(v.reduce((a, b) => a + (b - m) ** 2, 0) / Math.max(v.length - 1, 1));
};

interface Fit {
  fitted: (number | null)[];
  forecast: number[];
  lower: number[];
  upper: number[];
  exact: boolean;
  aic: number | null;
}

function fitMethod(
  method: Method,
  y: number[],
  h: number,
  period: number,
  s: ForecastScenario,
): Fit {
  const n = y.length;
  const roughBand = (fitted: (number | null)[], mean: number[], growth: (k: number) => number) => {
    const err = y
      .map((v, i) => (fitted[i] === null ? null : v - (fitted[i] as number)))
      .filter((v): v is number => v !== null);
    const sigma = sd(err);
    return {
      lower: mean.map((m, k) => m - 1.96 * sigma * Math.sqrt(growth(k + 1))),
      upper: mean.map((m, k) => m + 1.96 * sigma * Math.sqrt(growth(k + 1))),
    };
  };
  if (method === 'naive') {
    const fitted = y.map((_, i) => (i === 0 ? null : (y[i - 1] as number)));
    const mean = new Array<number>(h).fill(y[n - 1] as number);
    return {
      fitted,
      forecast: mean,
      ...roughBand(fitted, mean, (k) => k),
      exact: false,
      aic: null,
    };
  }
  if (method === 'seasonal-naive') {
    const fitted = y.map((_, i) => (i < period ? null : (y[i - period] as number)));
    const mean = Array.from({ length: h }, (_, k) => y[n - period + (k % period)] as number);
    return {
      fitted,
      forecast: mean,
      ...roughBand(fitted, mean, (k) => Math.floor((k - 1) / period) + 1),
      exact: false,
      aic: null,
    };
  }
  if (method === 'drift') {
    const slope = ((y[n - 1] as number) - (y[0] as number)) / (n - 1);
    const fitted = y.map((_, i) => (i === 0 ? null : (y[i - 1] as number) + slope));
    const mean = Array.from({ length: h }, (_, k) => (y[n - 1] as number) + slope * (k + 1));
    return {
      fitted,
      forecast: mean,
      ...roughBand(fitted, mean, (k) => k),
      exact: false,
      aic: null,
    };
  }
  if (method === 'ses' || method === 'holt' || method === 'holt-winters') {
    const fit =
      method === 'holt-winters'
        ? etsFit(y, 'holt-winters', { period, seasonal: 'additive' })
        : etsFit(y, method === 'ses' ? 'simple' : 'holt');
    const fitted: (number | null)[] = fit.fitted;
    const mean = fit.forecast(h);
    return {
      fitted,
      forecast: mean,
      ...roughBand(fitted, mean, (k) => k),
      exact: false,
      aic: null,
    };
  }
  const D = s.seasonalDiff ? 1 : 0;
  const fit = arimaFit(y, { p: s.p, d: s.d, q: s.q, D, Q: s.seasonalMa ? 1 : 0, period });
  const f = fit.forecast(h);
  // residuals live on the differenced scale: align them with the end of the series
  const lost = n - fit.residuals.length;
  const fitted: (number | null)[] = y.map((v, i) => {
    const j = i - lost;
    return j < s.p || j >= fit.residuals.length ? null : v - (fit.residuals[j] as number);
  });
  return { fitted, forecast: f.mean, lower: f.lower, upper: f.upper, exact: true, aic: fit.aic };
}

export function runForecast(s: ForecastScenario): ForecastRun {
  const info = SERIES[s.series];
  const all = info.values;
  const holdout = Math.min(s.holdout, Math.floor(all.length / 4));
  const trainRaw = all.slice(0, all.length - holdout);
  const test = all.slice(all.length - holdout);
  const fwd = (v: number) => (s.log ? Math.log(v) : v);
  const back = (v: number) => (s.log ? Math.exp(v) : v);
  const y = trainRaw.map(fwd);
  const fit = fitMethod(s.method, y, holdout, info.period, s);
  const forecast = fit.forecast.map(back);
  const lower = fit.lower.map(back);
  const upper = fit.upper.map(back);
  const fitted = fit.fitted.map((v) => (v === null ? null : back(v)));
  const residuals = trainRaw
    .map((v, i) => (fitted[i] === null ? null : v - (fitted[i] as number)))
    .filter((v): v is number => v !== null);
  const naive = fitMethod('naive', y, holdout, info.period, s).forecast.map(back);
  const snaive = fitMethod('seasonal-naive', y, holdout, info.period, s).forecast.map(back);
  const inside = test.filter(
    (v, i) => v >= (lower[i] as number) && v <= (upper[i] as number),
  ).length;
  return {
    scenario: s,
    info,
    train: trainRaw,
    test,
    fitted,
    forecast,
    lower,
    upper,
    exactBand: fit.exact,
    accuracy: accuracy(test, forecast),
    naive: accuracy(test, naive),
    seasonalNaive: accuracy(test, snaive),
    residuals,
    residualAcf: acf(residuals, 24),
    ljungBox: ljungBox(residuals, 12),
    decomposition: decompose(trainRaw, info.period, 'multiplicative'),
    aic: fit.aic,
    coverage: test.length ? inside / test.length : 0,
  };
}
