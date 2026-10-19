import type { ParamSpec } from '@/shared/lab-kit';
import {
  METHOD_LABELS,
  SERIES,
  type ForecastRun,
  type ForecastScenario,
  type Method,
  type SeriesId,
} from './forecastCore';

export const forecastParams: ParamSpec[] = [
  {
    key: 'series',
    label: 'Series',
    kind: 'segmented',
    options: (Object.keys(SERIES) as SeriesId[]).map((id) => ({
      value: id,
      label: SERIES[id].label,
    })),
  },
  {
    key: 'method',
    label: 'Forecasting method',
    kind: 'select',
    options: (Object.keys(METHOD_LABELS) as Method[]).map((m) => ({
      value: m,
      label: METHOD_LABELS[m],
    })),
  },
  {
    key: 'log',
    label: 'Work with the logarithm',
    kind: 'switch',
    hint: 'Turns growing swings into constant ones; forecasts are converted back.',
  },
  {
    key: 'holdout',
    label: 'Months held back to test the forecast',
    kind: 'ladder',
    values: [6, 12, 24, 36],
    format: (v) => String(v),
  },
  {
    key: 'p',
    label: 'AR order p',
    kind: 'slider',
    min: 0,
    max: 4,
    step: 1,
    showIf: (v) => v.method === 'arima',
    hint: 'How many past values the next one depends on.',
  },
  {
    key: 'd',
    label: 'Differences d',
    kind: 'slider',
    min: 0,
    max: 2,
    step: 1,
    showIf: (v) => v.method === 'arima',
    hint: 'How many times to difference the series to remove trend.',
  },
  {
    key: 'q',
    label: 'MA order q',
    kind: 'slider',
    min: 0,
    max: 4,
    step: 1,
    showIf: (v) => v.method === 'arima',
    hint: 'How many past forecast errors the next value depends on.',
  },
  {
    key: 'seasonalDiff',
    label: 'Seasonal difference (compare with a year ago)',
    kind: 'switch',
    showIf: (v) => v.method === 'arima',
  },
  {
    key: 'seasonalMa',
    label: 'Seasonal moving-average term',
    kind: 'switch',
    showIf: (v) => v.method === 'arima',
    hint: 'With (0,1,1) and a seasonal difference this is the classic airline model.',
  },
];

export interface ForecastExperiment {
  id: string;
  title: string;
  text: string;
  scenario: Partial<ForecastScenario>;
  claim: (run: ForecastRun) => boolean;
}

export const forecastExperiments: ForecastExperiment[] = [
  {
    id: 'benchmarks',
    title: 'Benchmarks first',
    text: 'The naive forecast (tomorrow = today) ignores the yearly rhythm and misses by about a quarter. Repeating last year is already clearly better. Any serious method has to beat these two.',
    scenario: { method: 'naive' },
    claim: (r) =>
      r.accuracy.mape > 0.2 && r.seasonalNaive.mape < 0.18 && r.naive.mape > r.seasonalNaive.mape,
  },
  {
    id: 'holt-winters',
    title: 'Level, trend and season',
    text: 'Holt-Winters smooths a level, a trend and a seasonal pattern separately and extrapolates all three. On the logarithm of the airline series it cuts the error to a fraction of the benchmarks.',
    scenario: { method: 'holt-winters', log: true },
    claim: (r) => r.accuracy.mape < 0.05 && r.accuracy.mape < r.seasonalNaive.mape / 2,
  },
  {
    id: 'no-log',
    title: 'Without the logarithm',
    text: 'The seasonal swings of the airline series grow with its level, which an additive seasonal pattern cannot follow. Working on the log scale (swings become constant) improves the forecast.',
    scenario: { method: 'holt-winters', log: false },
    claim: (r) => r.accuracy.mape > 0.06,
  },
  {
    id: 'plain-arima',
    title: 'ARIMA that ignores the seasons',
    text: 'An ordinary ARIMA(1,1,1) captures the trend but not the yearly cycle: it forecasts a smooth curve through the zig-zag, and the residuals are far from white noise (the Ljung-Box statistic is huge).',
    scenario: { method: 'arima', p: 1, d: 1, q: 1, seasonalDiff: false, seasonalMa: false },
    claim: (r) => r.accuracy.mape > 0.18 && r.ljungBox > 50,
  },
  {
    id: 'airline',
    title: 'The airline model',
    text: 'Box and Jenkins’ famous model for this very series: difference once ordinarily and once seasonally, with one moving-average term at each scale. Its residuals look like noise and a year of forecasts is within a few percent.',
    scenario: {
      method: 'arima',
      p: 0,
      d: 1,
      q: 1,
      seasonalDiff: true,
      seasonalMa: true,
      holdout: 12,
    },
    claim: (r) => r.accuracy.mape < 0.04 && r.ljungBox < 20 && r.coverage >= 0.9,
  },
  {
    id: 'random-walk',
    title: 'A random walk cannot be forecast',
    text: 'For a series that is a sum of random steps the best forecast is the last value, and the uncertainty grows like the square root of the horizon: 24 steps ahead the band is about five times as wide as one step ahead.',
    scenario: { series: 'walk', method: 'arima', p: 0, d: 1, q: 0, log: false },
    claim: (r) => {
      const ratio =
        (r.upper[r.upper.length - 1]! - r.lower[r.lower.length - 1]!) / (r.upper[0]! - r.lower[0]!);
      return new Set(r.forecast.map((v) => v.toFixed(6))).size === 1 && ratio > 4.5 && ratio < 5.3;
    },
  },
  {
    id: 'cycle',
    title: 'A cycle needs two lags',
    text: 'An oscillation that returns to a level is an AR(2): the next value depends on the last two. With p = 2 the residuals are white and the forecast beats the naive one by a wide margin; with p = 1 the model cannot represent a cycle and the residuals show it.',
    scenario: { series: 'cycle', method: 'arima', p: 2, d: 0, q: 0, log: false },
    claim: (r) => r.accuracy.mape < r.naive.mape / 2 && r.ljungBox < 15,
  },
  {
    id: 'cycle-too-simple',
    title: 'Too simple a model',
    text: 'The same cycle with an AR(1): the model cannot oscillate, so the pattern stays in the residuals (Ljung-Box far above the noise level).',
    scenario: { series: 'cycle', method: 'arima', p: 1, d: 0, q: 0, log: false },
    claim: (r) => r.ljungBox > 100,
  },
  {
    id: 'trend-season',
    title: 'Clean trend and season',
    text: 'On a synthetic series made of exactly a line, a sine wave and noise, Holt-Winters recovers the structure almost perfectly.',
    scenario: { series: 'seasonal', method: 'holt-winters', log: false },
    claim: (r) => r.accuracy.mape < 0.03 && r.accuracy.mape < r.naive.mape / 3,
  },
  {
    id: 'overconfident-band',
    title: 'A band is only as good as its model',
    text: 'The seasonal-naive forecast comes with a rough band based on its own past errors, yet the band catches only a quarter of what really happens: the method is simply wrong about the trend.',
    scenario: { series: 'seasonal', method: 'seasonal-naive', log: false },
    claim: (r) => r.coverage <= 0.5,
  },
];
