import { describe, expect, it } from 'vitest';
import { AIR_PASSENGERS, DEFAULT_FORECAST, SERIES, runForecast } from './forecastCore';
import { forecastExperiments } from './forecastSpec';

describe('forecasting lab', () => {
  for (const exp of forecastExperiments) {
    it(`experiment "${exp.title}" holds`, () => {
      expect(exp.claim(runForecast({ ...DEFAULT_FORECAST, ...exp.scenario }))).toBe(true);
    });
  }

  it('the embedded airline series is the classic one (144 months, mean 280.3)', () => {
    expect(AIR_PASSENGERS).toHaveLength(144);
    const mean = AIR_PASSENGERS.reduce((a, b) => a + b, 0) / 144;
    expect(mean).toBeCloseTo(280.2986, 3);
    expect(AIR_PASSENGERS[0]).toBe(112);
    expect(AIR_PASSENGERS[143]).toBe(432);
  });

  it('every series is positive, so the log scale is always available', () => {
    for (const s of Object.values(SERIES)) for (const v of s.values) expect(v).toBeGreaterThan(0);
  });

  it('the hold-out is cut from the end and never leaks into the fit', () => {
    const r = runForecast({ ...DEFAULT_FORECAST, holdout: 12 });
    expect(r.train).toHaveLength(132);
    expect(r.test).toEqual(AIR_PASSENGERS.slice(132));
    const changed = runForecast({ ...DEFAULT_FORECAST, holdout: 12 });
    expect(changed.forecast).toEqual(r.forecast);
  });
});
