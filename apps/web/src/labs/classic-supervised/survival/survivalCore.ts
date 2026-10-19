import { Matrix, createRng } from '@sim/core';
import {
  coxRegression,
  kaplanMeier,
  logRank,
  survivalAt,
  survivalQuantile,
  type CoxResult,
  type KaplanMeier,
} from '@sim/classic';

/**
 * Time-to-event data for two groups. Events happen at a constant rate (0.1 per month in the first group, 0.1 times the
 * hazard ratio in the second); follow-up ends at a random time between 0 and `followup`, so some subjects are only seen
 * to survive up to that time (censored).
 */
export interface SurvivalScenario {
  n: number;
  hazardRatio: number;
  /** Longest follow-up in months. */
  followup: number;
  seed: number;
}

export const DEFAULT_SURVIVAL: SurvivalScenario = {
  n: 400,
  hazardRatio: 2.5,
  followup: 14,
  seed: 6,
};

export const BASE_RATE = 0.1;
export const trueMedian = (rate: number): number => Math.LN2 / rate;

export interface SurvivalData {
  time: number[];
  event: number[];
  group: number[];
}

export function simulate(s: SurvivalScenario): SurvivalData {
  const rng = createRng(s.seed);
  const time: number[] = [];
  const event: number[] = [];
  const group: number[] = [];
  for (let i = 0; i < s.n; i++) {
    const g = i % 2;
    const rate = BASE_RATE * (g === 1 ? s.hazardRatio : 1);
    const t = -Math.log(1 - rng.next()) / rate;
    const c = rng.next() * s.followup;
    time.push(Math.min(t, c));
    event.push(t <= c ? 1 : 0);
    group.push(g);
  }
  return { time, event, group };
}

export interface SurvivalResult {
  scenario: SurvivalScenario;
  data: SurvivalData;
  censored: number;
  km: [KaplanMeier, KaplanMeier];
  /** Time grid and, per group, the Kaplan-Meier curve with a 95% band and the true curve. */
  grid: number[];
  curves: { estimate: number[]; low: number[]; high: number[]; truth: number[] }[];
  /** Median survival: estimated (Infinity if the curve never falls to a half), true, and the naive estimate that treats every censored time as an event. */
  median: { estimate: number; truth: number; naive: number }[];
  logRank: { statistic: number; pValue: number };
  cox: CoxResult;
  trueHazardRatio: number;
}

const medianOf = (v: number[]): number => {
  const s = [...v].sort((a, b) => a - b);
  const m = Math.floor(s.length / 2);
  return s.length % 2 ? (s[m] as number) : ((s[m - 1] as number) + (s[m] as number)) / 2;
};

export function runSurvival(s: SurvivalScenario): SurvivalResult {
  const data = simulate(s);
  const byGroup = [0, 1].map((g) => {
    const idx = data.group.map((v, i) => (v === g ? i : -1)).filter((i) => i >= 0);
    return {
      time: idx.map((i) => data.time[i] as number),
      event: idx.map((i) => data.event[i] as number),
    };
  });
  const km = byGroup.map((d) => kaplanMeier(d.time, d.event)) as [KaplanMeier, KaplanMeier];
  const top = Math.min(40, Math.max(...data.time) * 1.05);
  const grid = Array.from({ length: 81 }, (_, i) => (i / 80) * top);
  const curves = km.map((k, g) => {
    const rate = BASE_RATE * (g === 1 ? s.hazardRatio : 1);
    const at = (t: number) => {
      let idx = -1;
      for (let i = 0; i < k.times.length; i++) if ((k.times[i] as number) <= t) idx = i;
      return idx;
    };
    return {
      estimate: grid.map((t) => survivalAt(k, t)),
      low: grid.map((t) => {
        const i = at(t);
        return i < 0 ? 1 : Math.max(0, (k.survival[i] as number) - 1.96 * (k.se[i] as number));
      }),
      high: grid.map((t) => {
        const i = at(t);
        return i < 0 ? 1 : Math.min(1, (k.survival[i] as number) + 1.96 * (k.se[i] as number));
      }),
      truth: grid.map((t) => Math.exp(-rate * t)),
    };
  });
  const x = new Matrix(s.n, 1, Float64Array.from(data.group));
  return {
    scenario: s,
    data,
    censored: data.event.filter((e) => !e).length / s.n,
    km,
    grid,
    curves,
    median: [0, 1].map((g) => ({
      estimate: survivalQuantile(km[g] as KaplanMeier, 0.5),
      truth: trueMedian(BASE_RATE * (g === 1 ? s.hazardRatio : 1)),
      naive: medianOf((byGroup[g] as { time: number[] }).time),
    })),
    logRank: logRank(data.time, data.event, data.group),
    cox: coxRegression(data.time, data.event, x),
    trueHazardRatio: s.hazardRatio,
  };
}
