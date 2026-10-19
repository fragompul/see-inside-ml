import { Matrix, chi2Cdf, normalCdf, solve } from '@sim/core';

/**
 * Survival analysis: time until an event when some subjects are only observed up to a time (censored) and may never be
 * seen to have the event. The Kaplan-Meier estimate of the survival curve, the log-rank test to compare groups, and the
 * Cox proportional-hazards regression (Breslow handling of ties) fitted by Newton's method on the partial likelihood.
 */
export interface KaplanMeier {
  /** Distinct times at which at least one event happened. */
  times: number[];
  atRisk: number[];
  events: number[];
  /** Survival probability just after each time, and its Greenwood standard error. */
  survival: number[];
  se: number[];
}

export function kaplanMeier(time: ArrayLike<number>, event: ArrayLike<number>): KaplanMeier {
  const n = time.length;
  const order = Array.from({ length: n }, (_, i) => i).sort(
    (a, b) => (time[a] as number) - (time[b] as number),
  );
  const times: number[] = [];
  const atRisk: number[] = [];
  const events: number[] = [];
  const survival: number[] = [];
  const se: number[] = [];
  let s = 1;
  let greenwood = 0;
  let i = 0;
  while (i < n) {
    const t = time[order[i] as number] as number;
    let deaths = 0;
    let leaving = 0;
    while (i + leaving < n && time[order[i + leaving] as number] === t) {
      deaths += event[order[i + leaving] as number] ? 1 : 0;
      leaving++;
    }
    const risk = n - i;
    if (deaths > 0) {
      s *= 1 - deaths / risk;
      greenwood += risk > deaths ? deaths / (risk * (risk - deaths)) : 0;
      times.push(t);
      atRisk.push(risk);
      events.push(deaths);
      survival.push(s);
      se.push(s * Math.sqrt(greenwood));
    }
    i += leaving;
  }
  return { times, atRisk, events, survival, se };
}

/** Survival probability at time t from a Kaplan-Meier estimate (a step function, 1 before the first event). */
export function survivalAt(km: KaplanMeier, t: number): number {
  let s = 1;
  for (let i = 0; i < km.times.length; i++) {
    if ((km.times[i] as number) <= t) s = km.survival[i] as number;
    else break;
  }
  return s;
}

/** The first time at which the estimated survival falls to `p` or below (Infinity if it never does). */
export function survivalQuantile(km: KaplanMeier, p = 0.5): number {
  for (let i = 0; i < km.times.length; i++)
    if ((km.survival[i] as number) <= p) return km.times[i] as number;
  return Infinity;
}

/** Log-rank test for two groups: the statistic (chi-square with one degree of freedom) and its p-value. */
export function logRank(
  time: ArrayLike<number>,
  event: ArrayLike<number>,
  group: ArrayLike<number>,
): { statistic: number; pValue: number; observed: number; expected: number } {
  const n = time.length;
  const times = Array.from(
    new Set(Array.from({ length: n }, (_, i) => time[i] as number).filter((_, i) => event[i])),
  ).sort((a, b) => a - b);
  let observed = 0;
  let expected = 0;
  let variance = 0;
  for (const t of times) {
    let n1 = 0;
    let n2 = 0;
    let d1 = 0;
    let d = 0;
    for (let i = 0; i < n; i++) {
      if ((time[i] as number) < t) continue;
      if (group[i] === 1) n1++;
      else n2++;
      if (time[i] === t && event[i]) {
        d++;
        if (group[i] === 1) d1++;
      }
    }
    const total = n1 + n2;
    observed += d1;
    expected += (d * n1) / total;
    if (total > 1) variance += (d * (n1 / total) * (n2 / total) * (total - d)) / (total - 1);
  }
  const statistic = variance > 0 ? (observed - expected) ** 2 / variance : 0;
  return { statistic, pValue: 1 - chi2Cdf(statistic, 1), observed, expected };
}

export interface CoxResult {
  coef: Float64Array;
  se: Float64Array;
  /** Hazard ratios exp(coef) with 95% confidence intervals. */
  hazardRatio: Float64Array;
  low: Float64Array;
  high: Float64Array;
  pValue: Float64Array;
  logLikelihood: number;
  iterations: number;
}

/** Cox regression: hazard(t | x) = baseline(t) exp(x . beta), by maximising the partial likelihood. */
export function coxRegression(
  time: ArrayLike<number>,
  event: ArrayLike<number>,
  x: Matrix,
  options: { iterations: number } = { iterations: 30 },
): CoxResult {
  const n = x.rows;
  const p = x.cols;
  const order = Array.from({ length: n }, (_, i) => i).sort(
    (a, b) => (time[b] as number) - (time[a] as number),
  ); // latest first
  const beta = new Float64Array(p);
  const evaluate = (b: Float64Array) => {
    let ll = 0;
    const grad = new Float64Array(p);
    const hess = new Matrix(p, p);
    // running sums over the risk set, adding subjects from the latest time backwards (Breslow ties)
    let s0 = 0;
    const s1 = new Float64Array(p);
    const s2 = new Matrix(p, p);
    let i = 0;
    while (i < n) {
      const t = time[order[i] as number] as number;
      let j = i;
      const group: number[] = [];
      while (j < n && time[order[j] as number] === t) group.push(order[j++] as number);
      for (const r of group) {
        let eta = 0;
        for (let k = 0; k < p; k++) eta += (x.data[r * p + k] as number) * (b[k] as number);
        const w = Math.exp(eta);
        s0 += w;
        for (let k = 0; k < p; k++) {
          s1[k] = (s1[k] as number) + w * (x.data[r * p + k] as number);
          for (let l = 0; l < p; l++)
            s2.data[k * p + l] =
              (s2.data[k * p + l] as number) +
              w * (x.data[r * p + k] as number) * (x.data[r * p + l] as number);
        }
      }
      for (const r of group) {
        if (!event[r]) continue;
        let eta = 0;
        for (let k = 0; k < p; k++) eta += (x.data[r * p + k] as number) * (b[k] as number);
        ll += eta - Math.log(s0);
        for (let k = 0; k < p; k++) {
          grad[k] = (grad[k] as number) + (x.data[r * p + k] as number) - (s1[k] as number) / s0;
          for (let l = 0; l < p; l++)
            hess.data[k * p + l] =
              (hess.data[k * p + l] as number) -
              ((s2.data[k * p + l] as number) / s0 -
                ((s1[k] as number) * (s1[l] as number)) / s0 ** 2);
        }
      }
      i = j;
    }
    return { ll, grad, hess };
  };
  let iterations = 0;
  for (; iterations < options.iterations; iterations++) {
    const { grad, hess } = evaluate(beta);
    const step = solve(
      hess,
      new Matrix(
        p,
        1,
        grad.map((g) => -g),
      ),
    ).data;
    let size = 0;
    for (let k = 0; k < p; k++) {
      beta[k] = (beta[k] as number) + (step[k] as number);
      size = Math.max(size, Math.abs(step[k] as number));
    }
    if (size < 1e-10) {
      iterations++;
      break;
    }
  }
  const final = evaluate(beta);
  const information = new Matrix(
    p,
    p,
    final.hess.data.map((v) => -v),
  );
  const covariance = solve(information, identityOf(p));
  const se = Float64Array.from({ length: p }, (_, k) =>
    Math.sqrt(covariance.data[k * p + k] as number),
  );
  const z = 1.959963984540054;
  return {
    coef: beta,
    se,
    hazardRatio: beta.map(Math.exp),
    low: Float64Array.from(beta, (b, k) => Math.exp(b - z * (se[k] as number))),
    high: Float64Array.from(beta, (b, k) => Math.exp(b + z * (se[k] as number))),
    pValue: Float64Array.from(beta, (b, k) => 2 * (1 - normalCdf(Math.abs(b / (se[k] as number))))),
    logLikelihood: final.ll,
    iterations,
  };
}

function identityOf(n: number): Matrix {
  const m = new Matrix(n, n);
  for (let i = 0; i < n; i++) m.data[i * n + i] = 1;
  return m;
}
