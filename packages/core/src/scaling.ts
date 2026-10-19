/**
 * Fitting a scaling law  y = c + a x^(-alpha)  to measurements, as is done for loss against model size or
 * data size. With the irreducible part `c` known the fit is a straight line in log-log space; without it,
 * `c` is searched over a grid for the value that makes the line straightest.
 */
export interface PowerLaw {
  a: number;
  alpha: number;
  c: number;
  /** Coefficient of determination of the straight-line fit in log-log space. */
  r2: number;
  predict: (x: number) => number;
}

function lineFit(u: number[], v: number[]): { slope: number; intercept: number; r2: number } {
  const n = u.length;
  const mu = u.reduce((s, x) => s + x, 0) / n;
  const mv = v.reduce((s, x) => s + x, 0) / n;
  let suv = 0;
  let suu = 0;
  let svv = 0;
  for (let i = 0; i < n; i++) {
    suv += ((u[i] as number) - mu) * ((v[i] as number) - mv);
    suu += ((u[i] as number) - mu) ** 2;
    svv += ((v[i] as number) - mv) ** 2;
  }
  const slope = suv / suu;
  return { slope, intercept: mv - slope * mu, r2: svv === 0 ? 1 : (suv * suv) / (suu * svv) };
}

export function fitPowerLaw(x: readonly number[], y: readonly number[], floor?: number): PowerLaw {
  const lx = x.map(Math.log);
  const attempt = (c: number) => {
    const fit = lineFit(
      lx,
      y.map((v) => Math.log(v - c)),
    );
    return { c, a: Math.exp(fit.intercept), alpha: -fit.slope, r2: fit.r2 };
  };
  let best = attempt(0);
  if (floor !== undefined) best = attempt(floor);
  else {
    const top = Math.min(...y);
    for (let i = 0; i < 400; i++) {
      const candidate = attempt((top * 0.999 * i) / 399);
      if (candidate.r2 > best.r2) best = candidate;
    }
  }
  return { ...best, predict: (q) => best.c + best.a * q ** -best.alpha };
}
