/**
 * Nelder–Mead simplex search: minimises a function of several variables using only function values
 * (no gradients). Used to tune a few hyperparameters, where a gradient is unavailable or not worth it.
 */
export interface NelderMeadOptions {
  /** Edge length of the initial simplex (default 0.5). */
  step?: number;
  maxIterations?: number;
  /** Stop when the simplex's function values differ by less than this. */
  tolerance?: number;
}

export interface NelderMeadResult {
  x: number[];
  value: number;
  iterations: number;
  evaluations: number;
}

export function nelderMead(
  f: (x: readonly number[]) => number,
  start: readonly number[],
  options: NelderMeadOptions = {},
): NelderMeadResult {
  const { step = 0.5, maxIterations = 500, tolerance = 1e-10 } = options;
  const d = start.length;
  let evaluations = 0;
  const evalAt = (x: number[]) => {
    evaluations++;
    const v = f(x);
    return Number.isFinite(v) ? v : Infinity;
  };
  let simplex: { x: number[]; v: number }[] = [{ x: [...start], v: evalAt([...start]) }];
  for (let i = 0; i < d; i++) {
    const x = [...start];
    x[i] = (x[i] as number) + step;
    simplex.push({ x, v: evalAt(x) });
  }
  let iterations = 0;
  for (; iterations < maxIterations; iterations++) {
    simplex.sort((a, b) => a.v - b.v);
    const best = simplex[0]!;
    const worst = simplex[d]!;
    if (Math.abs(worst.v - best.v) <= tolerance * (1 + Math.abs(best.v))) break;
    const centroid = Array.from(
      { length: d },
      (_, j) => simplex.slice(0, d).reduce((s, p) => s + (p.x[j] as number), 0) / d,
    );
    const along = (t: number) => centroid.map((c, j) => c + t * ((worst.x[j] as number) - c));
    const reflected = along(-1);
    const rv = evalAt(reflected);
    if (rv < best.v) {
      const expanded = along(-2);
      const ev = evalAt(expanded);
      simplex[d] = ev < rv ? { x: expanded, v: ev } : { x: reflected, v: rv };
    } else if (rv < (simplex[d - 1] as { v: number }).v) {
      simplex[d] = { x: reflected, v: rv };
    } else {
      const contracted = along(rv < worst.v ? -0.5 : 0.5);
      const cv = evalAt(contracted);
      if (cv < Math.min(rv, worst.v)) simplex[d] = { x: contracted, v: cv };
      else
        simplex = simplex.map((p, i) => {
          if (i === 0) return p;
          const x = p.x.map((v, j) => (best.x[j] as number) + 0.5 * (v - (best.x[j] as number)));
          return { x, v: evalAt(x) };
        });
    }
  }
  simplex.sort((a, b) => a.v - b.v);
  const best = simplex[0]!;
  return { x: best.x, value: best.v, iterations, evaluations };
}
