import { Matrix } from './matrix';
import { symmetricEigen } from './linalg';
import { createRng, type Rng } from './rng';

/**
 * Derivative-free global search: algorithms that only evaluate the function. Each one is a
 * generator of "generations" so that the whole population can be drawn and compared.
 *   - Genetic algorithm: selection, crossover and mutation of real-valued genomes.
 *   - Simulated annealing: a single walker that accepts uphill moves with probability exp(-delta / T).
 *   - Particle swarm: points with velocity pulled towards their own and the swarm's best.
 *   - CMA-ES: samples from a Gaussian whose mean, scale and covariance adapt to the function.
 */
export interface Objective {
  id: string;
  label: string;
  note: string;
  f(x: readonly number[]): number;
  bounds: { x0: number; x1: number; y0: number; y1: number };
  /** Location and value of the global minimum. */
  optimum: { x: [number, number]; f: number };
}

export const OBJECTIVES: Record<string, Objective> = {
  sphere: {
    id: 'sphere',
    label: 'Sphere',
    note: 'A smooth bowl with one minimum: the sanity check.',
    f: (x) => x.reduce((s, v) => s + v * v, 0),
    bounds: { x0: -5, x1: 5, y0: -5, y1: 5 },
    optimum: { x: [0, 0], f: 0 },
  },
  ellipsoid: {
    id: 'ellipsoid',
    label: 'Tilted ellipsoid',
    note: 'A very elongated bowl at a 30-degree tilt: the right moves are not along the axes.',
    f: (x) => {
      const c = Math.cos(Math.PI / 6);
      const s = Math.sin(Math.PI / 6);
      const u = c * (x[0] as number) + s * (x[1] as number);
      const v = -s * (x[0] as number) + c * (x[1] as number);
      return u * u + 100 * v * v;
    },
    bounds: { x0: -5, x1: 5, y0: -5, y1: 5 },
    optimum: { x: [0, 0], f: 0 },
  },
  rastrigin: {
    id: 'rastrigin',
    label: 'Rastrigin',
    note: 'A bowl covered in a regular grid of small pits: about a hundred local minima in this window.',
    f: (x) => 20 + x.reduce((s, v) => s + v * v - 10 * Math.cos(2 * Math.PI * v), 0),
    bounds: { x0: -5.12, x1: 5.12, y0: -5.12, y1: 5.12 },
    optimum: { x: [0, 0], f: 0 },
  },
  ackley: {
    id: 'ackley',
    label: 'Ackley',
    note: 'A nearly flat outer plain with ripples, and a single deep funnel at the centre.',
    f: (x) => {
      const n = x.length;
      const a = x.reduce((s, v) => s + v * v, 0) / n;
      const b = x.reduce((s, v) => s + Math.cos(2 * Math.PI * v), 0) / n;
      return -20 * Math.exp(-0.2 * Math.sqrt(a)) - Math.exp(b) + 20 + Math.E;
    },
    bounds: { x0: -5, x1: 5, y0: -5, y1: 5 },
    optimum: { x: [0, 0], f: 0 },
  },
  deceptive: {
    id: 'deceptive',
    label: 'Deceptive valleys',
    note: 'A wide, shallow decoy basin far from a narrow, deeper one: greedy methods are drawn to the wrong place.',
    f: (x) => {
      const g = (cx: number, cy: number, w: number, depth: number) =>
        depth *
        Math.exp(-(((x[0] as number) - cx) ** 2 + ((x[1] as number) - cy) ** 2) / (2 * w * w));
      return 1 - g(-2.5, 2.5, 1.6, 0.7) - g(3, -3, 0.45, 0.95);
    },
    bounds: { x0: -5, x1: 5, y0: -5, y1: 5 },
    optimum: {
      x: [3, -3],
      f: 1 - 0.95 - 0.7 * Math.exp(-((5.5 ** 2 + 5.5 ** 2) / (2 * 1.6 * 1.6))),
    },
  },
};

export type Heuristic = 'genetic' | 'annealing' | 'swarm' | 'cmaes';
export const HEURISTICS: Heuristic[] = ['genetic', 'annealing', 'swarm', 'cmaes'];

export interface Generation {
  generation: number;
  /** The points evaluated in this generation (the population, swarm, or the walker). */
  points: [number, number][];
  values: number[];
  best: { x: [number, number]; f: number };
  /** Total function evaluations so far. */
  evaluations: number;
  /** CMA-ES only: current mean, step size and covariance (to draw the search distribution). */
  distribution?: {
    mean: [number, number];
    sigma: number;
    cov: [[number, number], [number, number]];
  };
  /** Annealing only: the current temperature. */
  temperature?: number;
}

export interface HeuristicOptions {
  population: number;
  generations: number;
  seed: number;
}

const clampTo = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v));

/** The running best only ever improves; the generators below keep it. */
function better(best: { x: [number, number]; f: number }, x: [number, number], f: number) {
  if (f < best.f) {
    best.x = [x[0], x[1]];
    best.f = f;
  }
}

export function* geneticAlgorithm(
  obj: Objective,
  options: HeuristicOptions,
): Generator<Generation, void, undefined> {
  const rng = createRng(options.seed);
  const { x0, x1, y0, y1 } = obj.bounds;
  const n = options.population;
  const evalAt = (p: [number, number]) => obj.f(p);
  let pop: [number, number][] = Array.from({ length: n }, () => [
    rng.uniform(x0, x1),
    rng.uniform(y0, y1),
  ]);
  let vals = pop.map(evalAt);
  let evaluations = n;
  const best = { x: [pop[0]![0], pop[0]![1]] as [number, number], f: vals[0] as number };
  pop.forEach((p, i) => better(best, p, vals[i] as number));
  yield {
    generation: 0,
    points: pop.map((p) => [...p] as [number, number]),
    values: [...vals],
    best: { ...best, x: [...best.x] as [number, number] },
    evaluations,
  };
  const tournament = () => {
    let w = Math.floor(rng.next() * n);
    for (let k = 0; k < 2; k++) {
      const c = Math.floor(rng.next() * n);
      if ((vals[c] as number) < (vals[w] as number)) w = c;
    }
    return pop[w] as [number, number];
  };
  for (let g = 1; g <= options.generations; g++) {
    const sigma = 0.4 * (1 - g / (options.generations + 1)) + 0.05;
    const kids: [number, number][] = [];
    // elitism: the best individual survives unchanged
    const eliteIdx = vals.indexOf(Math.min(...vals));
    kids.push([...(pop[eliteIdx] as [number, number])] as [number, number]);
    while (kids.length < n) {
      const a = tournament();
      const b = tournament();
      // blend crossover: a random point in the box spanned by the parents, widened by 30%
      const child = [0, 1].map((d) => {
        const lo = Math.min(a[d] as number, b[d] as number);
        const hi = Math.max(a[d] as number, b[d] as number);
        const w = hi - lo;
        let v = rng.uniform(lo - 0.3 * w, hi + 0.3 * w);
        if (rng.next() < 0.3) v += rng.normal() * sigma * (d === 0 ? x1 - x0 : y1 - y0) * 0.25;
        return v;
      });
      kids.push([clampTo(child[0] as number, x0, x1), clampTo(child[1] as number, y0, y1)]);
    }
    pop = kids;
    vals = pop.map(evalAt);
    evaluations += n - 1;
    pop.forEach((p, i) => better(best, p, vals[i] as number));
    yield {
      generation: g,
      points: pop.map((p) => [...p] as [number, number]),
      values: [...vals],
      best: { ...best, x: [...best.x] as [number, number] },
      evaluations,
    };
  }
}

export function* simulatedAnnealing(
  obj: Objective,
  options: HeuristicOptions,
): Generator<Generation, void, undefined> {
  const rng = createRng(options.seed);
  const { x0, x1, y0, y1 } = obj.bounds;
  // one "generation" is `population` proposals of the same walker
  let x: [number, number] = [rng.uniform(x0, x1), rng.uniform(y0, y1)];
  let fx = obj.f(x);
  let evaluations = 1;
  const best = { x: [x[0], x[1]] as [number, number], f: fx };
  let temperature = 2 * Math.max(1, Math.abs(fx));
  const total = options.generations * options.population;
  // geometric cooling from T0 to T0 / 1000 over the whole run
  const cool = 0.001 ** (1 / Math.max(total, 1));
  const span = Math.max(x1 - x0, y1 - y0);
  yield {
    generation: 0,
    points: [[...x]],
    values: [fx],
    best: { ...best, x: [...best.x] as [number, number] },
    evaluations,
    temperature,
  };
  for (let g = 1; g <= options.generations; g++) {
    const trail: [number, number][] = [];
    const trailVals: number[] = [];
    for (let k = 0; k < options.population; k++) {
      const step =
        0.02 * span + 0.2 * span * (temperature / (2 * Math.max(1, Math.abs(best.f) + 1)));
      const cand: [number, number] = [
        clampTo(x[0] + rng.normal() * step, x0, x1),
        clampTo(x[1] + rng.normal() * step, y0, y1),
      ];
      const fc = obj.f(cand);
      evaluations++;
      const delta = fc - fx;
      if (delta < 0 || rng.next() < Math.exp(-delta / temperature)) {
        x = cand;
        fx = fc;
        better(best, x, fx);
      }
      trail.push([...x]);
      trailVals.push(fx);
      temperature *= cool;
    }
    yield {
      generation: g,
      points: trail,
      values: trailVals,
      best: { ...best, x: [...best.x] as [number, number] },
      evaluations,
      temperature,
    };
  }
}

export function* particleSwarm(
  obj: Objective,
  options: HeuristicOptions,
): Generator<Generation, void, undefined> {
  const rng = createRng(options.seed);
  const { x0, x1, y0, y1 } = obj.bounds;
  const n = options.population;
  const w = 0.72;
  const c1 = 1.49;
  const c2 = 1.49;
  const pos: [number, number][] = Array.from({ length: n }, () => [
    rng.uniform(x0, x1),
    rng.uniform(y0, y1),
  ]);
  const vel: [number, number][] = Array.from({ length: n }, () => [
    rng.uniform(-1, 1),
    rng.uniform(-1, 1),
  ]);
  const vals = pos.map((p) => obj.f(p));
  const personal = pos.map((p) => [...p] as [number, number]);
  const personalVals = [...vals];
  let evaluations = n;
  const best = { x: [pos[0]![0], pos[0]![1]] as [number, number], f: vals[0] as number };
  pos.forEach((p, i) => better(best, p, vals[i] as number));
  yield {
    generation: 0,
    points: pos.map((p) => [...p] as [number, number]),
    values: [...vals],
    best: { ...best, x: [...best.x] as [number, number] },
    evaluations,
  };
  const vmax = 0.25 * Math.max(x1 - x0, y1 - y0);
  for (let g = 1; g <= options.generations; g++) {
    for (let i = 0; i < n; i++) {
      for (let d = 0; d < 2; d++) {
        const r1 = rng.next();
        const r2 = rng.next();
        const v =
          w * (vel[i]![d] as number) +
          c1 * r1 * ((personal[i]![d] as number) - (pos[i]![d] as number)) +
          c2 * r2 * ((best.x[d] as number) - (pos[i]![d] as number));
        vel[i]![d] = clampTo(v, -vmax, vmax);
        const lo = d === 0 ? x0 : y0;
        const hi = d === 0 ? x1 : y1;
        pos[i]![d] = clampTo((pos[i]![d] as number) + (vel[i]![d] as number), lo, hi);
      }
      vals[i] = obj.f(pos[i] as [number, number]);
      evaluations++;
      if ((vals[i] as number) < (personalVals[i] as number)) {
        personalVals[i] = vals[i] as number;
        personal[i] = [...(pos[i] as [number, number])] as [number, number];
      }
      better(best, pos[i] as [number, number], vals[i] as number);
    }
    yield {
      generation: g,
      points: pos.map((p) => [...p] as [number, number]),
      values: [...vals],
      best: { ...best, x: [...best.x] as [number, number] },
      evaluations,
    };
  }
}

/** Covariance Matrix Adaptation Evolution Strategy (Hansen), in two dimensions. */
export function* cmaEs(
  obj: Objective,
  options: HeuristicOptions,
): Generator<Generation, void, undefined> {
  const rng = createRng(options.seed);
  const { x0, x1, y0, y1 } = obj.bounds;
  const d = 2;
  const lambda = Math.max(options.population, 6);
  const mu = Math.floor(lambda / 2);
  const raw = Array.from({ length: mu }, (_, i) => Math.log(mu + 0.5) - Math.log(i + 1));
  const sumW = raw.reduce((a, b) => a + b, 0);
  const weights = raw.map((v) => v / sumW);
  const muEff = 1 / weights.reduce((a, b) => a + b * b, 0);
  const cs = (muEff + 2) / (d + muEff + 5);
  const damps = 1 + 2 * Math.max(0, Math.sqrt((muEff - 1) / (d + 1)) - 1) + cs;
  const cc = (4 + muEff / d) / (d + 4 + (2 * muEff) / d);
  const c1 = 2 / ((d + 1.3) ** 2 + muEff);
  const cmu = Math.min(1 - c1, (2 * (muEff - 2 + 1 / muEff)) / ((d + 2) ** 2 + muEff));
  const chiN = Math.sqrt(d) * (1 - 1 / (4 * d) + 1 / (21 * d * d));
  let mean: [number, number] = [rng.uniform(x0, x1), rng.uniform(y0, y1)];
  let sigma = 0.3 * Math.max(x1 - x0, y1 - y0);
  let cov: [[number, number], [number, number]] = [
    [1, 0],
    [0, 1],
  ];
  let pc: [number, number] = [0, 0];
  let ps: [number, number] = [0, 0];
  let evaluations = 0;
  const best = { x: [mean[0], mean[1]] as [number, number], f: Infinity };
  const decompose = (c: [[number, number], [number, number]]) => {
    const e = symmetricEigen(
      Matrix.from([
        [c[0][0], c[0][1]],
        [c[1][0], c[1][1]],
      ]),
    );
    const lam = [Math.max(e.values[0] as number, 1e-20), Math.max(e.values[1] as number, 1e-20)];
    return { b: e.vectors, lam };
  };
  const initialValue = obj.f(mean);
  evaluations++;
  best.f = initialValue;
  yield {
    generation: 0,
    points: [[...mean]],
    values: [initialValue],
    best: { ...best, x: [...best.x] as [number, number] },
    evaluations,
    distribution: {
      mean: [...mean],
      sigma,
      cov: [
        [cov[0][0], cov[0][1]],
        [cov[1][0], cov[1][1]],
      ],
    },
  };
  for (let g = 1; g <= options.generations; g++) {
    const { b, lam } = decompose(cov);
    // sample: x = mean + sigma * B D z
    const zs: [number, number][] = [];
    const ys: [number, number][] = [];
    const xs: [number, number][] = [];
    for (let k = 0; k < lambda; k++) {
      const z: [number, number] = [rng.normal(), rng.normal()];
      const dz: [number, number] = [
        Math.sqrt(lam[0] as number) * z[0],
        Math.sqrt(lam[1] as number) * z[1],
      ];
      const y: [number, number] = [
        b.get(0, 0) * dz[0] + b.get(0, 1) * dz[1],
        b.get(1, 0) * dz[0] + b.get(1, 1) * dz[1],
      ];
      zs.push(z);
      ys.push(y);
      xs.push([mean[0] + sigma * y[0], mean[1] + sigma * y[1]]);
    }
    // the box constrains the evaluated point only (the usual "repair by clipping" for display and scoring)
    const clipped = xs.map(
      (p) => [clampTo(p[0], x0, x1), clampTo(p[1], y0, y1)] as [number, number],
    );
    const fv = clipped.map((p) => obj.f(p));
    evaluations += lambda;
    clipped.forEach((p, i) => better(best, p, fv[i] as number));
    const order = fv.map((_, i) => i).sort((a, b2) => (fv[a] as number) - (fv[b2] as number));
    const yw: [number, number] = [0, 0];
    for (let i = 0; i < mu; i++) {
      const idx = order[i] as number;
      yw[0] += (weights[i] as number) * (ys[idx] as number[])[0]!;
      yw[1] += (weights[i] as number) * (ys[idx] as number[])[1]!;
    }
    mean = [mean[0] + sigma * yw[0], mean[1] + sigma * yw[1]];
    // C^(-1/2) yw = B D^(-1) B^T yw
    const bt = [
      b.get(0, 0) * yw[0] + b.get(1, 0) * yw[1],
      b.get(0, 1) * yw[0] + b.get(1, 1) * yw[1],
    ];
    const invSqrt: [number, number] = [
      b.get(0, 0) * ((bt[0] as number) / Math.sqrt(lam[0] as number)) +
        b.get(0, 1) * ((bt[1] as number) / Math.sqrt(lam[1] as number)),
      b.get(1, 0) * ((bt[0] as number) / Math.sqrt(lam[0] as number)) +
        b.get(1, 1) * ((bt[1] as number) / Math.sqrt(lam[1] as number)),
    ];
    const k1 = Math.sqrt(cs * (2 - cs) * muEff);
    ps = [(1 - cs) * ps[0] + k1 * invSqrt[0], (1 - cs) * ps[1] + k1 * invSqrt[1]];
    const psNorm = Math.hypot(ps[0], ps[1]);
    const hsig = psNorm / Math.sqrt(1 - (1 - cs) ** (2 * g)) / chiN < 1.4 + 2 / (d + 1) ? 1 : 0;
    const k2 = hsig * Math.sqrt(cc * (2 - cc) * muEff);
    pc = [(1 - cc) * pc[0] + k2 * yw[0], (1 - cc) * pc[1] + k2 * yw[1]];
    const next: [[number, number], [number, number]] = [
      [0, 0],
      [0, 0],
    ];
    const decay = 1 - c1 - cmu + (1 - hsig) * c1 * cc * (2 - cc);
    for (let a = 0; a < 2; a++)
      for (let c = 0; c < 2; c++) {
        let rank = 0;
        for (let i = 0; i < mu; i++) {
          const y = ys[order[i] as number] as [number, number];
          rank += (weights[i] as number) * (y[a] as number) * (y[c] as number);
        }
        (next[a] as [number, number])[c] =
          decay * (cov[a] as [number, number])[c]! +
          c1 * (pc[a] as number) * (pc[c] as number) +
          cmu * rank;
      }
    cov = next;
    sigma *= Math.exp((cs / damps) * (psNorm / chiN - 1));
    sigma = Math.min(sigma, 2 * Math.max(x1 - x0, y1 - y0));
    yield {
      generation: g,
      points: clipped,
      values: fv,
      best: { ...best, x: [...best.x] as [number, number] },
      evaluations,
      distribution: {
        mean: [...mean],
        sigma,
        cov: [
          [cov[0][0], cov[0][1]],
          [cov[1][0], cov[1][1]],
        ],
      },
    };
    if (sigma < 1e-14) break;
  }
}

export function runHeuristic(
  kind: Heuristic,
  obj: Objective,
  options: HeuristicOptions,
): Generation[] {
  const gen =
    kind === 'genetic'
      ? geneticAlgorithm(obj, options)
      : kind === 'annealing'
        ? simulatedAnnealing(obj, options)
        : kind === 'swarm'
          ? particleSwarm(obj, options)
          : cmaEs(obj, options);
  return [...gen];
}

export type { Rng };
