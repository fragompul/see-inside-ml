import { describe, expect, it } from 'vitest';
import { HEURISTICS, OBJECTIVES, runHeuristic, type Heuristic } from './metaheuristics';

const sphere = OBJECTIVES.sphere!;
const options = { population: 20, generations: 60, seed: 1 };

describe('every heuristic', () => {
  it.each(HEURISTICS)(
    '%s: the best value never gets worse and evaluations only grow',
    (h: Heuristic) => {
      const run = runHeuristic(h, OBJECTIVES.rastrigin!, options);
      for (let i = 1; i < run.length; i++) {
        expect(run[i]!.best.f).toBeLessThanOrEqual(run[i - 1]!.best.f);
        expect(run[i]!.evaluations).toBeGreaterThan(run[i - 1]!.evaluations);
      }
    },
  );

  it.each(HEURISTICS)(
    '%s: is deterministic for a seed and different for another',
    (h: Heuristic) => {
      const a = runHeuristic(h, sphere, options);
      const b = runHeuristic(h, sphere, options);
      const c = runHeuristic(h, sphere, { ...options, seed: 2 });
      expect(a[a.length - 1]!.best).toEqual(b[b.length - 1]!.best);
      expect(a[1]!.points).not.toEqual(c[1]!.points);
    },
  );

  it.each(HEURISTICS)('%s: solves the sphere', (h: Heuristic) => {
    const run = runHeuristic(h, sphere, { population: 20, generations: 150, seed: 3 });
    expect(run[run.length - 1]!.best.f).toBeLessThan(0.05);
  });

  it.each(HEURISTICS)('%s: stays inside the search window', (h: Heuristic) => {
    const run = runHeuristic(h, OBJECTIVES.ackley!, options);
    const { x0, x1, y0, y1 } = OBJECTIVES.ackley!.bounds;
    for (const g of run)
      for (const p of g.points) {
        expect(p[0]).toBeGreaterThanOrEqual(x0 - 1e-9);
        expect(p[0]).toBeLessThanOrEqual(x1 + 1e-9);
        expect(p[1]).toBeGreaterThanOrEqual(y0 - 1e-9);
        expect(p[1]).toBeLessThanOrEqual(y1 + 1e-9);
      }
  });
});

describe('annealing', () => {
  it('cools geometrically from T0 down to a thousandth of it', () => {
    const run = runHeuristic('annealing', sphere, { population: 10, generations: 50, seed: 1 });
    const t0 = run[0]!.temperature!;
    const tEnd = run[run.length - 1]!.temperature!;
    expect(tEnd / t0).toBeCloseTo(0.001, 6);
  });
});

describe('CMA-ES', () => {
  const ellipsoid = OBJECTIVES.ellipsoid!;

  it('learns the shape of a tilted ellipsoid: its covariance gets elongated along the valley', () => {
    const run = runHeuristic('cmaes', ellipsoid, { population: 12, generations: 120, seed: 4 });
    // sample the mid-run covariance (the last generations have collapsed to a point)
    const dist = run[60]!.distribution!;
    const a = dist.cov[0][0];
    const b = dist.cov[0][1];
    const d = dist.cov[1][1];
    const tr = a + d;
    const det = a * d - b * b;
    const disc = Math.sqrt(Math.max((tr * tr) / 4 - det, 0));
    const ratio = (tr / 2 + disc) / Math.max(tr / 2 - disc, 1e-30);
    expect(ratio).toBeGreaterThan(20);
    // the long axis lies along the valley floor: the flat direction of the bowl, at 30 degrees
    const angle = (0.5 * Math.atan2(2 * b, a - d) * 180) / Math.PI;
    expect(Math.abs(angle - 30)).toBeLessThan(15);
  });

  it('finds the ellipsoid minimum to high precision in few evaluations where the other methods are far off', () => {
    const budget = { population: 12, generations: 100, seed: 5 };
    const cma = runHeuristic('cmaes', ellipsoid, budget);
    const cmaBest = cma[cma.length - 1]!.best.f;
    expect(cmaBest).toBeLessThan(1e-6);
    for (const h of ['genetic', 'swarm', 'annealing'] as Heuristic[]) {
      const run = runHeuristic(h, ellipsoid, { population: 12, generations: 100, seed: 5 });
      expect(run[run.length - 1]!.best.f).toBeGreaterThan(cmaBest);
    }
  });
});
