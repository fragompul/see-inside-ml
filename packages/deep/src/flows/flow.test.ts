import { Matrix, createRng } from '@sim/core';
import { describe, expect, it } from 'vitest';
import { sampleReal } from '../gan/datasets';
import { DEFAULT_FLOW, DEFAULT_FLOW_TRAINING, Flow, trainFlow } from './flow';

const randomFlow = (seed: number, layers = 4, spread = 0.4) => {
  const flow = new Flow({ ...DEFAULT_FLOW, layers, hidden: 8 }, createRng(seed));
  // give the networks real weights, so that the map is far from the identity
  const rng = createRng(seed + 50);
  for (const p of flow.parameters)
    for (let i = 0; i < p.data.length; i++)
      p.data[i] = (p.data[i] as number) + spread * rng.normal();
  return flow;
};
const points = (n: number, seed: number) => {
  const rng = createRng(seed);
  return new Matrix(
    n,
    2,
    Float64Array.from({ length: 2 * n }, () => rng.normal() * 1.2),
  );
};

describe('flow', () => {
  it('is exactly invertible', () => {
    const flow = randomFlow(2);
    const x = points(30, 1);
    const back = flow.inverse(flow.forward(x, false).z);
    for (let i = 0; i < x.data.length; i++) expect(back.data[i]).toBeCloseTo(x.data[i]!, 10);
  });

  it('the log-determinant equals the log of the numerical Jacobian determinant', () => {
    const flow = randomFlow(3, 3);
    const x = points(6, 4);
    const { logdet } = flow.forward(x, false);
    const h = 1e-6;
    for (let r = 0; r < 6; r++) {
      const column = (axis: number) => {
        const up = new Matrix(1, 2, Float64Array.from(x.data.subarray(2 * r, 2 * r + 2)));
        const down = new Matrix(1, 2, Float64Array.from(x.data.subarray(2 * r, 2 * r + 2)));
        up.data[axis] = up.data[axis]! + h;
        down.data[axis] = down.data[axis]! - h;
        const a = flow.forward(up, false).z;
        const b = flow.forward(down, false).z;
        return [(a.data[0]! - b.data[0]!) / (2 * h), (a.data[1]! - b.data[1]!) / (2 * h)];
      };
      const [c0, c1] = [column(0), column(1)] as [number[], number[]];
      const det = c0[0]! * c1[1]! - c1[0]! * c0[1]!;
      expect(logdet[r]).toBeCloseTo(Math.log(Math.abs(det)), 5);
    }
  });

  it('backpropagation matches finite differences of the loss', () => {
    const flow = randomFlow(5, 3);
    const x = points(5, 6);
    flow.loss(x, true);
    const h = 1e-6;
    for (const p of flow.parameters) {
      const grad = Float64Array.from(p.grad);
      for (const i of [0, Math.floor(p.data.length / 2), p.data.length - 1]) {
        const saved = p.data[i]!;
        p.data[i] = saved + h;
        const up = flow.loss(x, false);
        p.data[i] = saved - h;
        const down = flow.loss(x, false);
        p.data[i] = saved;
        expect(grad[i]).toBeCloseTo((up - down) / (2 * h), 5);
      }
    }
  });

  it('the density integrates to one', () => {
    // a milder random map, so that almost all of the probability lies inside the grid
    const flow = randomFlow(7, 4, 0.12);
    const half = 9;
    const n = 180;
    const cell = (2 * half) / n;
    const grid = new Matrix(n * n, 2);
    for (let i = 0; i < n; i++)
      for (let j = 0; j < n; j++) {
        grid.data[(i * n + j) * 2] = -half + (i + 0.5) * cell;
        grid.data[(i * n + j) * 2 + 1] = -half + (j + 0.5) * cell;
      }
    const total = flow.logProb(grid).reduce((a, lp) => a + Math.exp(lp) * cell * cell, 0);
    expect(total).toBeGreaterThan(0.93);
    expect(total).toBeLessThan(1.02);
  });

  it('learns two moons: far better likelihood than the best single Gaussian', () => {
    const data = new Matrix(600, 2, sampleReal('moons', 600, createRng(3)));
    const run = trainFlow(
      data,
      { layers: 6, hidden: 24, scale: 2 },
      { ...DEFAULT_FLOW_TRAINING, steps: 500 },
    );
    // the best Gaussian (maximum likelihood) has negative log-likelihood log(2 pi e sqrt(det covariance))
    let mx = 0;
    let my = 0;
    for (let i = 0; i < 600; i++) {
      mx += data.data[2 * i]! / 600;
      my += data.data[2 * i + 1]! / 600;
    }
    let sxx = 0;
    let syy = 0;
    let sxy = 0;
    for (let i = 0; i < 600; i++) {
      const dx = data.data[2 * i]! - mx;
      const dy = data.data[2 * i + 1]! - my;
      sxx += (dx * dx) / 600;
      syy += (dy * dy) / 600;
      sxy += (dx * dy) / 600;
    }
    const gaussian = Math.log(2 * Math.PI * Math.E) + 0.5 * Math.log(sxx * syy - sxy * sxy);
    const last = run.history[run.history.length - 1]!.nll;
    expect(last).toBeLessThan(run.history[0]!.nll);
    expect(last).toBeLessThan(gaussian - 0.3);
  });
});
