import { Optimizer, Parameter, createRng, type Rng } from '@sim/core';

/**
 * A neural ODE classifier. The state of a point, a vector of `dim` numbers whose first two are the input, flows for
 * unit time under a velocity given by a small network, dx/dt = f(x) = W2 tanh(W1 x + b1) + b2, integrated with a fixed
 * number of Euler or Runge-Kutta steps; a linear read-out of the final state gives the class. Gradients are obtained by
 * differentiating through the solver steps. In two dimensions the flow cannot make paths cross, so it cannot
 * separate a point inside a ring from the ring around it; extra dimensions, filled with zeros at the start, give it room.
 */
export type Solver = 'euler' | 'rk4';

export interface NodeConfig {
  /** Dimensions of the state (at least 2). */
  dim: number;
  hidden: number;
  solver: Solver;
  /** Solver steps for unit time. */
  steps: number;
}

export class NeuralOde {
  readonly w1: Float64Array;
  readonly b1: Float64Array;
  readonly w2: Float64Array;
  readonly b2: Float64Array;
  readonly wo: Float64Array;
  readonly bo: Float64Array;
  private readonly params: Parameter[];

  constructor(
    readonly config: NodeConfig,
    rng: Rng,
  ) {
    const { dim, hidden } = config;
    this.w1 = Float64Array.from(
      { length: dim * hidden },
      () => (rng.normal() * 0.8) / Math.sqrt(dim),
    );
    this.b1 = new Float64Array(hidden);
    this.w2 = Float64Array.from(
      { length: hidden * dim },
      () => (rng.normal() * 0.3) / Math.sqrt(hidden),
    );
    this.b2 = new Float64Array(dim);
    this.wo = Float64Array.from({ length: dim }, () => rng.normal() * 0.3);
    this.bo = new Float64Array(1);
    // weights (not biases) may be penalised, which keeps the flow gentle
    this.params = [this.w1, this.b1, this.w2, this.b2, this.wo, this.bo].map(
      (d, i) => new Parameter(d, i === 0 || i === 2 || i === 4),
    );
  }

  get parameters(): Parameter[] {
    return this.params;
  }

  /** The velocity field at x. */
  velocity(x: ArrayLike<number>): Float64Array {
    const { dim, hidden } = this.config;
    const out = Float64Array.from(this.b2);
    for (let j = 0; j < hidden; j++) {
      let s = this.b1[j] as number;
      for (let i = 0; i < dim; i++) s += (x[i] as number) * (this.w1[i * hidden + j] as number);
      const a = Math.tanh(s);
      for (let k = 0; k < dim; k++)
        out[k] = (out[k] as number) + a * (this.w2[j * dim + k] as number);
    }
    return out;
  }

  /** Vector-Jacobian product of the velocity at x: returns J^T g and adds the parameter gradients for output gradient g. */
  private vjp(x: ArrayLike<number>, g: ArrayLike<number>, accumulate: boolean): Float64Array {
    const { dim, hidden } = this.config;
    const gx = new Float64Array(dim);
    for (let j = 0; j < hidden; j++) {
      let s = this.b1[j] as number;
      for (let i = 0; i < dim; i++) s += (x[i] as number) * (this.w1[i * hidden + j] as number);
      const a = Math.tanh(s);
      let ga = 0;
      for (let k = 0; k < dim; k++) {
        ga += (g[k] as number) * (this.w2[j * dim + k] as number);
        if (accumulate) {
          const p = this.params[2] as Parameter;
          p.grad[j * dim + k] = (p.grad[j * dim + k] as number) + (g[k] as number) * a;
        }
      }
      const gs = ga * (1 - a * a);
      if (accumulate) {
        const pb = this.params[1] as Parameter;
        pb.grad[j] = (pb.grad[j] as number) + gs;
      }
      for (let i = 0; i < dim; i++) {
        gx[i] = (gx[i] as number) + gs * (this.w1[i * hidden + j] as number);
        if (accumulate) {
          const p = this.params[0] as Parameter;
          p.grad[i * hidden + j] = (p.grad[i * hidden + j] as number) + gs * (x[i] as number);
        }
      }
    }
    if (accumulate) {
      const pb2 = this.params[3] as Parameter;
      for (let k = 0; k < dim; k++) pb2.grad[k] = (pb2.grad[k] as number) + (g[k] as number);
    }
    return gx;
  }

  private step(x: Float64Array, h: number): { next: Float64Array; stages: Float64Array[] } {
    const dim = this.config.dim;
    if (this.config.solver === 'euler') {
      const k1 = this.velocity(x);
      return { next: x.map((v, i) => v + h * (k1[i] as number)), stages: [x] };
    }
    const a2 = new Float64Array(dim);
    const a3 = new Float64Array(dim);
    const a4 = new Float64Array(dim);
    const k1 = this.velocity(x);
    for (let i = 0; i < dim; i++) a2[i] = (x[i] as number) + 0.5 * h * (k1[i] as number);
    const k2 = this.velocity(a2);
    for (let i = 0; i < dim; i++) a3[i] = (x[i] as number) + 0.5 * h * (k2[i] as number);
    const k3 = this.velocity(a3);
    for (let i = 0; i < dim; i++) a4[i] = (x[i] as number) + h * (k3[i] as number);
    const k4 = this.velocity(a4);
    const next = new Float64Array(dim);
    for (let i = 0; i < dim; i++)
      next[i] =
        (x[i] as number) +
        (h / 6) *
          ((k1[i] as number) + 2 * (k2[i] as number) + 2 * (k3[i] as number) + (k4[i] as number));
    return { next, stages: [x, a2, a3, a4] };
  }

  /** The start state of a point: its two coordinates followed by zeros. */
  start(x1: number, x2: number): Float64Array {
    const s = new Float64Array(this.config.dim);
    s[0] = x1;
    s[1] = x2;
    return s;
  }

  /** The whole path of a point, from t = 0 to t = 1 (steps + 1 states). */
  trajectory(x1: number, x2: number): Float64Array[] {
    const h = 1 / this.config.steps;
    const path = [this.start(x1, x2)];
    for (let n = 0; n < this.config.steps; n++)
      path.push(this.step(path[n] as Float64Array, h).next);
    return path;
  }

  /** Probability that the point belongs to class 1. */
  probability(x1: number, x2: number): number {
    const z = this.trajectory(x1, x2).pop() as Float64Array;
    let s = this.bo[0] as number;
    for (let i = 0; i < this.config.dim; i++) s += (z[i] as number) * (this.wo[i] as number);
    return 1 / (1 + Math.exp(-s));
  }

  /** Mean cross-entropy over the points, with parameter gradients filled in when `backprop` is set. */
  loss(xs: ArrayLike<number>, ys: ArrayLike<number>, backprop: boolean): number {
    const { dim, steps } = this.config;
    const n = ys.length;
    const h = 1 / steps;
    if (backprop) for (const p of this.params) p.grad.fill(0);
    let total = 0;
    for (let r = 0; r < n; r++) {
      const states: Float64Array[] = [this.start(xs[2 * r] as number, xs[2 * r + 1] as number)];
      const stages: Float64Array[][] = [];
      for (let k = 0; k < steps; k++) {
        const out = this.step(states[k] as Float64Array, h);
        states.push(out.next);
        stages.push(out.stages);
      }
      const z = states[steps] as Float64Array;
      let s = this.bo[0] as number;
      for (let i = 0; i < dim; i++) s += (z[i] as number) * (this.wo[i] as number);
      const p = 1 / (1 + Math.exp(-s));
      const y = ys[r] as number;
      total -= (y * Math.log(Math.max(p, 1e-12)) + (1 - y) * Math.log(Math.max(1 - p, 1e-12))) / n;
      if (!backprop) continue;
      const ds = (p - y) / n;
      const pb = this.params[5] as Parameter;
      pb.grad[0] = (pb.grad[0] as number) + ds;
      const pw = this.params[4] as Parameter;
      let g: Float64Array = new Float64Array(dim);
      for (let i = 0; i < dim; i++) {
        pw.grad[i] = (pw.grad[i] as number) + ds * (z[i] as number);
        g[i] = ds * (this.wo[i] as number);
      }
      for (let k = steps - 1; k >= 0; k--) {
        const st = stages[k] as Float64Array[];
        if (this.config.solver === 'euler') {
          const gx = this.vjp(
            st[0] as Float64Array,
            g.map((v) => v * h),
            true,
          );
          g = g.map((v, i) => v + (gx[i] as number));
        } else {
          const [x, a2, a3, a4] = st as [Float64Array, Float64Array, Float64Array, Float64Array];
          const gk1 = g.map((v) => (v * h) / 6);
          const gk2 = g.map((v) => (v * h) / 3);
          const gk3 = g.map((v) => (v * h) / 3);
          const gk4 = g.map((v) => (v * h) / 6);
          const next = Float64Array.from(g);
          const ga4 = this.vjp(a4, gk4, true);
          for (let i = 0; i < dim; i++) {
            next[i] = (next[i] as number) + (ga4[i] as number);
            gk3[i] = (gk3[i] as number) + h * (ga4[i] as number);
          }
          const ga3 = this.vjp(a3, gk3, true);
          for (let i = 0; i < dim; i++) {
            next[i] = (next[i] as number) + (ga3[i] as number);
            gk2[i] = (gk2[i] as number) + 0.5 * h * (ga3[i] as number);
          }
          const ga2 = this.vjp(a2, gk2, true);
          for (let i = 0; i < dim; i++) {
            next[i] = (next[i] as number) + (ga2[i] as number);
            gk1[i] = (gk1[i] as number) + 0.5 * h * (ga2[i] as number);
          }
          const ga1 = this.vjp(x, gk1, true);
          for (let i = 0; i < dim; i++) next[i] = (next[i] as number) + (ga1[i] as number);
          g = next;
        }
      }
    }
    return total;
  }

  accuracy(xs: ArrayLike<number>, ys: ArrayLike<number>): number {
    let right = 0;
    for (let r = 0; r < ys.length; r++)
      if ((this.probability(xs[2 * r] as number, xs[2 * r + 1] as number) > 0.5 ? 1 : 0) === ys[r])
        right++;
    return right / ys.length;
  }
}

export interface NodeRun {
  model: NeuralOde;
  history: { epoch: number; loss: number; accuracy: number }[];
}

export function trainNeuralOde(
  config: NodeConfig,
  data: { xs: Float64Array; ys: Float64Array },
  options: { epochs: number; learningRate: number; seed: number; decay?: number },
  every = 10,
): NodeRun {
  const model = new NeuralOde(config, createRng(options.seed));
  const optimizer = new Optimizer('adam');
  const history: NodeRun['history'] = [];
  for (let epoch = 1; epoch <= options.epochs; epoch++) {
    model.loss(data.xs, data.ys, true);
    optimizer.step(model.parameters, options.learningRate, { l1: 0, l2: options.decay ?? 0 });
    if (epoch % every === 0 || epoch === options.epochs)
      history.push({
        epoch,
        loss: model.loss(data.xs, data.ys, false),
        accuracy: model.accuracy(data.xs, data.ys),
      });
  }
  return { model, history };
}
