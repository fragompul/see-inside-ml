import { Optimizer, createRng, Parameter, type Rng } from '@sim/core';

/**
 * A sparsely gated mixture of experts for regression on two inputs. A gate looks at the input and scores every
 * expert; only the `top` best-scoring experts run, and their outputs are combined with the gate probabilities. With
 * more than one expert running the probabilities are renormalised among them; with one (as in the Switch
 * Transformer) the output is simply scaled by the gate probability, which is what lets the task teach the gate.
 * The rest of the experts cost nothing for that input. A balance term in the loss
 * (Shazeer et al., Fedus et al.) pushes the gate to share the inputs among the experts instead of sending nearly all of them to one.
 *     balance = E * sum_e f_e P_e,  f_e = share of inputs routed to e (a count), P_e = mean gate probability of e
 */
export interface MoeConfig {
  experts: number;
  /** Hidden units of every expert. */
  hidden: number;
  /** Experts that run for each input. */
  top: number;
}

export interface MoeTrainingOptions {
  steps: number;
  batch: number;
  learningRate: number;
  /** Weight of the balance term. */
  balance: number;
  seed: number;
}

export const DEFAULT_MOE_TRAINING: MoeTrainingOptions = {
  steps: 1500,
  batch: 64,
  learningRate: 0.01,
  balance: 0.1,
  seed: 1,
};

const randomMatrix = (n: number, std: number, rng: Rng) =>
  Float64Array.from({ length: n }, () => rng.normal() * std);

export class Moe {
  /** Gate: scores = x · wg + bg (2 x E and E). */
  readonly wg: Float64Array;
  readonly bg: Float64Array;
  /** Expert e: scalar output = v_e . tanh(x · w1_e + b1_e) + c_e. */
  readonly w1: Float64Array;
  readonly b1: Float64Array;
  readonly v: Float64Array;
  readonly c: Float64Array;
  private readonly params: Parameter[];

  constructor(
    readonly config: MoeConfig,
    rng: Rng,
  ) {
    const { experts: E, hidden: H } = config;
    this.wg = randomMatrix(2 * E, 0.3, rng);
    this.bg = new Float64Array(E);
    this.w1 = randomMatrix(E * 2 * H, 1.2, rng);
    this.b1 = randomMatrix(E * H, 0.5, rng);
    this.v = randomMatrix(E * H, 0.5, rng);
    this.c = new Float64Array(E);
    this.params = [this.wg, this.bg, this.w1, this.b1, this.v, this.c].map(
      (d) => new Parameter(d, false),
    );
  }

  get parameters(): Parameter[] {
    return this.params;
  }

  get parameterCount(): number {
    return this.params.reduce((a, p) => a + p.data.length, 0);
  }

  /** Parameters that are used for one input: the gate and `top` experts. */
  get activeParameterCount(): number {
    const { experts: E, hidden: H, top } = this.config;
    return 2 * E + E + top * (2 * H + H + H + 1);
  }

  /** Gate probabilities of one input. */
  gate(x1: number, x2: number): Float64Array {
    const { experts: E } = this.config;
    const g = new Float64Array(E);
    let top = -Infinity;
    for (let e = 0; e < E; e++) {
      g[e] = (this.bg[e] as number) + x1 * (this.wg[e] as number) + x2 * (this.wg[E + e] as number);
      top = Math.max(top, g[e] as number);
    }
    let z = 0;
    for (let e = 0; e < E; e++) {
      g[e] = Math.exp((g[e] as number) - top);
      z += g[e] as number;
    }
    return g.map((v) => v / z);
  }

  private expert(e: number, x1: number, x2: number, hidden?: Float64Array): number {
    const { hidden: H } = this.config;
    let out = this.c[e] as number;
    for (let j = 0; j < H; j++) {
      const a = Math.tanh(
        (this.b1[e * H + j] as number) +
          x1 * (this.w1[e * 2 * H + j] as number) +
          x2 * (this.w1[e * 2 * H + H + j] as number),
      );
      if (hidden) hidden[j] = a;
      out += (this.v[e * H + j] as number) * a;
    }
    return out;
  }

  /** The experts chosen for an input, best first. */
  route(x1: number, x2: number): number[] {
    const p = this.gate(x1, x2);
    return Array.from(p.keys())
      .sort((a, b) => (p[b] as number) - (p[a] as number))
      .slice(0, this.config.top);
  }

  predict(x1: number, x2: number): number {
    const p = this.gate(x1, x2);
    const chosen = this.route(x1, x2);
    let z = 0;
    for (const e of chosen) z += p[e] as number;
    if (this.config.top === 1) z = 1;
    let y = 0;
    for (const e of chosen) y += ((p[e] as number) / z) * this.expert(e, x1, x2);
    return y;
  }

  /**
   * Loss on a batch (mean squared error plus `balance` times the balance term) with the gradients filled in when
   * `backprop` is set. Routing is a hard choice and gets no gradient; the gate learns through the probabilities.
   */
  loss(
    xs: ArrayLike<number>,
    ts: ArrayLike<number>,
    balance: number,
    backprop: boolean,
  ): { mse: number; balance: number; total: number; share: number[] } {
    const { experts: E, hidden: H, top } = this.config;
    const n = ts.length;
    if (backprop) for (const p of this.params) p.grad.fill(0);
    const meanP = new Float64Array(E);
    const count = new Float64Array(E);
    const rows: {
      p: Float64Array;
      chosen: number[];
      z: number;
      outs: Float64Array;
      hidden: Float64Array[];
      y: number;
    }[] = [];
    let mse = 0;
    for (let i = 0; i < n; i++) {
      const x1 = xs[2 * i] as number;
      const x2 = xs[2 * i + 1] as number;
      const p = this.gate(x1, x2);
      const chosen = Array.from(p.keys())
        .sort((a, b) => (p[b] as number) - (p[a] as number))
        .slice(0, top);
      let z = 0;
      for (const e of chosen) z += p[e] as number;
      if (top === 1) z = 1;
      const outs = new Float64Array(E);
      const hidden: Float64Array[] = Array.from({ length: E }, () => new Float64Array(H));
      let y = 0;
      for (const e of chosen) {
        outs[e] = this.expert(e, x1, x2, hidden[e]);
        y += ((p[e] as number) / z) * (outs[e] as number);
        count[e] = (count[e] as number) + 1 / (n * top);
      }
      for (let e = 0; e < E; e++) meanP[e] = (meanP[e] as number) + (p[e] as number) / n;
      mse += (y - (ts[i] as number)) ** 2 / n;
      rows.push({ p, chosen, z, outs, hidden, y });
    }
    let bal = 0;
    for (let e = 0; e < E; e++) bal += (count[e] as number) * (meanP[e] as number);
    bal *= E;
    if (backprop) {
      for (let i = 0; i < n; i++) {
        const { p, chosen, z, outs, hidden, y } = rows[i] as (typeof rows)[number];
        const x1 = xs[2 * i] as number;
        const x2 = xs[2 * i + 1] as number;
        const dy = (2 * (y - (ts[i] as number))) / n;
        const dp = new Float64Array(E);
        for (const e of chosen) {
          const weight = (p[e] as number) / z;
          dp[e] = (dy * ((outs[e] as number) - (top === 1 ? 0 : y))) / z;
          const dOut = dy * weight;
          (this.params[5] as Parameter).grad[e] =
            ((this.params[5] as Parameter).grad[e] as number) + dOut;
          for (let j = 0; j < H; j++) {
            const a = (hidden[e] as Float64Array)[j] as number;
            (this.params[4] as Parameter).grad[e * H + j] =
              ((this.params[4] as Parameter).grad[e * H + j] as number) + dOut * a;
            const da = dOut * (this.v[e * H + j] as number) * (1 - a * a);
            (this.params[3] as Parameter).grad[e * H + j] =
              ((this.params[3] as Parameter).grad[e * H + j] as number) + da;
            (this.params[2] as Parameter).grad[e * 2 * H + j] =
              ((this.params[2] as Parameter).grad[e * 2 * H + j] as number) + da * x1;
            (this.params[2] as Parameter).grad[e * 2 * H + H + j] =
              ((this.params[2] as Parameter).grad[e * 2 * H + H + j] as number) + da * x2;
          }
        }
        // the balance term sends E f_e / n to every gate probability
        for (let e = 0; e < E; e++)
          dp[e] = (dp[e] as number) + (balance * E * (count[e] as number)) / n;
        let dot = 0;
        for (let e = 0; e < E; e++) dot += (p[e] as number) * (dp[e] as number);
        for (let e = 0; e < E; e++) {
          const dg = (p[e] as number) * ((dp[e] as number) - dot);
          (this.params[1] as Parameter).grad[e] =
            ((this.params[1] as Parameter).grad[e] as number) + dg;
          (this.params[0] as Parameter).grad[e] =
            ((this.params[0] as Parameter).grad[e] as number) + dg * x1;
          (this.params[0] as Parameter).grad[E + e] =
            ((this.params[0] as Parameter).grad[E + e] as number) + dg * x2;
        }
      }
    }
    return { mse, balance: bal, total: mse + balance * bal, share: Array.from(count) };
  }
}

export interface MoeRun {
  model: Moe;
  /** Test error and the share of test inputs sent to each expert after every 100 steps. */
  history: { step: number; mse: number; share: number[] }[];
}

/** Trains a mixture on samples drawn from `sample` (a function of a random generator that returns a point and its target). */
export function trainMoe(
  config: MoeConfig,
  options: MoeTrainingOptions,
  sample: (rng: Rng) => { x: [number, number]; t: number },
  test: { xs: Float64Array; ts: Float64Array },
): MoeRun {
  const rng = createRng(options.seed);
  const model = new Moe(config, createRng(options.seed + 1));
  const optimizer = new Optimizer('adam');
  const history: MoeRun['history'] = [];
  for (let step = 1; step <= options.steps; step++) {
    const xs = new Float64Array(options.batch * 2);
    const ts = new Float64Array(options.batch);
    for (let i = 0; i < options.batch; i++) {
      const s = sample(rng);
      xs[2 * i] = s.x[0];
      xs[2 * i + 1] = s.x[1];
      ts[i] = s.t;
    }
    model.loss(xs, ts, options.balance, true);
    optimizer.step(model.parameters, options.learningRate);
    if (step % 100 === 0 || step === options.steps) {
      const r = model.loss(test.xs, test.ts, 0, false);
      history.push({ step, mse: r.mse, share: r.share });
    }
  }
  return { model, history };
}
