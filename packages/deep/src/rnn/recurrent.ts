import { Parameter, type Rng } from '@sim/core';

/**
 * A single-layer recurrent classifier with a hand-written backpropagation through time (BPTT).
 * Tokens are integers (one-hot inputs), so "input × weights" is just a row lookup. After the last
 * token the final hidden state goes through a softmax layer: many-to-one classification.
 *
 * Gate layouts (blocks of `hidden` columns):
 *   rnn  h = tanh(x·Wx + h·Wh + b)
 *   lstm [i, f, g, o]   c = f·c + i·g,  h = o·tanh(c)
 *   gru  [r, z, n]      n = tanh(x·Wxn + r·(h·Whn) + bn),  h = (1 − z)·n + z·h
 */
export const CELL_NAMES = ['rnn', 'lstm', 'gru'] as const;
export type CellName = (typeof CELL_NAMES)[number];

export const cellInfo: Record<CellName, { label: string; note: string; gates: number }> = {
  rnn: {
    label: 'Vanilla RNN',
    note: 'One tanh layer rewrites the whole memory at every step.',
    gates: 1,
  },
  lstm: {
    label: 'LSTM',
    note: 'A separate cell state, edited through forget, input and output gates.',
    gates: 4,
  },
  gru: {
    label: 'GRU',
    note: 'Two gates (reset, update) blend the old memory with a new candidate.',
    gates: 3,
  },
};

export interface RecurrentConfig {
  cell: CellName;
  hidden: number;
  vocab: number;
  classes: number;
}

export const RECURRENT_LIMITS = { hidden: [2, 48] } as const;

export function validateRecurrent(c: RecurrentConfig): void {
  if (!(CELL_NAMES as readonly string[]).includes(c.cell)) throw new RangeError('Unknown cell');
  const [lo, hi] = RECURRENT_LIMITS.hidden;
  if (!Number.isInteger(c.hidden) || c.hidden < lo || c.hidden > hi)
    throw new RangeError(`Hidden size must be an integer in [${lo}, ${hi}]`);
  if (!Number.isInteger(c.vocab) || c.vocab < 2) throw new RangeError('Vocabulary must be ≥ 2');
  if (!Number.isInteger(c.classes) || c.classes < 2) throw new RangeError('Need ≥ 2 classes');
}

/** Everything one sequence produced on its way through the network. */
export interface RecurrentTrace {
  tokens: Int32Array;
  steps: number;
  hidden: number;
  /** Hidden states, `(steps + 1) × hidden`; row 0 is the all-zero start. */
  h: Float64Array;
  /** Cell states (LSTM only), `(steps + 1) × hidden`. */
  c: Float64Array | null;
  /**
   * Gate activations after their nonlinearity, `steps × hidden` each, named by the cell:
   * rnn: none · lstm: input, forget, candidate, output · gru: reset, update, candidate.
   */
  gates: Record<string, Float64Array>;
  logits: number[];
  probs: number[];
}

const sigmoid = (x: number) => 1 / (1 + Math.exp(-x));
const at = (a: Float64Array, i: number) => a[i] as number;

export class RecurrentNet {
  readonly cell: CellName;
  readonly hidden: number;
  readonly vocab: number;
  readonly classes: number;
  /** Number of gate blocks per row of `wx`/`wh`. */
  readonly blocks: number;
  /** Input weights `(vocab, blocks·hidden)`, hidden weights `(hidden, blocks·hidden)`. */
  readonly wx: Parameter;
  readonly wh: Parameter;
  readonly b: Parameter;
  readonly wo: Parameter;
  readonly bo: Parameter;

  constructor(
    readonly config: RecurrentConfig,
    rng: Rng,
  ) {
    validateRecurrent(config);
    this.cell = config.cell;
    this.hidden = config.hidden;
    this.vocab = config.vocab;
    this.classes = config.classes;
    this.blocks = cellInfo[config.cell].gates;
    const H = this.hidden;
    const G = this.blocks * H;
    const limit = 1 / Math.sqrt(H); // the standard recurrent initialisation
    const fill = (n: number, bound: number) => {
      const a = new Float64Array(n);
      for (let i = 0; i < n; i++) a[i] = rng.uniform(-bound, bound);
      return a;
    };
    this.wx = new Parameter(fill(this.vocab * G, limit), true);
    this.wh = new Parameter(fill(H * G, limit), true);
    const bias = new Float64Array(G);
    if (this.cell === 'lstm') bias.fill(1, H, 2 * H); // forget gate starts open: remember by default
    this.b = new Parameter(bias, false);
    const outLimit = Math.sqrt(6 / (H + this.classes));
    this.wo = new Parameter(fill(H * this.classes, outLimit), true);
    this.bo = new Parameter(new Float64Array(this.classes), false);
  }

  get parameters(): Parameter[] {
    return [this.wx, this.wh, this.b, this.wo, this.bo];
  }

  get parameterCount(): number {
    return this.parameters.reduce((sum, p) => sum + p.data.length, 0);
  }

  zeroGrad(): void {
    for (const p of this.parameters) p.grad.fill(0);
  }

  /** Runs one sequence and keeps every state and gate. */
  run(tokens: ArrayLike<number>): RecurrentTrace {
    const T = tokens.length;
    const H = this.hidden;
    const G = this.blocks * H;
    const { wx, wh, b, wo, bo } = this;
    const h = new Float64Array((T + 1) * H);
    const c = this.cell === 'lstm' ? new Float64Array((T + 1) * H) : null;
    const gates: Record<string, Float64Array> = {};
    const names: Record<CellName, string[]> = {
      rnn: [],
      lstm: ['input', 'forget', 'candidate', 'output'],
      gru: ['reset', 'update', 'candidate', 'hiddenPart'],
    };
    for (const name of names[this.cell]) gates[name] = new Float64Array(T * H);
    const pre = new Float64Array(G);
    const hh = new Float64Array(G); // h·Wh, kept separate for the GRU candidate

    for (let t = 0; t < T; t++) {
      const token = tokens[t] as number;
      if (!Number.isInteger(token) || token < 0 || token >= this.vocab)
        throw new RangeError(`Token ${token} is outside the vocabulary`);
      const prev = t * H;
      const now = (t + 1) * H;
      // pre = Wx[token] + b ;  hh = h_prev · Wh
      for (let k = 0; k < G; k++) {
        pre[k] = at(wx.data, token * G + k) + at(b.data, k);
        hh[k] = 0;
      }
      for (let j = 0; j < H; j++) {
        const hj = at(h, prev + j);
        if (hj === 0) continue;
        for (let k = 0; k < G; k++) hh[k] = (hh[k] as number) + hj * at(wh.data, j * G + k);
      }

      if (this.cell === 'rnn') {
        for (let k = 0; k < H; k++) h[now + k] = Math.tanh((pre[k] as number) + (hh[k] as number));
      } else if (this.cell === 'lstm') {
        for (let k = 0; k < H; k++) {
          const i = sigmoid((pre[k] as number) + (hh[k] as number));
          const f = sigmoid((pre[H + k] as number) + (hh[H + k] as number));
          const g = Math.tanh((pre[2 * H + k] as number) + (hh[2 * H + k] as number));
          const o = sigmoid((pre[3 * H + k] as number) + (hh[3 * H + k] as number));
          const cell = f * at(c as Float64Array, prev + k) + i * g;
          (c as Float64Array)[now + k] = cell;
          h[now + k] = o * Math.tanh(cell);
          (gates.input as Float64Array)[t * H + k] = i;
          (gates.forget as Float64Array)[t * H + k] = f;
          (gates.candidate as Float64Array)[t * H + k] = g;
          (gates.output as Float64Array)[t * H + k] = o;
        }
      } else {
        for (let k = 0; k < H; k++) {
          const r = sigmoid((pre[k] as number) + (hh[k] as number));
          const z = sigmoid((pre[H + k] as number) + (hh[H + k] as number));
          const hn = hh[2 * H + k] as number;
          const n = Math.tanh((pre[2 * H + k] as number) + r * hn);
          h[now + k] = (1 - z) * n + z * at(h, prev + k);
          (gates.reset as Float64Array)[t * H + k] = r;
          (gates.update as Float64Array)[t * H + k] = z;
          (gates.candidate as Float64Array)[t * H + k] = n;
          (gates.hiddenPart as Float64Array)[t * H + k] = hn;
        }
      }
    }

    const logits: number[] = [];
    for (let k = 0; k < this.classes; k++) {
      let sum = at(bo.data, k);
      for (let j = 0; j < H; j++) sum += at(h, T * H + j) * at(wo.data, j * this.classes + k);
      logits.push(sum);
    }
    const max = Math.max(...logits);
    const exps = logits.map((v) => Math.exp(v - max));
    const total = exps.reduce((a, v) => a + v, 0);
    const probs = exps.map((v) => v / total);
    return {
      tokens: Int32Array.from(tokens as ArrayLike<number>),
      steps: T,
      hidden: H,
      h,
      c,
      gates,
      logits,
      probs,
    };
  }

  /**
   * Backpropagation through time for one sequence. Adds `scale · dL/dθ` to every `grad` buffer and
   * returns ‖dL/dh_t‖ for t = 1..T (index 0 is the gradient reaching the first step's output):
   * how strongly the loss still "feels" each step. Cross-entropy loss is returned too.
   */
  backward(
    trace: RecurrentTrace,
    target: number,
    scale = 1,
  ): { loss: number; gradientNorms: Float64Array } {
    const { steps: T, h, c, gates, probs, tokens } = trace;
    const H = this.hidden;
    const G = this.blocks * H;
    const C = this.classes;
    const { wx, wh, b, wo, bo } = this;
    const loss = -Math.log(Math.max(probs[target] as number, 1e-300));

    // dL/dlogits, then into the output layer and the last hidden state
    if (!(scale > 0)) throw new RangeError('scale must be positive');
    let dh = new Float64Array(H);
    for (let k = 0; k < C; k++) {
      const dl = ((probs[k] as number) - (k === target ? 1 : 0)) * scale;
      bo.grad[k] = (bo.grad[k] as number) + dl;
      for (let j = 0; j < H; j++) {
        wo.grad[j * C + k] = (wo.grad[j * C + k] as number) + at(h, T * H + j) * dl;
        dh[j] = (dh[j] as number) + at(wo.data, j * C + k) * dl;
      }
    }
    let dc = new Float64Array(H);
    const norms = new Float64Array(T);
    const dpre = new Float64Array(G);

    for (let t = T; t >= 1; t--) {
      let sq = 0;
      for (let j = 0; j < H; j++) sq += (dh[j] as number) ** 2;
      // the stored norm is divided by `scale` so it does not depend on the batch size
      norms[t - 1] = Math.sqrt(sq) / scale;

      const prev = (t - 1) * H;
      const now = t * H;
      const token = tokens[t - 1] as number;
      const gi = (t - 1) * H;
      const dhPrev = new Float64Array(H);
      const dcPrev = new Float64Array(H);

      if (this.cell === 'rnn') {
        for (let k = 0; k < H; k++) {
          const hk = at(h, now + k);
          dpre[k] = (dh[k] as number) * (1 - hk * hk);
        }
      } else if (this.cell === 'lstm') {
        const cs = c as Float64Array;
        for (let k = 0; k < H; k++) {
          const i = at(gates.input as Float64Array, gi + k);
          const f = at(gates.forget as Float64Array, gi + k);
          const g = at(gates.candidate as Float64Array, gi + k);
          const o = at(gates.output as Float64Array, gi + k);
          const tc = Math.tanh(at(cs, now + k));
          const dhk = dh[k] as number;
          const dck = (dc[k] as number) + dhk * o * (1 - tc * tc);
          dpre[k] = dck * g * i * (1 - i);
          dpre[H + k] = dck * at(cs, prev + k) * f * (1 - f);
          dpre[2 * H + k] = dck * i * (1 - g * g);
          dpre[3 * H + k] = dhk * tc * o * (1 - o);
          dcPrev[k] = dck * f;
        }
      } else {
        for (let k = 0; k < H; k++) {
          const r = at(gates.reset as Float64Array, gi + k);
          const z = at(gates.update as Float64Array, gi + k);
          const n = at(gates.candidate as Float64Array, gi + k);
          const hn = at(gates.hiddenPart as Float64Array, gi + k);
          const hp = at(h, prev + k);
          const dhk = dh[k] as number;
          const da = dhk * (1 - z) * (1 - n * n);
          dpre[k] = da * hn * r * (1 - r);
          dpre[H + k] = dhk * (hp - n) * z * (1 - z);
          dpre[2 * H + k] = da;
          dhPrev[k] = dhk * z;
        }
      }

      // parameter gradients
      for (let k = 0; k < G; k++) {
        const d = dpre[k] as number;
        wx.grad[token * G + k] = (wx.grad[token * G + k] as number) + d;
        b.grad[k] = (b.grad[k] as number) + d;
      }
      for (let j = 0; j < H; j++) {
        const hj = at(h, prev + j);
        let back = 0;
        for (let k = 0; k < G; k++) {
          // the GRU candidate block sees h through the reset gate: its h-side gradient is da · r
          let d = dpre[k] as number;
          if (this.cell === 'gru' && k >= 2 * H)
            d *= at(gates.reset as Float64Array, gi + (k - 2 * H));
          wh.grad[j * G + k] = (wh.grad[j * G + k] as number) + hj * d;
          back += at(wh.data, j * G + k) * d;
        }
        dhPrev[j] = (dhPrev[j] as number) + back;
      }
      dh = dhPrev;
      dc = dcPrev;
    }
    return { loss, gradientNorms: norms };
  }

  /** Class probabilities only. */
  predict(tokens: ArrayLike<number>): number[] {
    return this.run(tokens).probs;
  }
}
