import { Parameter, createRng, type Rng } from '@sim/core';
import { cellInfo, type CellName } from './recurrent';

/**
 * A character-level language model: a recurrent network reads text one character at a time and, at
 * every step, predicts the next character. Hand-written backpropagation through time (the same cells
 * as `RecurrentNet`, but with a prediction at every step instead of one at the end) and Adam.
 */
export interface CharLmConfig {
  cell: CellName;
  hidden: number;
  vocab: number;
}

export interface CharTrace {
  tokens: Int32Array;
  steps: number;
  h: Float64Array;
  c: Float64Array | null;
  /** Gate activations per step; names depend on the cell. */
  gates: Record<string, Float64Array>;
  probs: Float64Array;
}

const sigmoid = (x: number) => 1 / (1 + Math.exp(-x));
const at = (a: Float64Array, i: number) => a[i] as number;

export class CharLM {
  readonly cell: CellName;
  readonly hidden: number;
  readonly vocab: number;
  readonly blocks: number;
  readonly wx: Parameter;
  readonly wh: Parameter;
  readonly b: Parameter;
  readonly wo: Parameter;
  readonly bo: Parameter;
  private m: Float64Array[] = [];
  private v: Float64Array[] = [];
  private t = 0;

  constructor(
    readonly config: CharLmConfig,
    rng: Rng,
  ) {
    this.cell = config.cell;
    this.hidden = config.hidden;
    this.vocab = config.vocab;
    this.blocks = cellInfo[config.cell].gates;
    const H = this.hidden;
    const G = this.blocks * H;
    const limit = 1 / Math.sqrt(H);
    const fill = (n: number, bound: number) => {
      const a = new Float64Array(n);
      for (let i = 0; i < n; i++) a[i] = rng.uniform(-bound, bound);
      return a;
    };
    this.wx = new Parameter(fill(this.vocab * G, limit), true);
    this.wh = new Parameter(fill(H * G, limit), true);
    const bias = new Float64Array(G);
    if (this.cell === 'lstm') bias.fill(1, H, 2 * H);
    this.b = new Parameter(bias, false);
    this.wo = new Parameter(fill(H * this.vocab, Math.sqrt(6 / (H + this.vocab))), true);
    this.bo = new Parameter(new Float64Array(this.vocab), false);
  }

  get parameters(): Parameter[] {
    return [this.wx, this.wh, this.b, this.wo, this.bo];
  }

  get parameterCount(): number {
    return this.parameters.reduce((s, p) => s + p.data.length, 0);
  }

  zeroGrad(): void {
    for (const p of this.parameters) p.grad.fill(0);
  }

  /** Reads `tokens` and predicts, after each one, the next. Returns the mean cross-entropy (nats per character). */
  forward(tokens: ArrayLike<number>): { loss: number; trace: CharTrace } {
    const T = tokens.length - 1;
    if (T < 1) throw new RangeError('Need at least two characters');
    const H = this.hidden;
    const G = this.blocks * H;
    const V = this.vocab;
    const { wx, wh, b, wo, bo } = this;
    const h = new Float64Array((T + 1) * H);
    const c = this.cell === 'lstm' ? new Float64Array((T + 1) * H) : null;
    const gateNames: Record<CellName, string[]> = {
      rnn: [],
      lstm: ['input', 'forget', 'candidate', 'output'],
      gru: ['reset', 'update', 'candidate', 'hiddenPart'],
    };
    const gates: Record<string, Float64Array> = {};
    for (const n of gateNames[this.cell]) gates[n] = new Float64Array(T * H);
    const probs = new Float64Array(T * V);
    const pre = new Float64Array(G);
    const hh = new Float64Array(G);
    let loss = 0;
    for (let t = 0; t < T; t++) {
      const token = tokens[t] as number;
      const prev = t * H;
      const now = (t + 1) * H;
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
      // next-character distribution
      let max = -Infinity;
      for (let k = 0; k < V; k++) {
        let s = at(bo.data, k);
        for (let j = 0; j < H; j++) s += at(h, now + j) * at(wo.data, j * V + k);
        probs[t * V + k] = s;
        if (s > max) max = s;
      }
      let z = 0;
      for (let k = 0; k < V; k++) {
        const e = Math.exp((probs[t * V + k] as number) - max);
        probs[t * V + k] = e;
        z += e;
      }
      for (let k = 0; k < V; k++) probs[t * V + k] = (probs[t * V + k] as number) / z;
      loss -= Math.log(Math.max(probs[t * V + (tokens[t + 1] as number)] as number, 1e-300));
    }
    return {
      loss: loss / T,
      trace: { tokens: Int32Array.from(tokens as ArrayLike<number>), steps: T, h, c, gates, probs },
    };
  }

  /** Backpropagation through time for the mean loss of one window; adds `scale` times the gradient to every `grad`. */
  backward(trace: CharTrace, scale = 1): void {
    const { steps: T, h, c, gates, probs, tokens } = trace;
    const H = this.hidden;
    const G = this.blocks * H;
    const V = this.vocab;
    const { wx, wh, b, wo, bo } = this;
    const unit = scale / T;
    let dh = new Float64Array(H);
    let dc = new Float64Array(H);
    const dpre = new Float64Array(G);
    for (let t = T; t >= 1; t--) {
      // the prediction made after reading character t-1 sees hidden state t
      const target = tokens[t] as number;
      for (let k = 0; k < V; k++) {
        const dl = ((probs[(t - 1) * V + k] as number) - (k === target ? 1 : 0)) * unit;
        bo.grad[k] = (bo.grad[k] as number) + dl;
        for (let j = 0; j < H; j++) {
          wo.grad[j * V + k] = (wo.grad[j * V + k] as number) + at(h, t * H + j) * dl;
          dh[j] = (dh[j] as number) + at(wo.data, j * V + k) * dl;
        }
      }
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
      for (let k = 0; k < G; k++) {
        const d = dpre[k] as number;
        wx.grad[token * G + k] = (wx.grad[token * G + k] as number) + d;
        b.grad[k] = (b.grad[k] as number) + d;
      }
      for (let j = 0; j < H; j++) {
        const hj = at(h, prev + j);
        let back = 0;
        for (let k = 0; k < G; k++) {
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
  }

  /** One Adam step on the accumulated gradients, with the global norm clipped to `clip`. Returns the norm before clipping. */
  adamStep(learningRate: number, clip = 5): number {
    if (this.m.length === 0) {
      this.m = this.parameters.map((p) => new Float64Array(p.data.length));
      this.v = this.parameters.map((p) => new Float64Array(p.data.length));
    }
    let sq = 0;
    for (const p of this.parameters) for (const g of p.grad) sq += g * g;
    const norm = Math.sqrt(sq);
    const factor = norm > clip ? clip / norm : 1;
    this.t += 1;
    const b1 = 0.9;
    const b2 = 0.999;
    this.parameters.forEach((p, idx) => {
      const m = this.m[idx] as Float64Array;
      const v = this.v[idx] as Float64Array;
      for (let i = 0; i < p.data.length; i++) {
        const g = (p.grad[i] as number) * factor;
        m[i] = b1 * (m[i] as number) + (1 - b1) * g;
        v[i] = b2 * (v[i] as number) + (1 - b2) * g * g;
        const mh = (m[i] as number) / (1 - b1 ** this.t);
        const vh = (v[i] as number) / (1 - b2 ** this.t);
        p.data[i] = (p.data[i] as number) - (learningRate * mh) / (Math.sqrt(vh) + 1e-8);
      }
    });
    return norm;
  }

  /** Mean loss (nats per character) of a window, without touching gradients. */
  evaluate(tokens: ArrayLike<number>): number {
    return this.forward(tokens).loss;
  }

  /** Continues `prefix` for `length` characters, drawing from the predicted distribution at the given temperature. */
  sample(prefix: ArrayLike<number>, length: number, temperature: number, rng: Rng): number[] {
    const out = Array.from(prefix);
    const V = this.vocab;
    for (let n = 0; n < length; n++) {
      // run the whole history each time: windows are short, so this stays cheap
      const window = out.slice(-48);
      const padded = window.length < 2 ? [window[0] ?? 0, window[0] ?? 0] : window;
      const { trace } = this.forward([...padded, 0]);
      const last = trace.steps - 1;
      const p = Array.from({ length: V }, (_, k) => trace.probs[last * V + k] as number);
      // temperature: rescale log-probabilities
      const logits = p.map((v) => Math.log(Math.max(v, 1e-12)) / Math.max(temperature, 1e-3));
      const m = Math.max(...logits);
      const e = logits.map((v) => Math.exp(v - m));
      const z = e.reduce((a, b) => a + b, 0);
      let u = rng.next() * z;
      let pick = V - 1;
      for (let k = 0; k < V; k++) {
        u -= e[k] as number;
        if (u <= 0) {
          pick = k;
          break;
        }
      }
      out.push(pick);
    }
    return out;
  }
}

// ---- text helpers ------------------------------------------------------------------------------

export interface Corpus {
  text: string;
  alphabet: string[];
  tokens: Int32Array;
}

export function makeCorpus(text: string): Corpus {
  const alphabet = [...new Set(text)].sort();
  const index = new Map(alphabet.map((c, i) => [c, i]));
  return { text, alphabet, tokens: Int32Array.from([...text].map((c) => index.get(c) as number)) };
}

export const decode = (corpus: Corpus, tokens: ArrayLike<number>): string =>
  Array.from(tokens, (t) => corpus.alphabet[t] ?? '?').join('');

export interface CharTrainOptions {
  cell: CellName;
  hidden: number;
  learningRate: number;
  window: number;
  /** Windows per update. */
  batch: number;
  seed: number;
}

export const DEFAULT_CHAR_TRAIN: CharTrainOptions = {
  cell: 'gru',
  hidden: 32,
  learningRate: 0.02,
  window: 24,
  batch: 4,
  seed: 1,
};

/** Training state: one step draws random windows, backpropagates and takes an Adam step. */
export class CharTrainer {
  readonly model: CharLM;
  readonly rng: Rng;
  steps = 0;
  history: number[] = [];

  constructor(
    readonly corpus: Corpus,
    readonly options: CharTrainOptions,
    /** Characters held back for validation (never trained on). */
    readonly holdout = 0,
  ) {
    this.rng = createRng(options.seed);
    this.model = new CharLM(
      { cell: options.cell, hidden: options.hidden, vocab: corpus.alphabet.length },
      createRng(options.seed + 1),
    );
  }

  private get trainLength(): number {
    return this.corpus.tokens.length - this.holdout;
  }

  step(): number {
    const { window, batch, learningRate } = this.options;
    this.model.zeroGrad();
    let loss = 0;
    for (let b = 0; b < batch; b++) {
      const start = Math.floor(this.rng.next() * (this.trainLength - window - 1));
      const { loss: l, trace } = this.model.forward(
        this.corpus.tokens.subarray(start, start + window + 1),
      );
      this.model.backward(trace, 1 / batch);
      loss += l / batch;
    }
    this.model.adamStep(learningRate);
    this.steps += 1;
    this.history.push(loss);
    return loss;
  }

  /** Loss on the held-out tail, or on the training text when nothing was held back. */
  validationLoss(): number {
    const start = this.holdout > 0 ? this.trainLength : 0;
    const end = this.corpus.tokens.length;
    const window = this.options.window;
    let sum = 0;
    let n = 0;
    for (let s = start; s + window + 1 <= end; s += window) {
      sum += this.model.evaluate(this.corpus.tokens.subarray(s, s + window + 1));
      n += 1;
    }
    return n ? sum / n : NaN;
  }

  generate(length: number, temperature: number, seed: number, prefix = ''): string {
    const seedTokens = [...prefix]
      .map((c) => this.corpus.alphabet.indexOf(c))
      .filter((i) => i >= 0);
    const start = seedTokens.length ? seedTokens : [this.corpus.tokens[0] as number];
    return decode(this.corpus, this.model.sample(start, length, temperature, createRng(seed)));
  }
}
