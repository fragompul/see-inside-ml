import { Matrix, addRowVector, hconcat, matmul, matmulBT, sliceCols } from '@sim/core';

export const LN_EPS = 1e-5;
const SQRT_2_OVER_PI = Math.sqrt(2 / Math.PI);

export interface TransformerConfig {
  vocab: number;
  /** Maximum sequence length (size of the learned position table). */
  ctx: number;
  /** Model width. */
  d: number;
  layers: number;
  heads: number;
}

export interface TensorEntry {
  name: string;
  shape: number[];
  /** Byte offset inside weights.bin. */
  offset: number;
}

export interface TransformerManifest {
  config: TransformerConfig;
  eotId: number;
  dtype: 'float32';
  tensors: TensorEntry[];
}

/** Switching attention heads off, to see what each one contributes. */
export interface ForwardOptions {
  /** Heads to silence, as `"layer.head"` (both counted from 0). */
  ablate?: ReadonlySet<string>;
  /** `zero` replaces the head's output by zeros; `mean` by its average over the positions of this text. */
  ablateMode?: 'zero' | 'mean';
}

/** Everything a block computed for one forward pass, ready to be drawn. */
export interface LayerTrace {
  /** LayerNorm output feeding attention (T × d). */
  ln1: Matrix;
  /** Per head (T × headDim). */
  q: Matrix[];
  k: Matrix[];
  v: Matrix[];
  /** Per head, scaled dot-products before masking and softmax (T × T). */
  scores: Matrix[];
  /** Per head, attention weights: each row sums to 1 and is zero above the diagonal (T × T). */
  probs: Matrix[];
  /** Heads concatenated (T × d), before the output projection. */
  headsOut: Matrix;
  attnOut: Matrix;
  /** Residual stream after attention is added. */
  residMid: Matrix;
  ln2: Matrix;
  /** MLP hidden layer before / after GELU (T × 4d). */
  mlpPre: Matrix;
  mlpAct: Matrix;
  mlpOut: Matrix;
  /** Residual stream leaving the block. */
  residOut: Matrix;
}

export interface ForwardTrace {
  ids: number[];
  /** Token embeddings, position embeddings and their sum (all T × d). */
  tok: Matrix;
  pos: Matrix;
  embed: Matrix;
  layers: LayerTrace[];
  finalLn: Matrix;
  /** Next-token scores for every position (T × vocab). */
  logits: Matrix;
}

export const gelu = (x: number): number =>
  0.5 * x * (1 + Math.tanh(SQRT_2_OVER_PI * (x + 0.044715 * x * x * x)));

/** Row-wise LayerNorm: (x − mean) / sqrt(var + ε) · g + b, with the population variance. */
export function layerNorm(x: Matrix, g: Float64Array, b: Float64Array): Matrix {
  const out = new Matrix(x.rows, x.cols);
  for (let i = 0; i < x.rows; i++) {
    let mean = 0;
    for (let j = 0; j < x.cols; j++) mean += x.get(i, j);
    mean /= x.cols;
    let variance = 0;
    for (let j = 0; j < x.cols; j++) variance += (x.get(i, j) - mean) ** 2;
    variance /= x.cols;
    const inv = 1 / Math.sqrt(variance + LN_EPS);
    for (let j = 0; j < x.cols; j++) {
      out.set(i, j, (x.get(i, j) - mean) * inv * (g[j] as number) + (b[j] as number));
    }
  }
  return out;
}

/** Row-wise softmax where entries right of the diagonal are masked out (causal attention). */
export function causalSoftmax(scores: Matrix): Matrix {
  const out = new Matrix(scores.rows, scores.cols);
  for (let i = 0; i < scores.rows; i++) {
    let max = -Infinity;
    for (let j = 0; j <= i; j++) max = Math.max(max, scores.get(i, j));
    let sum = 0;
    for (let j = 0; j <= i; j++) {
      const e = Math.exp(scores.get(i, j) - max);
      out.set(i, j, e);
      sum += e;
    }
    for (let j = 0; j <= i; j++) out.set(i, j, out.get(i, j) / sum);
  }
  return out;
}

function linear(x: Matrix, w: Matrix, b: Float64Array): Matrix {
  const out = matmul(x, w);
  addRowVector(out, b);
  return out;
}

/**
 * GPT-2-style decoder-only transformer, forward pass only. Mirrors tools/transformer/gpt.py
 * operation for operation (parity-tested) and keeps every intermediate tensor on request.
 */
export class TransformerModel {
  readonly headDim: number;
  private readonly matrices = new Map<string, Matrix>();
  private readonly vectors = new Map<string, Float64Array>();

  constructor(
    readonly config: TransformerConfig,
    tensors: ReadonlyMap<string, { shape: number[]; data: Float64Array }>,
    readonly eotId: number,
  ) {
    if (config.d % config.heads !== 0) throw new RangeError('d must be divisible by heads');
    this.headDim = config.d / config.heads;
    for (const [name, { shape, data }] of tensors) {
      if (shape.length === 2) {
        this.matrices.set(name, new Matrix(shape[0] as number, shape[1] as number, data));
      } else {
        this.vectors.set(name, data);
      }
    }
    this.validate();
  }

  /** Builds a model from `model.json` and the contents of `weights.bin`. */
  static fromBuffer(manifest: TransformerManifest, weights: ArrayBuffer): TransformerModel {
    const tensors = new Map<string, { shape: number[]; data: Float64Array }>();
    for (const entry of manifest.tensors) {
      const length = entry.shape.reduce((a, b) => a * b, 1);
      if (entry.offset + length * 4 > weights.byteLength) {
        throw new RangeError(`weights.bin is too short for tensor "${entry.name}"`);
      }
      // DataView avoids alignment requirements and fixes the byte order (little-endian).
      const view = new DataView(weights, entry.offset, length * 4);
      const data = new Float64Array(length);
      for (let i = 0; i < length; i++) data[i] = view.getFloat32(i * 4, true);
      tensors.set(entry.name, { shape: entry.shape, data });
    }
    return new TransformerModel(manifest.config, tensors, manifest.eotId);
  }

  private mat(name: string): Matrix {
    const m = this.matrices.get(name);
    if (!m) throw new Error(`Missing weight matrix "${name}"`);
    return m;
  }

  private vec(name: string): Float64Array {
    const v = this.vectors.get(name);
    if (!v) throw new Error(`Missing weight vector "${name}"`);
    return v;
  }

  private validate(): void {
    const { vocab, ctx, d, layers } = this.config;
    const shapeOf = (name: string, rows: number, cols: number) => {
      const m = this.mat(name);
      if (m.rows !== rows || m.cols !== cols) {
        throw new RangeError(`"${name}" should be ${rows}×${cols}, got ${m.rows}×${m.cols}`);
      }
    };
    shapeOf('wte', vocab, d);
    shapeOf('wpe', ctx, d);
    for (let i = 0; i < layers; i++) {
      shapeOf(`h${i}.attn.w`, d, 3 * d);
      shapeOf(`h${i}.proj.w`, d, d);
      shapeOf(`h${i}.fc.w`, d, 4 * d);
      shapeOf(`h${i}.fc2.w`, 4 * d, d);
    }
  }

  get parameterCount(): number {
    let n = 0;
    for (const m of this.matrices.values()) n += m.data.length;
    for (const v of this.vectors.values()) n += v.length;
    return n;
  }

  /** Token embedding table (vocab × d), e.g. for the embedding-space explorer. */
  get tokenEmbeddings(): Matrix {
    return this.mat('wte');
  }

  /**
   * Runs the model over `ids` (≤ ctx tokens). Returns the logits for every position and the full
   * trace of intermediate tensors.
   */
  forward(ids: readonly number[], options: ForwardOptions = {}): ForwardTrace {
    const { vocab, ctx, d, layers, heads } = this.config;
    const T = ids.length;
    if (T === 0) throw new RangeError('Need at least one token');
    if (T > ctx) throw new RangeError(`Sequence of ${T} tokens exceeds the context of ${ctx}`);
    for (const id of ids) {
      if (!Number.isInteger(id) || id < 0 || id >= vocab)
        throw new RangeError(`Invalid token id ${id}`);
    }
    const hd = this.headDim;
    const wte = this.mat('wte');
    const wpe = this.mat('wpe');

    const tok = new Matrix(T, d);
    const pos = new Matrix(T, d);
    const x0 = new Matrix(T, d);
    for (let t = 0; t < T; t++) {
      for (let j = 0; j < d; j++) {
        const a = wte.get(ids[t] as number, j);
        const p = wpe.get(t, j);
        tok.set(t, j, a);
        pos.set(t, j, p);
        x0.set(t, j, a + p);
      }
    }

    const layerTraces: LayerTrace[] = [];
    let x = x0;
    for (let l = 0; l < layers; l++) {
      const ln1 = layerNorm(x, this.vec(`h${l}.ln1.g`), this.vec(`h${l}.ln1.b`));
      const qkv = linear(ln1, this.mat(`h${l}.attn.w`), this.vec(`h${l}.attn.b`));

      const q: Matrix[] = [];
      const k: Matrix[] = [];
      const v: Matrix[] = [];
      const scores: Matrix[] = [];
      const probs: Matrix[] = [];
      const outs: Matrix[] = [];
      const scale = 1 / Math.sqrt(hd);
      for (let h = 0; h < heads; h++) {
        const qh = sliceCols(qkv, h * hd, (h + 1) * hd);
        const kh = sliceCols(qkv, d + h * hd, d + (h + 1) * hd);
        const vh = sliceCols(qkv, 2 * d + h * hd, 2 * d + (h + 1) * hd);
        const s = matmulBT(qh, kh);
        for (let i = 0; i < s.data.length; i++) s.data[i] = (s.data[i] as number) * scale;
        const p = causalSoftmax(s);
        q.push(qh);
        k.push(kh);
        v.push(vh);
        scores.push(s);
        probs.push(p);
        let out = matmul(p, vh);
        if (options.ablate?.has(`${l}.${h}`)) {
          out = new Matrix(out.rows, out.cols);
          if (options.ablateMode === 'mean') {
            const mean = new Float64Array(out.cols);
            const full = matmul(p, vh);
            for (let r = 0; r < full.rows; r++)
              for (let c = 0; c < full.cols; c++)
                mean[c] = (mean[c] as number) + full.get(r, c) / full.rows;
            for (let r = 0; r < out.rows; r++) out.data.set(mean, r * out.cols);
          }
        }
        outs.push(out);
      }
      const headsOut = outs.reduce((acc, m) => hconcat(acc, m));
      const attnOut = linear(headsOut, this.mat(`h${l}.proj.w`), this.vec(`h${l}.proj.b`));
      const residMid = addMatrices(x, attnOut);

      const ln2 = layerNorm(residMid, this.vec(`h${l}.ln2.g`), this.vec(`h${l}.ln2.b`));
      const mlpPre = linear(ln2, this.mat(`h${l}.fc.w`), this.vec(`h${l}.fc.b`));
      const mlpAct = mlpPre.map(gelu);
      const mlpOut = linear(mlpAct, this.mat(`h${l}.fc2.w`), this.vec(`h${l}.fc2.b`));
      const residOut = addMatrices(residMid, mlpOut);

      layerTraces.push({
        ln1,
        q,
        k,
        v,
        scores,
        probs,
        headsOut,
        attnOut,
        residMid,
        ln2,
        mlpPre,
        mlpAct,
        mlpOut,
        residOut,
      });
      x = residOut;
    }

    const finalLn = layerNorm(x, this.vec('lnf.g'), this.vec('lnf.b'));
    const logits = matmulBT(finalLn, wte); // output projection tied to the token embedding
    return { ids: [...ids], tok, pos, embed: x0, layers: layerTraces, finalLn, logits };
  }

  /** Logits for the token that would follow `ids`. */
  nextTokenLogits(ids: readonly number[]): Float64Array {
    const { logits } = this.forward(ids);
    return logits.data.slice((logits.rows - 1) * logits.cols);
  }
}

function addMatrices(a: Matrix, b: Matrix): Matrix {
  const out = new Matrix(a.rows, a.cols);
  for (let i = 0; i < a.data.length; i++)
    out.data[i] = (a.data[i] as number) + (b.data[i] as number);
  return out;
}
