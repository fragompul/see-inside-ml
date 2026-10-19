import { Matrix, createRng, permutation } from '@sim/core';
import { pca } from '../transformer/embeddings';
import {
  ALL_ANALOGIES,
  ANALOGY_GROUPS,
  VOCABULARY,
  categoryOf,
  generateCorpus,
  type Analogy,
} from './corpus';
import { SkipGram, W2V_LIMITS, cosine, nearestWords, noiseSampler } from './word2vec';

export interface NlpLabConfig {
  /** Size of every word vector. */
  dim: number;
  /** Words on each side of the centre word that count as its context. */
  window: number;
  /** Random words contrasted with each true context word. */
  negatives: number;
  /** Learning rate per (centre, context) pair. */
  lr: number;
  /** (Centre, context) pairs per training step. */
  batchSize: number;
  /** Sentences of the toy-world corpus. */
  sentences: number;
  seed: number;
}

export const DEFAULT_NLP_LAB: NlpLabConfig = {
  dim: 16,
  window: 4,
  negatives: 5,
  lr: 0.05,
  batchSize: 64,
  sentences: 1500,
  seed: 5,
};

export const NLP_LAB_LIMITS = { sentences: [200, 4000], batchSize: [8, 512] } as const;

/** A loss above this (a fresh model starts near 4) means training has blown up. */
export const DIVERGED_LOSS = 25;

/** Statistics are recorded every this many steps. */
export const NLP_RECORD_EVERY = 25;

export interface NlpStats {
  step: number;
  /** Mean skip-gram loss since the last record. */
  loss: number;
  /** Share of analogy questions answered correctly (top-1). */
  analogy: number;
  /** Share of each content word's 3 nearest neighbours that belong to its own category. */
  purity: number;
}

export function validateNlpLab(c: NlpLabConfig): void {
  const within = (v: number, [lo, hi]: readonly [number, number], name: string) => {
    if (!Number.isInteger(v) || v < lo || v > hi)
      throw new RangeError(`${name} must be an integer in [${lo}, ${hi}]`);
  };
  within(c.dim, W2V_LIMITS.dim, 'Dimension');
  within(c.window, W2V_LIMITS.window, 'Window');
  within(c.negatives, W2V_LIMITS.negatives, 'Negatives');
  within(c.batchSize, NLP_LAB_LIMITS.batchSize, 'Batch size');
  within(c.sentences, NLP_LAB_LIMITS.sentences, 'Sentences');
  if (!(c.lr > 0)) throw new RangeError('Learning rate must be positive');
}

const stream = (seed: number, index: number) =>
  createRng((seed + Math.imul(index + 1, 0x9e3779b1)) >>> 0);

const dataKey = (c: NlpLabConfig) => JSON.stringify([c.window, c.sentences, c.seed]);
const modelKey = (c: NlpLabConfig) => JSON.stringify([c.dim, c.window, c.sentences, c.seed]);

const index = new Map<string, number>(VOCABULARY.map((w, i) => [w, i]));
export const wordId = (word: string): number => {
  const id = index.get(word);
  if (id === undefined) throw new RangeError(`Unknown word "${word}"`);
  return id;
};

const CONTENT: readonly string[] = VOCABULARY.filter((w) => categoryOf(w) !== 'function');

/** Word vectors learned from a toy-world corpus, with everything needed to explore them. */
export class NlpSession {
  private cfg: NlpLabConfig;
  private keys = { data: '', model: '' };
  model!: SkipGram;
  /** The corpus as lists of words (for display). */
  corpus!: string[][];
  private centers = new Int32Array(0);
  private contexts = new Int32Array(0);
  private draw!: (rng: ReturnType<typeof createRng>) => number;
  private order: Uint32Array = new Uint32Array(0);
  private cursor = 0;
  private shuffle = createRng(0);
  private negRng = createRng(0);
  step = 0;
  /** Set when the loss blew up (learning rate too high); training stops until a reset. */
  diverged = false;
  history: NlpStats[] = [];
  private acc = { loss: 0, count: 0 };

  constructor(config: NlpLabConfig) {
    validateNlpLab(config);
    this.cfg = { ...config };
    this.build(true);
  }

  get config(): Readonly<NlpLabConfig> {
    return this.cfg;
  }

  /** Number of (centre, context) training pairs the corpus yields. */
  get pairCount(): number {
    return this.centers.length;
  }

  /** Whole passes over the pairs completed so far. */
  get epoch(): number {
    return Math.floor((this.step * this.cfg.batchSize) / Math.max(1, this.centers.length));
  }

  /** Hyper-parameters apply live; data or model-shape changes rebuild (and restart training). */
  update(patch: Partial<NlpLabConfig>): 'live' | 'rebuilt' {
    const next = { ...this.cfg, ...patch };
    validateNlpLab(next);
    const dataChanged = dataKey(next) !== this.keys.data;
    const modelChanged = modelKey(next) !== this.keys.model;
    this.cfg = next;
    if (dataChanged || modelChanged) {
      this.build(dataChanged);
      return 'rebuilt';
    }
    return 'live';
  }

  reset(): void {
    this.build(false);
  }

  /** Runs `count` SGD steps; returns the statistics recorded meanwhile. */
  train(count: number): NlpStats[] {
    const recorded: NlpStats[] = [];
    const { batchSize, negatives, lr } = this.cfg;
    const total = this.centers.length;
    for (let n = 0; n < count && !this.diverged; n++) {
      if (this.cursor + batchSize > total || this.order.length === 0) {
        this.order = permutation(total, this.shuffle);
        this.cursor = 0;
      }
      const batch = this.order.subarray(this.cursor, this.cursor + batchSize);
      this.cursor += batchSize;
      const centers = Int32Array.from(batch, (i) => this.centers[i] as number);
      const contexts = Int32Array.from(batch, (i) => this.contexts[i] as number);
      const negs = Int32Array.from({ length: batchSize * negatives }, () => this.draw(this.negRng));
      const loss = this.model.step(centers, contexts, negs, negatives, lr);
      if (!Number.isFinite(loss) || loss > DIVERGED_LOSS) {
        this.diverged = true;
        break;
      }
      this.step += 1;
      this.acc.loss += loss;
      this.acc.count += 1;
      if (this.step % NLP_RECORD_EVERY === 0) {
        const stats: NlpStats = {
          step: this.step,
          loss: this.acc.loss / this.acc.count,
          analogy: this.analogyAccuracy(),
          purity: this.neighbourPurity(),
        };
        this.acc = { loss: 0, count: 0 };
        this.history.push(stats);
        recorded.push(stats);
      }
    }
    return recorded;
  }

  /** The embedding of every word, `vocabulary × dim`. */
  vectors(): Float64Array {
    return this.model.input;
  }

  vector(word: string): Float64Array {
    return this.model.vector(wordId(word));
  }

  similarity(a: string, b: string): number {
    return cosine(this.vector(a), this.vector(b));
  }

  /** The `k` words closest to `word` by cosine similarity (not including itself). */
  nearest(word: string, k = 5): { word: string; similarity: number }[] {
    const id = wordId(word);
    return nearestWords(this.model.input, this.cfg.dim, this.vector(word), k, [id]).map((n) => ({
      word: VOCABULARY[n.id] as string,
      similarity: n.similarity,
    }));
  }

  /** "a is to b as c is to ?": the words closest to `b − a + c`, excluding a, b and c. */
  analogy(a: string, b: string, c: string, k = 3): { word: string; similarity: number }[] {
    const va = this.vector(a);
    const vb = this.vector(b);
    const vc = this.vector(c);
    const query = vb.map((v, i) => v - (va[i] as number) + (vc[i] as number));
    return nearestWords(this.model.input, this.cfg.dim, query, k, [
      wordId(a),
      wordId(b),
      wordId(c),
    ]).map((n) => ({ word: VOCABULARY[n.id] as string, similarity: n.similarity }));
  }

  /** Whether the top answer to each question in `questions` is the expected word. */
  analogyAccuracy(questions: readonly Analogy[] = ALL_ANALOGIES): number {
    let right = 0;
    for (const [a, b, c, d] of questions) if (this.analogy(a, b, c, 1)[0]?.word === d) right += 1;
    return right / questions.length;
  }

  /** Accuracy per named group of analogies. */
  analogyAccuracyByGroup(): { id: string; label: string; accuracy: number }[] {
    return ANALOGY_GROUPS.map((g) => ({
      id: g.id,
      label: g.label,
      accuracy: this.analogyAccuracy(g.questions),
    }));
  }

  /** How often a content word's 3 nearest neighbours are in its own category (0..1). */
  neighbourPurity(): number {
    let same = 0;
    let total = 0;
    for (const word of CONTENT) {
      const category = categoryOf(word);
      for (const n of this.nearest(word, 3)) {
        total += 1;
        if (categoryOf(n.word) === category) same += 1;
      }
    }
    return same / total;
  }

  /** Content words on the plane of their two main principal axes (`words × 2`, row-major). */
  projection(): { words: string[]; coords: Float64Array; explained: number[] } {
    const { dim } = this.cfg;
    const m = new Matrix(CONTENT.length, dim);
    CONTENT.forEach((w, row) => m.data.set(this.vector(w), row * dim));
    const result = pca(m, 2);
    return {
      words: [...CONTENT],
      coords: Float64Array.from(result.projection.data),
      explained: result.explained,
    };
  }

  private build(regenerate: boolean): void {
    const c = this.cfg;
    if (regenerate) {
      this.corpus = generateCorpus(c.sentences, stream(c.seed, 0));
      const centers: number[] = [];
      const contexts: number[] = [];
      const counts = new Array<number>(VOCABULARY.length).fill(0);
      for (const sentence of this.corpus) {
        const ids = sentence.map(wordId);
        ids.forEach((id, i) => {
          counts[id] = (counts[id] as number) + 1;
          for (let o = -c.window; o <= c.window; o++) {
            const j = i + o;
            if (o !== 0 && j >= 0 && j < ids.length) {
              centers.push(id);
              contexts.push(ids[j] as number);
            }
          }
        });
      }
      this.centers = Int32Array.from(centers);
      this.contexts = Int32Array.from(contexts);
      this.draw = noiseSampler(counts.map((n) => Math.max(1, n)));
    }
    this.model = new SkipGram({ vocabulary: VOCABULARY.length, dim: c.dim }, stream(c.seed, 1));
    this.shuffle = stream(c.seed, 2);
    this.negRng = stream(c.seed, 3);
    this.order = new Uint32Array(0);
    this.cursor = 0;
    this.step = 0;
    this.diverged = false;
    this.history = [];
    this.acc = { loss: 0, count: 0 };
    this.keys = { data: dataKey(c), model: modelKey(c) };
  }
}
