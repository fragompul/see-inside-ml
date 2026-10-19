/**
 * Retrieval for a retrieval-augmented system: documents and questions become sparse vectors of weighted terms and the
 * documents closest to a question are returned. Terms are either whole words or overlapping groups of characters
 * (character n-grams); character grams still match a name with a typo in it, whole words do not.
 *     weight(term, doc) = (1 + log count) * idf(term),   idf = log((N + 1) / (df + 1)) + 1,   cosine similarity
 */
export type TermKind = 'words' | 'chars';

export const words = (text: string): string[] => text.toLowerCase().match(/[a-z0-9]+/g) ?? [];

export function charGrams(text: string, n = 3): string[] {
  const out: string[] = [];
  for (const w of words(text)) {
    const padded = `^${w}$`;
    for (let i = 0; i + n <= padded.length; i++) out.push(padded.slice(i, i + n));
  }
  return out;
}

const terms = (text: string, kind: TermKind): string[] =>
  kind === 'words' ? words(text) : charGrams(text);

type Sparse = Map<string, number>;

export class RetrievalIndex {
  private readonly idf = new Map<string, number>();
  private readonly vectors: Sparse[] = [];

  constructor(
    readonly documents: readonly string[],
    readonly kind: TermKind,
  ) {
    const counts = documents.map((d) => {
      const m = new Map<string, number>();
      for (const t of terms(d, kind)) m.set(t, (m.get(t) ?? 0) + 1);
      return m;
    });
    const df = new Map<string, number>();
    for (const m of counts) for (const t of m.keys()) df.set(t, (df.get(t) ?? 0) + 1);
    const n = documents.length;
    for (const [t, d] of df) this.idf.set(t, Math.log((n + 1) / (d + 1)) + 1);
    for (const m of counts) this.vectors.push(this.weigh(m));
  }

  private weigh(counts: Map<string, number>): Sparse {
    const v: Sparse = new Map();
    let norm = 0;
    for (const [t, c] of counts) {
      const idf = this.idf.get(t);
      if (idf === undefined) continue; // a term no document contains cannot match anything
      const w = (1 + Math.log(c)) * idf;
      v.set(t, w);
      norm += w * w;
    }
    norm = Math.sqrt(norm);
    if (norm > 0) for (const [t, w] of v) v.set(t, w / norm);
    return v;
  }

  /** Cosine similarity of the question with every document. */
  scores(query: string): number[] {
    const counts = new Map<string, number>();
    for (const t of terms(query, this.kind)) counts.set(t, (counts.get(t) ?? 0) + 1);
    const q = this.weigh(counts);
    return this.vectors.map((d) => {
      let dot = 0;
      for (const [t, w] of q) dot += w * (d.get(t) ?? 0);
      return dot;
    });
  }

  /** Indices of the `k` best documents, best first. */
  search(query: string, k: number): number[] {
    const s = this.scores(query);
    return Array.from(s.keys())
      .sort((a, b) => (s[b] as number) - (s[a] as number) || a - b)
      .slice(0, k);
  }

  /** Share of questions whose right document is among the top `k`. */
  recallAtK(questions: readonly string[], answers: readonly number[], k: number): number {
    let hit = 0;
    questions.forEach((q, i) => {
      if (this.search(q, k).includes(answers[i] as number)) hit++;
    });
    return hit / questions.length;
  }
}
