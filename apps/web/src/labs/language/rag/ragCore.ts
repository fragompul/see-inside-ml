import { Matrix, createRng, type Rng } from '@sim/core';
import { DEFAULT_TOOLBOX, RetrievalIndex, charGrams, trainToolbox, type TermKind } from '@sim/deep';

/**
 * A question-answering toy with a knowledge base of invented facts: "The colour of Velmora is teal." A closed-book
 * model must carry all the facts in its weights; a retrieval-augmented system looks the relevant passage up and reads the
 * answer from it.
 */
export const COLOURS = [
  'teal',
  'amber',
  'violet',
  'crimson',
  'olive',
  'ivory',
  'indigo',
  'scarlet',
  'azure',
  'maroon',
  'coral',
  'umber',
];

export interface RagScenario {
  /** Number of facts in the knowledge base. */
  facts: number;
  /** Hidden units of the closed-book network (its memory). */
  memory: number;
  /** What the retriever compares: whole words or groups of characters. */
  retriever: TermKind;
  /** Share of questions in which the name is misspelt (one letter deleted or swapped). */
  typos: number;
  /** Share of facts that change after training. */
  updated: number;
}

export const DEFAULT_RAG: RagScenario = {
  facts: 200,
  memory: 8,
  retriever: 'words',
  typos: 0,
  updated: 0.1,
};

export interface RagResult {
  scenario: RagScenario;
  /** Closed-book and retrieval accuracy for knowledge bases of growing size, with this memory and retriever. */
  sweep: { facts: number; closedBook: number; rag: number }[];
  /** Accuracy on the facts it was built or trained on. */
  closedBook: number;
  rag: number;
  /** Share of questions for which the right passage is the retriever's first choice. */
  recall: number;
  /** After `updated` of the facts change: accuracy on the changed facts only, without retraining the closed-book model. */
  closedBookUpdated: number;
  ragUpdated: number;
  /** Parameters of the closed-book model against the number of facts. */
  parameters: number;
  /** Example questions with what each system said. */
  examples: { question: string; truth: string; closedBook: string; rag: string; passage: string }[];
}

const SYLLABLES = [
  'ka',
  'vel',
  'mor',
  'ost',
  'ra',
  'ti',
  'lu',
  'dren',
  'sa',
  'bel',
  'nor',
  'quin',
  'ze',
  'fa',
  'lan',
  'pri',
  'to',
  'mir',
  'dun',
  'ey',
];

function makeNames(n: number, rng: Rng): string[] {
  const seen = new Set<string>();
  while (seen.size < n) {
    const parts = 2 + Math.floor(rng.next() * 2);
    const name = Array.from(
      { length: parts },
      () => SYLLABLES[Math.floor(rng.next() * SYLLABLES.length)] as string,
    ).join('');
    seen.add(name.charAt(0).toUpperCase() + name.slice(1));
  }
  return [...seen];
}

const FEATURES = 160;

function hashed(name: string): Float64Array {
  const x = new Float64Array(FEATURES);
  for (const g of charGrams(name)) {
    let h = 7;
    for (let i = 0; i < g.length; i++) h = (h * 31 + g.charCodeAt(i)) % 100003;
    x[h % FEATURES] = 1;
  }
  return x;
}

function typo(name: string, rng: Rng): string {
  if (name.length < 4) return name;
  const i = 1 + Math.floor(rng.next() * (name.length - 2));
  return rng.next() < 0.5
    ? name.slice(0, i) + name.slice(i + 1)
    : name.slice(0, i) + (name[i + 1] ?? '') + (name[i] ?? '') + name.slice(i + 2);
}

const passage = (name: string, colour: string) => `The colour of ${name} is ${colour}.`;

const cache = new Map<string, RagResult>();

export const SWEEP = [50, 100, 200, 400, 800];

export function runRag(s: RagScenario): RagResult {
  const result = core(s);
  const sweep = SWEEP.map((facts) => {
    const r = core({ ...s, facts });
    return { facts, closedBook: r.closedBook, rag: r.rag };
  });
  return { ...result, sweep };
}

function core(s: RagScenario): RagResult {
  const key = JSON.stringify(s);
  const hit = cache.get(key);
  if (hit) return hit;
  const rng = createRng(11);
  const names = makeNames(s.facts, rng);
  const answers = names.map(() => Math.floor(rng.next() * COLOURS.length));

  // the closed-book model memorises name -> colour
  const x = new Matrix(s.facts, FEATURES);
  names.forEach((n, i) => x.data.set(hashed(n), i * FEATURES));
  const trained = trainToolbox(
    { ...DEFAULT_TOOLBOX, layers: [FEATURES, s.memory, COLOURS.length], seed: 3 },
    { x, y: answers },
    { x, y: answers },
    { epochs: 80, batchSize: 32, learningRate: 0.01, optimizer: 'adam', patience: 0, seed: 3 },
  );
  const closedGuess = (name: string) =>
    Array.from(trained.model.predict(new Matrix(1, FEATURES, hashed(name))))[0] as number;

  const docs = names.map((n, i) => passage(n, COLOURS[answers[i] as number] as string));
  const index = new RetrievalIndex(docs, s.retriever);
  const questions = names.map((n, i) => {
    const asked = rng.next() < s.typos ? typo(n, rng) : n;
    return { asked, question: `What is the colour of ${asked}?`, i };
  });
  let closed = 0;
  let rag = 0;
  let first = 0;
  const examples: RagResult['examples'] = [];
  for (const { asked, question, i } of questions) {
    const found = index.search(question, 1)[0] as number;
    const said = (docs[found] as string).replace(/\.$/, '').split(' ').pop() as string;
    const cb = COLOURS[closedGuess(asked)] as string;
    const truth = COLOURS[answers[i] as number] as string;
    if (cb === truth) closed++;
    if (said === truth) rag++;
    if (found === i) first++;
    if (examples.length < 6)
      examples.push({ question, truth, closedBook: cb, rag: said, passage: docs[found] as string });
  }

  // some facts change: the retrieval system just rewrites a passage, the closed-book model is not retrained
  const changed = Array.from(
    { length: Math.max(1, Math.round(s.updated * s.facts)) },
    (_, k) => (k * Math.floor(1 / Math.max(s.updated, 0.01))) % s.facts,
  );
  const newAnswers = answers.slice();
  const newDocs = docs.slice();
  for (const i of new Set(changed)) {
    newAnswers[i] = ((answers[i] as number) + 5) % COLOURS.length;
    newDocs[i] = passage(names[i] as string, COLOURS[newAnswers[i] as number] as string);
  }
  const updatedIndex = new RetrievalIndex(newDocs, s.retriever);
  let ragNew = 0;
  let closedNew = 0;
  const touched = [...new Set(changed)];
  for (const i of touched) {
    const found = updatedIndex.search(`What is the colour of ${names[i]}?`, 1)[0] as number;
    if (
      (newDocs[found] as string).replace(/\.$/, '').split(' ').pop() ===
      COLOURS[newAnswers[i] as number]
    )
      ragNew++;
    if (closedGuess(names[i] as string) === newAnswers[i]) closedNew++;
  }
  const out: RagResult = {
    scenario: s,
    sweep: [],
    closedBook: closed / s.facts,
    rag: rag / s.facts,
    recall: first / s.facts,
    closedBookUpdated: closedNew / touched.length,
    ragUpdated: ragNew / touched.length,
    parameters: FEATURES * s.memory + s.memory + s.memory * COLOURS.length + COLOURS.length,
    examples,
  };
  cache.set(key, out);
  return out;
}
