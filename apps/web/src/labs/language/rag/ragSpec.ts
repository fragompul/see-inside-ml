import type { ParamSpec } from '@/shared/lab-kit';
import type { RagResult, RagScenario } from './ragCore';

export const ragParams: ParamSpec[] = [
  {
    key: 'facts',
    label: 'Facts in the knowledge base',
    kind: 'ladder',
    values: [50, 200, 800],
    format: (v) => String(v),
  },
  {
    key: 'memory',
    label: 'Memory of the closed-book model',
    kind: 'ladder',
    values: [8, 32],
    format: (v) => `${v} hidden units`,
  },
  {
    key: 'retriever',
    label: 'Retriever compares',
    kind: 'segmented',
    options: [
      { value: 'words', label: 'Whole words' },
      { value: 'chars', label: 'Letter groups' },
    ],
  },
  {
    key: 'typos',
    label: 'Questions with a misspelt name',
    kind: 'ladder',
    values: [0, 0.5, 1],
    format: (v) => `${v * 100}%`,
  },
];

export interface RagExperiment {
  id: string;
  title: string;
  text: string;
  scenario: Partial<RagScenario>;
  claim: (run: RagResult) => boolean;
}

const at = (r: RagResult, facts: number) =>
  r.sweep.find((p) => p.facts === facts) as RagResult['sweep'][number];

export const ragExperiments: RagExperiment[] = [
  {
    id: 'capacity',
    title: 'A closed-book model runs out of memory',
    text: 'A small network with 1,396 weights can learn 50 invented facts perfectly and 200 almost perfectly (97%), but with 800 facts it gets 68% right. The retrieval system, which keeps the facts as text, is right every time at every size.',
    scenario: { memory: 8, typos: 0, retriever: 'words' },
    claim: (r) =>
      at(r, 50).closedBook >= 0.99 &&
      at(r, 800).closedBook < 0.75 &&
      r.sweep.every((p) => p.rag >= 0.99),
  },
  {
    id: 'grow',
    title: 'Its memory grows with its weights',
    text: 'With four times as many weights (5,548) the same network learns all 800 facts. Closed-book knowledge costs parameters in proportion to the amount of knowledge.',
    scenario: { memory: 32, typos: 0, retriever: 'words' },
    claim: (r) => at(r, 800).closedBook >= 0.99 && r.parameters === 5548,
  },
  {
    id: 'update',
    title: 'Changing a fact is free for retrieval',
    text: 'One in ten facts changes. The retrieval system rewrites a passage and answers all the new facts correctly at once; the closed-book model, not retrained, still gives the old answer and gets none of them right.',
    scenario: { updated: 0.1 },
    claim: (r) => r.ragUpdated >= 0.99 && r.closedBookUpdated <= 0.05,
  },
  {
    id: 'typos-words',
    title: 'A word-matching retriever is brittle',
    text: 'When every question misspells the name, a retriever that compares whole words finds the right passage for 1 question in 100 and the system is right 12% of the time: the garbled name matches nothing.',
    scenario: { retriever: 'words', typos: 1, memory: 32, facts: 200 },
    claim: (r) => r.recall <= 0.05 && r.rag <= 0.2,
  },
  {
    id: 'typos-chars',
    title: 'Letter groups survive typos',
    text: 'Comparing groups of three letters, a misspelt name still shares most of its groups with the right passage: recall is 58% when every name is misspelt and the system answers 63% of questions, more than the closed-book network (41%), which has to guess from garbled features.',
    scenario: { retriever: 'chars', typos: 1, memory: 32, facts: 200 },
    claim: (r) => r.recall >= 0.5 && r.rag >= 0.55 && r.rag > r.closedBook + 0.15,
  },
];
