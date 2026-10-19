import { describe, expect, it } from 'vitest';
import { COLOURS, DEFAULT_RAG, SWEEP, runRag } from './ragCore';
import { ragExperiments } from './ragSpec';

describe('Retrieval-augmented generation lab', () => {
  for (const exp of ragExperiments) {
    it(`experiment "${exp.title}" holds`, () => {
      expect(exp.claim(runRag({ ...DEFAULT_RAG, ...exp.scenario }))).toBe(true);
    }, 200_000);
  }

  it('sweeps the sizes of the knowledge base and shows example questions', () => {
    const r = runRag({ ...DEFAULT_RAG, facts: 50 });
    expect(r.sweep.map((p) => p.facts)).toEqual(SWEEP);
    expect(r.examples).toHaveLength(6);
    for (const e of r.examples) expect(COLOURS).toContain(e.truth);
  });
});
