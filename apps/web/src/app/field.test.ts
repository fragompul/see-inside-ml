import { describe, expect, it } from 'vitest';
import { buildField } from './field';

describe('buildField', () => {
  const sizes = [3, 5, 4, 2];
  const field = buildField(sizes, 1);

  it('creates one node per neuron', () => {
    expect(field.nodes).toHaveLength(14);
    expect(field.layers.map((l) => l.length)).toEqual(sizes);
  });

  it('fully connects consecutive layers only', () => {
    expect(field.edges).toHaveLength(3 * 5 + 5 * 4 + 4 * 2);
    for (const edge of field.edges) {
      const from = field.nodes[edge.from];
      const to = field.nodes[edge.to];
      expect(to?.layer).toBe((from?.layer ?? -1) + 1);
    }
  });

  it('keeps weights inside (-1, 1) and mixes both signs', () => {
    const weights = field.edges.map((e) => e.weight);
    expect(weights.every((w) => w > -1 && w < 1)).toBe(true);
    expect(weights.some((w) => w > 0)).toBe(true);
    expect(weights.some((w) => w < 0)).toBe(true);
  });

  it('indexes incoming / outgoing edges consistently', () => {
    const outgoing = field.outgoing.reduce((sum, e) => sum + e.length, 0);
    const incoming = field.incoming.reduce((sum, e) => sum + e.length, 0);
    expect(outgoing).toBe(field.edges.length);
    expect(incoming).toBe(field.edges.length);
  });

  it('is deterministic for a given seed', () => {
    expect(buildField(sizes, 1).edges).toEqual(field.edges);
    expect(buildField(sizes, 2).edges).not.toEqual(field.edges);
  });
});
