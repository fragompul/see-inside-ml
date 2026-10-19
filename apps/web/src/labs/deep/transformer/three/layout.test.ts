import { readFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import {
  createTransformer,
  nextTokenDistribution,
  type TokenizerData,
  type TransformerManifest,
} from '@sim/deep';
import { describe, expect, it } from 'vitest';
import {
  COLUMN_W,
  HEAD_SIZE,
  SLAB_GAP,
  SLAB_H,
  buildSlabs,
  buildStages,
  cellIndex,
  cellOf,
  cellPosition,
  columnX,
  headCells,
  headOrigin,
  outputBars,
  outputZ,
  rowHeight,
  slabScale,
} from './layout';

const dir = resolve(
  dirname(fileURLToPath(import.meta.url)),
  '../../../../../../../packages/deep/models/tiny-gpt',
);
const read = (name: string) => readFileSync(resolve(dir, name));
const json = <T>(name: string) => JSON.parse(read(name).toString('utf-8')) as T;
const raw = read('weights.bin');
const { model, tokenizer } = createTransformer(
  json<TransformerManifest>('model.json'),
  raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength) as ArrayBuffer,
  json<TokenizerData>('tokenizer.json'),
);
const ids = tokenizer.encode('Anna found a ball. She');
const trace = model.forward(ids);
const T = ids.length;
const L = model.config.layers;

describe('slabs', () => {
  const slabs = buildSlabs(trace);

  it('has one slab for the embedding, five per block and the final norm, in depth order', () => {
    expect(slabs).toHaveLength(1 + 5 * L + 1);
    expect(slabs[0]?.id).toBe('embed');
    expect(slabs.at(-1)?.id).toBe('final');
    slabs.forEach((s, i) => expect(s.z).toBeCloseTo(-i * SLAB_GAP, 12));
    expect(new Set(slabs.map((s) => s.id)).size).toBe(slabs.length);
  });

  it('carries token-major data with the right shape for every slab', () => {
    for (const s of slabs) {
      expect(s.tokens).toBe(T);
      expect(s.values.length).toBe(T * s.rows);
      expect(s.rows).toBe(s.kind === 'hidden' ? 4 * model.config.d : model.config.d);
    }
  });

  it('puts the real residual stream into the stream slabs', () => {
    const mid = slabs.find((s) => s.id === 'mid-0');
    expect(Array.from(mid?.values ?? [])).toEqual(Array.from(trace.layers[0]?.residMid.data ?? []));
  });

  it('places the output bars one gap behind the last slab', () => {
    expect(outputZ(slabs)).toBeCloseTo((slabs.at(-1)?.z ?? 0) - SLAB_GAP, 12);
  });
});

describe('stages', () => {
  const slabs = buildSlabs(trace);
  const stages = buildStages(trace, slabs);

  it('walks the forward pass in order and ends at the probabilities', () => {
    expect(stages[0]?.id).toBe('embed');
    expect(stages.at(-1)?.output).toBe(true);
    expect(stages).toHaveLength(1 + 4 * L + 2);
    const zs = stages.map((s) => s.z);
    expect([...zs].sort((a, b) => b - a)).toEqual(zs); // monotonically deeper
  });

  it('only refers to slabs that exist, and attention stages show their own heads', () => {
    const ids = new Set(slabs.map((s) => s.id));
    for (const stage of stages) for (const id of stage.slabIds) expect(ids.has(id)).toBe(true);
    stages.filter((s) => s.id.startsWith('attention')).forEach((s, l) => expect(s.heads).toBe(l));
    expect(stages.every((s) => s.text.length > 40)).toBe(true);
  });
});

describe('cells', () => {
  it('index and (token,row) are inverse and a token column is contiguous', () => {
    const rows = 64;
    for (const [t, r] of [
      [0, 0],
      [3, 17],
      [T - 1, rows - 1],
    ] as const) {
      expect(cellOf(cellIndex(t, r, rows), rows)).toEqual({ token: t, row: r });
    }
    expect(cellIndex(2, 0, rows)).toBe(cellIndex(1, rows - 1, rows) + 1);
  });

  it('every slab is SLAB_H tall regardless of its rows, centred on the origin', () => {
    for (const rows of [64, 256]) {
      const s = { rows, tokens: T, z: 0 };
      const bottom = cellPosition(s, 0, 0);
      const top = cellPosition(s, 0, rows - 1);
      expect(top.y - bottom.y + rowHeight(rows)).toBeCloseTo(SLAB_H, 10);
      expect(top.y + bottom.y).toBeCloseTo(0, 10);
    }
  });

  it('columns are evenly spaced and centred', () => {
    expect(columnX(0, T) + columnX(T - 1, T)).toBeCloseTo(0, 12);
    expect(columnX(1, T) - columnX(0, T)).toBeCloseTo(COLUMN_W, 12);
  });

  it('cells never overlap: they are smaller than their pitch', () => {
    const c = cellPosition({ rows: 256, tokens: T, z: 0 }, 1, 1);
    expect(c.w).toBeLessThan(COLUMN_W);
    expect(c.h).toBeLessThan(rowHeight(256));
  });
});

describe('attention heads', () => {
  const probs = trace.layers[0]!.probs[0]!;
  const heads = model.config.heads;

  it('shows only the unmasked lower triangle, and rows sum to 1', () => {
    const cells = headCells(probs, 0, heads);
    expect(cells).toHaveLength((T * (T + 1)) / 2);
    expect(cells.every((c) => c.key <= c.query)).toBe(true);
    const lastRow = cells.filter((c) => c.query === T - 1).reduce((s, c) => s + c.weight, 0);
    expect(lastRow).toBeCloseTo(1, 10);
  });

  it('lays heads side by side without overlapping, under the slab', () => {
    const origins = Array.from({ length: heads }, (_, h) => headOrigin(h, heads));
    for (let h = 1; h < heads; h++) {
      expect((origins[h]?.x ?? 0) - (origins[h - 1]?.x ?? 0)).toBeGreaterThan(HEAD_SIZE);
    }
    expect(origins.every((o) => o.y + HEAD_SIZE / 2 < -SLAB_H / 2)).toBe(true);
    const xs = origins.map((o) => o.x);
    expect(xs.reduce((a, b) => a + b, 0)).toBeCloseTo(0, 10);
  });

  it('orders rows top to bottom (query 0 at the top)', () => {
    const cells = headCells(probs, 1, heads);
    const q0 = cells.find((c) => c.query === 0)!;
    const qLast = cells.find((c) => c.query === T - 1)!;
    expect(q0.y).toBeGreaterThan(qLast.y);
    // and every head matrix stays inside its square
    const origin = headOrigin(1, heads);
    for (const c of cells) {
      expect(Math.abs(c.x - origin.x)).toBeLessThanOrEqual(HEAD_SIZE / 2);
      expect(Math.abs(c.y - origin.y)).toBeLessThanOrEqual(HEAD_SIZE / 2);
    }
  });
});

describe('output bars', () => {
  const dist = nextTokenDistribution(
    trace.logits.data.slice((T - 1) * trace.logits.cols, T * trace.logits.cols),
    { temperature: 1, topK: 0, topP: 1 },
  );
  const bars = outputBars(dist);

  it('is sorted, scaled to the tallest bar and centred', () => {
    expect(bars).toHaveLength(10);
    expect(bars[0]?.height).toBeCloseTo(SLAB_H, 12);
    for (let i = 1; i < bars.length; i++) {
      expect(bars[i]?.height).toBeLessThanOrEqual(bars[i - 1]?.height as number);
    }
    expect(bars.reduce((s, b) => s + b.x, 0)).toBeCloseTo(0, 10);
  });

  it('shows the real probabilities and flags filtered tokens', () => {
    expect(bars[0]?.prob).toBeCloseTo(dist.candidates[0]?.prob as number, 12);
    const filtered = outputBars(
      nextTokenDistribution(trace.logits.data.slice((T - 1) * 512, T * 512), {
        temperature: 1,
        topK: 3,
        topP: 1,
      }),
    );
    expect(filtered.filter((b) => b.kept)).toHaveLength(3);
  });

  it('never collapses a tiny probability to nothing', () => {
    expect(bars.every((b) => b.height >= 0.03)).toBe(true);
  });
});

describe('slabScale', () => {
  it('is the largest magnitude and never zero', () => {
    expect(slabScale([1, -4, 2])).toBe(4);
    expect(slabScale([0, 0])).toBeGreaterThan(0);
  });
});
