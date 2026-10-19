import { describe, expect, it } from 'vitest';
import { PlaygroundHost, type PlaygroundResponse } from './host';
import {
  DEFAULT_CONFIG,
  PlaygroundSession,
  gridCoordinates,
  type PlaygroundConfig,
} from './session';

// Available in Node and browsers; declared here because the engine's tsconfig has no DOM/Node typings.
declare const structuredClone: <T>(value: T) => T;

const config = (patch: Partial<PlaygroundConfig> = {}): PlaygroundConfig => ({
  ...structuredClone(DEFAULT_CONFIG),
  ...patch,
});

describe('PlaygroundSession', () => {
  it('learns the circle dataset (tanh MLP + Adam)', () => {
    const s = new PlaygroundSession(config({ hidden: [{ neurons: 6, activation: 'tanh' }] }));
    const history = s.step(300);
    const first = history[0];
    const last = history[history.length - 1];
    expect(last?.trainLoss).toBeLessThan((first?.trainLoss as number) * 0.5);
    expect(last?.testMetric).toBeGreaterThan(0.9);
  });

  it('learns a plane with a linear model (regression)', () => {
    const s = new PlaygroundSession(
      config({
        dataset: 'plane',
        noise: 0,
        hidden: [],
        features: ['X1', 'X2'],
        lr: 0.05,
        loss: 'mse',
      }),
    );
    const history = s.step(250);
    expect(history[history.length - 1]?.testMetric).toBeLessThan(0.01);
  });

  it('is fully reproducible for a seed, and reset replays the same run', () => {
    const a = new PlaygroundSession(config());
    const b = new PlaygroundSession(config());
    expect(a.step(5)).toEqual(b.step(5));

    const before = a.step(3);
    a.reset();
    a.step(5);
    expect(a.step(3)).toEqual(before);
  });

  it('applies hyper-parameters live without losing progress', () => {
    const s = new PlaygroundSession(config());
    s.step(5);
    expect(s.update({ lr: 0.01, optimizer: 'sgd', batchSize: 40, regularization: 'l2' })).toBe(
      'live',
    );
    expect(s.epoch).toBe(5);
    expect(s.history).toHaveLength(5);
    const stats = s.step(1)[0];
    expect(stats?.lr).toBeCloseTo(0.01);
    expect(stats?.epoch).toBe(6);
  });

  it('rebuilds on architecture changes but keeps the dataset', () => {
    const s = new PlaygroundSession(config());
    const before = s.scene().train.xy.slice();
    s.step(3);
    expect(s.update({ hidden: [{ neurons: 8, activation: 'relu' }] })).toBe('rebuilt');
    expect(s.epoch).toBe(0);
    expect(s.scene().train.xy).toEqual(before);
    expect(s.scene().layerSizes).toEqual([2, 8, 2]);
  });

  it('regenerates data when the dataset changes', () => {
    const s = new PlaygroundSession(config());
    const before = s.scene().train.xy.slice();
    expect(s.update({ dataset: 'xor' })).toBe('rebuilt');
    expect(s.scene().train.xy).not.toEqual(before);
  });

  it('switching problem type adapts the output layer', () => {
    const s = new PlaygroundSession(config({ classes: 3 }));
    expect(s.scene().layerSizes.at(-1)).toBe(3);
    s.update({ dataset: 'saddle' });
    expect(s.scene().problem).toBe('regression');
    expect(s.scene().layerSizes.at(-1)).toBe(1);
  });

  it('skip connections change parameter count but still train', () => {
    const plain = new PlaygroundSession(config());
    const skip = new PlaygroundSession(config({ skipConnections: true }));
    expect(skip.scene().parameterCount).toBeGreaterThan(plain.scene().parameterCount);
    expect(Number.isFinite(skip.step(5)[4]?.trainLoss)).toBe(true);
  });

  it('snapshots expose boundary and per-neuron maps on a grid', () => {
    const s = new PlaygroundSession(config({ classes: 3 }));
    s.step(2);
    const snap = s.snapshot({ resolution: 12, neuronMaps: true });
    expect(snap.boundary?.channels).toBe(3);
    expect(snap.boundary?.values.length).toBe(12 * 12 * 3);
    expect(snap.inputMaps).toHaveLength(2);
    expect(snap.neuronMaps).toHaveLength(3); // two hidden layers + output
    expect(snap.neuronMaps?.[0]).toHaveLength(4);
    expect(snap.neuronMaps?.[0]?.[0]?.length).toBe(144);
    // class probabilities sum to 1 in every cell
    const v = snap.boundary?.values as Float32Array;
    for (let cell = 0; cell < 144; cell++) {
      expect(
        (v[cell * 3] as number) + (v[cell * 3 + 1] as number) + (v[cell * 3 + 2] as number),
      ).toBeCloseTo(1, 5);
    }
    expect(s.snapshot().boundary).toBeNull();
  });

  it('can sample neuron maps at a coarser resolution than the boundary', () => {
    const s = new PlaygroundSession(config());
    const snap = s.snapshot({ resolution: 30, neuronMaps: true, neuronResolution: 10 });
    expect(snap.boundary?.resolution).toBe(30);
    expect(snap.neuronMaps?.[0]?.[0]?.length).toBe(100);
    expect(snap.inputMaps?.[0]?.length).toBe(100);
  });

  it('snapshots do not disturb training', () => {
    const a = new PlaygroundSession(config());
    const b = new PlaygroundSession(config());
    a.step(3);
    a.snapshot({ resolution: 20, neuronMaps: true });
    b.step(3);
    expect(a.step(3)).toEqual(b.step(3));
  });

  it('validates configuration', () => {
    expect(() => new PlaygroundSession(config({ features: [] }))).toThrow(/feature/i);
    expect(() => new PlaygroundSession(config({ features: ['X1', 'X1'] }))).toThrow(/Duplicate/);
    expect(() => new PlaygroundSession(config({ lr: 0 }))).toThrow(/Learning rate/);
    expect(() => new PlaygroundSession(config({ noise: 2 }))).toThrow(/noise/);
    expect(
      () =>
        new PlaygroundSession(
          config({ hidden: Array(9).fill({ neurons: 2, activation: 'relu' }) }),
        ),
    ).toThrow(/hidden layers/);
    const s = new PlaygroundSession(config());
    expect(() => s.update({ samples: 5 })).toThrow(/samples/);
    // a rejected update leaves the session usable and unchanged
    expect(s.config.samples).toBe(DEFAULT_CONFIG.samples);
    expect(s.step(1)).toHaveLength(1);
  });
});

describe('gridCoordinates', () => {
  it('uses cell centres with y increasing along rows', () => {
    const g = gridCoordinates(4);
    expect(g.rows).toBe(16);
    expect(g.get(0, 0)).toBeCloseTo(-4.5);
    expect(g.get(0, 1)).toBeCloseTo(-4.5);
    expect(g.get(3, 0)).toBeCloseTo(4.5);
    expect(g.get(4, 1)).toBeCloseTo(-1.5);
  });
});

describe('PlaygroundHost protocol', () => {
  const kinds = (r: PlaygroundResponse) => r.type;

  it('requires init first and reports errors instead of throwing', () => {
    const host = new PlaygroundHost();
    const res = host.handle({ id: 1, type: 'step', epochs: 1 });
    expect(res).toMatchObject({ id: 1, type: 'error' });
  });

  it('runs a full conversation and echoes request ids', () => {
    const host = new PlaygroundHost();
    const init = host.handle({
      id: 10,
      type: 'init',
      config: config(),
      snapshot: { resolution: 8 },
    });
    expect(kinds(init)).toBe('scene');
    expect(init.id).toBe(10);

    const step = host.handle({ id: 11, type: 'step', epochs: 4, snapshot: { resolution: 8 } });
    expect(step).toMatchObject({ id: 11, type: 'progress' });
    if (step.type === 'progress') {
      expect(step.stats).toHaveLength(4);
      expect(step.snapshot?.boundary?.resolution).toBe(8);
    }

    const live = host.handle({ id: 12, type: 'update', patch: { lr: 0.1 } });
    expect(kinds(live)).toBe('progress');
    const rebuilt = host.handle({ id: 13, type: 'update', patch: { dataset: 'spiral' } });
    expect(kinds(rebuilt)).toBe('scene');
    const reset = host.handle({ id: 14, type: 'reset' });
    expect(kinds(reset)).toBe('scene');
    const snap = host.handle({ id: 15, type: 'snapshot', snapshot: { resolution: 6 } });
    expect(snap).toMatchObject({ type: 'progress' });
  });

  it('survives invalid requests and keeps the previous session', () => {
    const host = new PlaygroundHost();
    host.handle({ id: 1, type: 'init', config: config() });
    expect(host.handle({ id: 2, type: 'update', patch: { lr: -1 } })).toMatchObject({
      type: 'error',
    });
    expect(host.handle({ id: 3, type: 'step', epochs: 2 })).toMatchObject({ type: 'progress' });
    expect(host.handle({ id: 4, type: 'init', config: config({ features: [] }) })).toMatchObject({
      type: 'error',
    });
  });

  it('responses are plain structured-cloneable data', () => {
    const host = new PlaygroundHost();
    const res = host.handle({
      id: 1,
      type: 'init',
      config: config(),
      snapshot: { resolution: 10, neuronMaps: true },
    });
    expect(() => structuredClone(res)).not.toThrow();
  });
});
