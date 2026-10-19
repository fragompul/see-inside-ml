import { describe, expect, it } from 'vitest';
import { PlaygroundHost } from './host';
import { DEFAULT_CONFIG, PlaygroundSession, type PlaygroundConfig } from './session';

const config = (patch: Partial<PlaygroundConfig> = {}): PlaygroundConfig => ({
  ...DEFAULT_CONFIG,
  dataset: 'gaussian',
  samples: 60,
  hidden: [{ neurons: 4, activation: 'tanh' }],
  ...patch,
});

describe('editing the training points', () => {
  it('adds, moves and erases points without touching the weights or the test set', () => {
    const s = new PlaygroundSession(config());
    s.step(5);
    const weights = s.snapshot().weights;
    const before = s.scene();
    s.editPoints([{ op: 'add', x: 1.5, y: -2, label: 1 }]);
    const added = s.scene();
    expect(added.train.target.length).toBe(before.train.target.length + 1);
    expect(added.train.xy[added.train.xy.length - 2]).toBeCloseTo(1.5, 6);
    expect(added.train.target[added.train.target.length - 1]).toBe(1);
    expect(Array.from(added.test.xy)).toEqual(Array.from(before.test.xy));
    expect(s.snapshot().weights).toEqual(weights);

    s.editPoints([{ op: 'move', index: 0, x: -3, y: 4 }]);
    const moved = s.scene();
    expect(moved.train.xy[0]).toBeCloseTo(-3, 6);
    expect(moved.train.xy[1]).toBeCloseTo(4, 6);
    expect(moved.train.target[0]).toBe(added.train.target[0]);

    s.editPoints([{ op: 'erase', index: 0 }]);
    const erased = s.scene();
    expect(erased.train.target.length).toBe(added.train.target.length - 1);
    expect(s.edits).toBe(3);
    expect(s.trainSize).toBe(erased.train.target.length);
  });

  it('clamps coordinates to the plane and rejects impossible edits', () => {
    const s = new PlaygroundSession(config());
    s.editPoints([{ op: 'add', x: 99, y: -99, label: 0 }]);
    const sc = s.scene();
    expect(sc.train.xy[sc.train.xy.length - 2]).toBe(sc.domain);
    expect(sc.train.xy[sc.train.xy.length - 1]).toBe(-sc.domain);
    expect(() => s.editPoints([{ op: 'add', x: 0, y: 0, label: 9 }])).toThrow(/Class/);
    expect(() => s.editPoints([{ op: 'move', index: 9999, x: 0, y: 0 }])).toThrow(/Training point/);
    expect(() => s.editPoints([{ op: 'erase', index: -1 }])).toThrow(/Training point/);
    const reg = new PlaygroundSession(config({ dataset: 'plane' }));
    expect(() => reg.editPoints([{ op: 'add', x: 0, y: 0, label: 0 }])).toThrow(/classification/);
  });

  it('keeps at least two training points', () => {
    const s = new PlaygroundSession(config({ samples: 60, trainRatio: 10 }));
    for (let i = 0; i < 100 && s.trainSize > 2; i++) s.editPoints([{ op: 'erase', index: 0 }]);
    expect(s.trainSize).toBe(2);
    expect(() => s.editPoints([{ op: 'erase', index: 0 }])).toThrow(/at least two/);
  });

  it('training uses the edited points: a point added deep inside the other class gets learned', () => {
    const s = new PlaygroundSession(
      config({
        dataset: 'xor',
        samples: 80,
        noise: 0,
        hidden: [{ neurons: 8, activation: 'tanh' }],
        lr: 0.05,
      }),
    );
    s.step(150);
    // plant a point of class 1 at the centre of a class-0 quadrant and watch the loss jump, then fall
    const sc = s.scene();
    let probe = { x: 3, y: 3, label: 1 };
    for (let i = 0; i < sc.train.target.length; i++) {
      const x = sc.train.xy[2 * i] as number;
      const y = sc.train.xy[2 * i + 1] as number;
      if (x > 2 && y > 2) probe = { x: 3, y: 3, label: 1 - (sc.train.target[i] as number) };
    }
    const lossBefore = s.snapshot().stats.trainLoss;
    s.editPoints(Array.from({ length: 6 }, () => ({ op: 'add' as const, ...probe })));
    const lossAfterEdit = s.snapshot().stats.trainLoss;
    expect(lossAfterEdit).toBeGreaterThan(lossBefore);
    s.step(300);
    expect(s.snapshot().stats.trainLoss).toBeLessThan(lossAfterEdit);
  });

  it('edits survive a reset of the weights, and restoring regenerates the original data', () => {
    const s = new PlaygroundSession(config());
    const original = s.scene().train.target.length;
    s.editPoints([{ op: 'add', x: 0, y: 0, label: 0 }]);
    s.reset();
    expect(s.scene().train.target.length).toBe(original + 1);
    s.restoreData();
    expect(s.scene().train.target.length).toBe(original);
    expect(s.edits).toBe(0);
  });

  it('restoring the data keeps the weights and the epoch counter', () => {
    const s = new PlaygroundSession(config());
    s.step(7);
    s.editPoints([{ op: 'add', x: 0, y: 0, label: 1 }]);
    const weights = s.snapshot().weights;
    s.restoreData();
    expect(s.epoch).toBe(7);
    expect(s.snapshot().weights).toEqual(weights);
    expect(s.scene().train.target.length).toBe(
      new PlaygroundSession(config()).scene().train.target.length,
    );
  });

  it('a change of dataset discards the edits (the data are regenerated)', () => {
    const s = new PlaygroundSession(config());
    s.editPoints([{ op: 'add', x: 0, y: 0, label: 0 }]);
    s.update({ dataset: 'moons' });
    expect(s.edits).toBe(0);
  });
});

describe('the host protocol for edits', () => {
  it('answers with a scene that says the training history stays', () => {
    const host = new PlaygroundHost();
    host.handle({ id: 1, type: 'init', config: config() });
    host.handle({ id: 2, type: 'step', epochs: 3 });
    const res = host.handle({
      id: 3,
      type: 'edit',
      edits: [{ op: 'add', x: 1, y: 1, label: 0 }],
      snapshot: { resolution: 8 },
    });
    expect(res.type).toBe('scene');
    if (res.type === 'scene') {
      expect(res.keepHistory).toBe(true);
      expect(res.snapshot.boundary?.resolution).toBe(8);
      expect(res.snapshot.stats.epoch).toBe(3);
    }
    const bad = host.handle({ id: 4, type: 'edit', edits: [{ op: 'erase', index: 99999 }] });
    expect(bad.type).toBe('error');
    const restored = host.handle({ id: 5, type: 'edit', edits: [], restore: true });
    expect(restored.type).toBe('scene');
  });
});
