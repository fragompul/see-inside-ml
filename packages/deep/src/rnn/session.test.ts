import { describe, expect, it } from 'vitest';
import { createRng } from '@sim/core';
import { DEFAULT_RNN_LAB, RnnSession, createNet, gradientProfile } from './session';

const make = (patch = {}) => new RnnSession({ ...DEFAULT_RNN_LAB, ...patch });

describe('RnnSession', () => {
  it('trains deterministically for a seed', () => {
    const a = make({ trainSize: 64 });
    const b = make({ trainSize: 64 });
    a.step(2);
    b.step(2);
    expect(a.history).toEqual(b.history);
    expect(a.network.wh.data).toEqual(b.network.wh.data);
  });

  it('batch-wise training equals epoch-wise training', () => {
    const batches = make({ trainSize: 64, batchSize: 16 });
    const epochs = make({ trainSize: 64, batchSize: 16 });
    const done = batches.trainBatches(8); // two epochs of four batches
    epochs.step(2);
    expect(done).toHaveLength(2);
    expect(batches.history).toEqual(epochs.history);
  });

  it('applies learning settings live and rebuilds on structure changes', () => {
    const s = make({ trainSize: 64 });
    s.step(1);
    expect(s.update({ lr: 0.02, clip: 1, optimizer: 'rmsprop', batchSize: 8 })).toBe('live');
    expect(s.history).toHaveLength(1);
    expect(s.update({ hidden: 8 })).toBe('rebuilt');
    expect(s.history).toHaveLength(0);
    expect(s.network.hidden).toBe(8);
    expect(s.update({ cell: 'gru' })).toBe('rebuilt');
    expect(s.update({ seqLen: 7 })).toBe('rebuilt');
  });

  it('switching task keeps the length legal and resizes the network', () => {
    const s = make({ seqLen: 30 });
    expect(s.update({ task: 'parity' })).toBe('rebuilt');
    expect(s.config.seqLen).toBe(24);
    expect(s.network.vocab).toBe(2);
    expect(s.network.classes).toBe(2);
    expect(s.sample(0).tokens).toHaveLength(24);
  });

  it('rejects invalid settings and keeps the previous ones', () => {
    const s = make();
    expect(() => s.update({ lr: 0 })).toThrow();
    expect(() => s.update({ hidden: 1 })).toThrow();
    expect(() => s.update({ seqLen: 99 })).toThrow();
    expect(() => s.update({ batchSize: 2 })).toThrow();
    expect(() => s.update({ clip: -1 })).toThrow();
    expect(() => s.update({ trainSize: 5 })).toThrow();
    expect(s.config.lr).toBe(DEFAULT_RNN_LAB.lr);
  });

  it('confusion counts every test sequence; trace and sample work', () => {
    const s = make({ trainSize: 64 });
    const total = s
      .confusion()
      .flat()
      .reduce((a, b) => a + b, 0);
    expect(total).toBe(s.testCount);
    const { tokens, label } = s.sample(3);
    expect(label).toBeGreaterThanOrEqual(0);
    expect(s.trace(tokens).probs).toHaveLength(4);
    expect(() => s.sample(9999)).toThrow();
  });

  it('inspecting gradients does not disturb training', () => {
    const a = make({ trainSize: 64 });
    const b = make({ trainSize: 64 });
    a.trainBatches(1);
    b.trainBatches(1);
    a.gradientNorms(a.sample(0).tokens);
    a.profile(4);
    a.trainBatches(1);
    b.trainBatches(1);
    expect(a.network.wx.data).toEqual(b.network.wx.data);
  });

  it('clipping caps the update: a tiny clip makes SGD move the weights less', () => {
    const free = make({ trainSize: 64, optimizer: 'sgd', lr: 0.5, clip: 0 });
    const clipped = make({ trainSize: 64, optimizer: 'sgd', lr: 0.5, clip: 0.01 });
    const start = Float64Array.from(free.network.wh.data);
    free.trainBatches(1);
    clipped.trainBatches(1);
    const moved = (w: Float64Array) =>
      w.reduce((s, v, i) => s + Math.abs(v - (start[i] as number)), 0);
    expect(moved(clipped.network.wh.data)).toBeLessThan(moved(free.network.wh.data));
  });
});

describe('what the lab teaches (claims asserted)', () => {
  it('learning signal fades towards the start of a long sequence, far faster in a plain RNN', () => {
    const profile = (cell: 'rnn' | 'lstm') =>
      gradientProfile(createNet(cell, 16, 'first', createRng(5)), 'first', 30, 30, createRng(1));
    const rnn = profile('rnn');
    const lstm = profile('lstm');
    expect((rnn[0] as number) / (rnn[29] as number)).toBeLessThan(1e-5); // it practically vanishes
    expect(lstm[0] as number).toBeGreaterThan(100 * (rnn[0] as number)); // gates keep a path open
    for (let t = 1; t < 30; t++) expect(rnn[t] as number).toBeGreaterThan(rnn[t - 1] as number);
  });

  it('a plain RNN cannot learn to pick the symbol after a marker; GRU and LSTM can', () => {
    const accuracy = (cell: 'rnn' | 'gru' | 'lstm') => {
      const s = make({ task: 'marker', seqLen: 12, cell });
      s.step(25);
      return s.history[24]?.testAccuracy as number;
    };
    expect(accuracy('rnn')).toBeLessThan(0.7);
    expect(accuracy('gru')).toBeGreaterThan(0.95);
    expect(accuracy('lstm')).toBeGreaterThan(0.95);
  }, 60_000);

  it('remembering the first symbol is easy when the sequence is short', () => {
    const s = make({ task: 'first', seqLen: 6, cell: 'rnn' });
    s.step(15);
    expect(s.history[14]?.testAccuracy as number).toBeGreaterThan(0.95);
  }, 30_000);

  it('a trained network is no better on much longer sequences than on the ones it saw', () => {
    const s = make({ task: 'first', seqLen: 8, cell: 'lstm' });
    s.step(20);
    expect(s.accuracyAtLength(8)).toBeGreaterThan(0.95);
    expect(s.accuracyAtLength(8)).toBeGreaterThanOrEqual(s.accuracyAtLength(40));
  }, 30_000);
});
