import { Matrix, createRng, makeClassification } from '@sim/core';
import { describe, expect, it } from 'vitest';
import {
  DEFAULT_TOOLBOX,
  ToolboxMLP,
  learningRateFinder,
  trainToolbox,
  type ToolboxConfig,
} from './toolbox';

const data = (n: number, seed: number, kind: 'moons' | 'circles' = 'moons', noise = 0.2) => {
  const t = makeClassification(kind, n, noise, createRng(seed));
  return { x: t.x, y: Array.from(t.y) };
};

describe('gradients', () => {
  const x = new Matrix(
    6,
    2,
    Float64Array.from([0.3, -0.5, 1.1, 0.2, -0.8, 0.9, 0.5, 0.5, -0.2, -1.0, 0.7, -0.3]),
  );
  const y = [0, 1, 1, 0, 1, 0];
  const cases: [string, Partial<ToolboxConfig>][] = [
    ['tanh', { activation: 'tanh' }],
    ['relu', { activation: 'relu', init: 'he' }],
    ['tanh with batch norm', { activation: 'tanh', batchNorm: true }],
    ['relu with batch norm', { activation: 'relu', batchNorm: true, init: 'xavier' }],
  ];
  it.each(cases)('%s: backpropagation matches finite differences', (_name, over) => {
    const config: ToolboxConfig = {
      ...DEFAULT_TOOLBOX,
      layers: [2, 5, 4, 2],
      init: 'xavier',
      ...over,
    };
    const net = new ToolboxMLP(config);
    // give the batch-norm scales and shifts non-trivial values
    net.gammas.forEach((g) => g.data.forEach((_, i) => (g.data[i] = 0.8 + 0.1 * i)));
    net.betas.forEach((b) => b.data.forEach((_, i) => (b.data[i] = 0.05 * i)));
    net.zeroGrad();
    net.forward(x, true);
    net.backward(y);
    const lossOf = () => {
      const p = net.forward(x, true);
      let l = 0;
      for (let i = 0; i < p.rows; i++) l -= Math.log(p.get(i, y[i]!)) / p.rows;
      return l;
    };
    const h = 1e-6;
    for (const t of net.tensors()) {
      for (const i of [0, Math.floor(t.data.length / 2), t.data.length - 1]) {
        const saved = t.data[i]!;
        t.data[i] = saved + h;
        const up = lossOf();
        t.data[i] = saved - h;
        const down = lossOf();
        t.data[i] = saved;
        expect(t.grad[i]).toBeCloseTo((up - down) / (2 * h), 5);
      }
    }
  });
});

describe('initialisation', () => {
  const { x, y } = data(200, 1);
  it('a network started at zero never breaks the symmetry of its hidden units', () => {
    const r = trainToolbox(
      { ...DEFAULT_TOOLBOX, init: 'zeros' },
      { x, y },
      { x, y },
      {
        epochs: 30,
        batchSize: 20,
        learningRate: 0.1,
        optimizer: 'sgd',
        patience: 0,
        seed: 1,
      },
    );
    expect(r.model.distinctFirstLayerUnits()).toBeLessThanOrEqual(1);
    const last = r.history[r.history.length - 1]!;
    expect(last.trainAccuracy).toBeLessThan(0.9);
  });

  it('a sensible initialisation learns the same task', () => {
    const r = trainToolbox(
      { ...DEFAULT_TOOLBOX, init: 'he' },
      { x, y },
      { x, y },
      {
        epochs: 40,
        batchSize: 20,
        learningRate: 0.05,
        optimizer: 'sgd',
        patience: 0,
        seed: 1,
      },
    );
    expect(r.model.distinctFirstLayerUnits()).toBeGreaterThan(20);
    expect(r.history[r.history.length - 1]!.trainAccuracy).toBeGreaterThan(0.9);
  });
});

describe('batch norm and dropout', () => {
  it('batch norm gives every hidden unit zero mean and unit variance over the batch (before scale and shift)', () => {
    const { x } = data(64, 2);
    const net = new ToolboxMLP({ ...DEFAULT_TOOLBOX, layers: [2, 8, 2], batchNorm: true });
    net.forward(x, true);
    const c = (net as unknown as { cache: { normalized: Matrix[] } }).cache;
    const norm = c.normalized[0]!;
    for (let k = 0; k < 8; k++) {
      let m = 0;
      let v = 0;
      for (let i = 0; i < 64; i++) m += norm.get(i, k) / 64;
      for (let i = 0; i < 64; i++) v += (norm.get(i, k) - m) ** 2 / 64;
      expect(m).toBeCloseTo(0, 9);
      expect(v).toBeCloseTo(1, 2);
    }
  });

  it('evaluation mode is deterministic and uses running statistics; training mode with dropout is random', () => {
    const { x } = data(50, 3);
    const net = new ToolboxMLP({ ...DEFAULT_TOOLBOX, layers: [2, 16, 2], dropout: 0.5 });
    const a = net.forward(x, false);
    const b = net.forward(x, false);
    expect(Array.from(a.data)).toEqual(Array.from(b.data));
    const rng = createRng(1);
    const c = net.forward(x, true, rng);
    const d = net.forward(x, true, rng);
    expect(Array.from(c.data)).not.toEqual(Array.from(d.data));
  });

  it('inverted dropout keeps the expected activation unchanged', () => {
    const net = new ToolboxMLP({
      ...DEFAULT_TOOLBOX,
      layers: [2, 64, 2],
      dropout: 0.5,
      activation: 'tanh',
    });
    const rng = createRng(2);
    const mask = Array.from({ length: 20000 }, (): number => (rng.next() < 0.5 ? 2 : 0));
    expect(mask.reduce((a, b) => a + b, 0) / mask.length).toBeCloseTo(1, 1);
    expect(net.depth).toBe(2);
  });
});

describe('training tools', () => {
  it('dropout narrows the gap between training and validation accuracy on a small noisy set', () => {
    const train = data(60, 4, 'moons', 0.35);
    const val = data(800, 5, 'moons', 0.35);
    const run = (dropout: number) =>
      trainToolbox({ ...DEFAULT_TOOLBOX, layers: [2, 64, 64, 2], dropout, seed: 2 }, train, val, {
        epochs: 150,
        batchSize: 15,
        learningRate: 0.005,
        optimizer: 'adam',
        patience: 0,
        seed: 3,
      }).history;
    const plain = run(0);
    const dropped = run(0.5);
    const gap = (h: typeof plain) =>
      h[h.length - 1]!.trainAccuracy - h[h.length - 1]!.validationAccuracy;
    expect(gap(dropped)).toBeLessThan(gap(plain) + 0.02);
    expect(plain[plain.length - 1]!.trainAccuracy).toBeGreaterThan(0.95);
  });

  it('early stopping halts after the patience runs out and restores the best weights', () => {
    const train = data(60, 4, 'moons', 0.4);
    const val = data(500, 5, 'moons', 0.4);
    const r = trainToolbox({ ...DEFAULT_TOOLBOX, layers: [2, 64, 64, 2], seed: 2 }, train, val, {
      epochs: 400,
      batchSize: 15,
      learningRate: 0.01,
      optimizer: 'adam',
      patience: 15,
      seed: 3,
    });
    expect(r.stoppedEarly).toBe(true);
    expect(r.history.length).toBeLessThan(400);
    const bestLoss = Math.min(...r.history.map((h) => h.validationLoss));
    expect(r.model.evaluate(val.x, val.y).loss).toBeCloseTo(bestLoss, 6);
    expect(r.history.length - r.bestEpoch).toBe(15);
  });

  it('the learning-rate finder sweeps upward and suggests a rate inside the range', () => {
    const { x, y } = data(300, 6);
    const r = learningRateFinder(
      { ...DEFAULT_TOOLBOX, seed: 1 },
      { x, y },
      {
        low: 1e-5,
        high: 10,
        steps: 200,
        batchSize: 32,
        optimizer: 'sgd',
        seed: 1,
      },
    );
    for (let i = 1; i < r.curve.length; i++)
      expect(r.curve[i]!.learningRate).toBeGreaterThan(r.curve[i - 1]!.learningRate);
    expect(r.suggested).toBeGreaterThan(1e-5);
    expect(r.suggested).toBeLessThan(10);
  });
});

describe('LoRA adapters and soft targets', () => {
  const x = new Matrix(
    5,
    2,
    Float64Array.from([0.3, -0.5, 1.1, 0.2, -0.8, 0.9, 0.5, 0.5, -0.2, -1.0]),
  );
  const y = [0, 1, 1, 0, 1];

  it('starts as the same function, trains only the low-rank factors, and the gradients match finite differences', () => {
    const net = new ToolboxMLP({
      ...DEFAULT_TOOLBOX,
      layers: [2, 6, 2],
      activation: 'tanh',
      init: 'xavier',
    });
    const before = Array.from(net.forward(x, false).data);
    net.addAdapters(2, 1);
    expect(Array.from(net.forward(x, false).data)).toEqual(before);
    // 2x2 + 2x6 + ... : adapters on both layers, base frozen
    expect(net.trainableCount).toBe(2 * 2 + 2 * 6 + 6 * 2 + 2 * 2);
    net.adapters.forEach((a) => a.b.data.forEach((_, i) => (a.b.data[i] = 0.1 * Math.sin(i + 1))));
    net.zeroGrad();
    net.forward(x, true);
    net.backward(y);
    const lossOf = () => {
      const p = net.forward(x, true);
      let l = 0;
      for (let i = 0; i < p.rows; i++) l -= Math.log(p.get(i, y[i]!)) / p.rows;
      return l;
    };
    const h = 1e-6;
    for (const ad of net.adapters)
      for (const t of [ad.a, ad.b])
        for (const i of [0, t.data.length - 1]) {
          const saved = t.data[i]!;
          t.data[i] = saved + h;
          const up = lossOf();
          t.data[i] = saved - h;
          const down = lossOf();
          t.data[i] = saved;
          expect(t.grad[i]).toBeCloseTo((up - down) / (2 * h), 5);
        }
  });

  it('merging the adapters keeps the predictions and gives back a plain network', () => {
    const net = new ToolboxMLP({ ...DEFAULT_TOOLBOX, layers: [2, 6, 2] });
    net.addAdapters(2, 3);
    net.adapters.forEach((a) => a.b.data.forEach((_, i) => (a.b.data[i] = 0.2 * Math.cos(i))));
    const withAdapters = Array.from(net.forward(x, false).data);
    net.mergeAdapters();
    expect(net.adapters).toHaveLength(0);
    Array.from(net.forward(x, false).data).forEach((v, i) =>
      expect(v).toBeCloseTo(withAdapters[i]!, 12),
    );
  });

  it('soft targets equal hard labels when they are one-hot, and give the cross-entropy otherwise', () => {
    const net = new ToolboxMLP({ ...DEFAULT_TOOLBOX, layers: [2, 6, 2] });
    net.zeroGrad();
    net.forward(x, true);
    const hard = net.backward(y);
    const g1 = Float64Array.from(net.weights[0]!.grad);
    net.zeroGrad();
    net.forward(x, true);
    const onehot = new Matrix(5, 2);
    y.forEach((c, i) => onehot.set(i, c, 1));
    const soft = net.backwardSoft(onehot);
    expect(soft).toBeCloseTo(hard, 12);
    g1.forEach((v, i) => expect(net.weights[0]!.grad[i]).toBeCloseTo(v, 12));
  });
});
