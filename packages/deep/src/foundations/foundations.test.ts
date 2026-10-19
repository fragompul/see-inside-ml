import { describe, expect, it } from 'vitest';
import { DEFAULT_DESCENT, DESCENT_LIMITS, Descent, validateDescent } from './descent';
import {
  NEURON_ACTIVATIONS,
  NEURON_DATASETS,
  bestLine,
  lossLandscape,
  neuronAccuracy,
  neuronActivate,
  neuronData,
  neuronLoss,
  weightedSum,
} from './neuron';

describe('the neuron', () => {
  it('computes the weighted sum and the four activations', () => {
    expect(weightedSum({ w1: 2, w2: -1, b: 0.5 }, 3, 4)).toBe(2.5);
    expect(neuronActivate('step', 0.001)).toBe(1);
    expect(neuronActivate('step', 0)).toBe(0);
    expect(neuronActivate('sigmoid', 0)).toBe(0.5);
    expect(neuronActivate('sigmoid', 40)).toBeCloseTo(1, 12);
    expect(neuronActivate('tanh', 0)).toBe(0);
    expect(neuronActivate('tanh', 100)).toBeCloseTo(1, 12);
    expect(neuronActivate('relu', -3)).toBe(0);
    expect(neuronActivate('relu', 2.5)).toBe(2.5);
    expect(NEURON_ACTIVATIONS).toHaveLength(4);
  });

  it('a neuron with the right weights implements AND and OR; none implements XOR', () => {
    const and = neuronData('and');
    const or = neuronData('or');
    const xor = neuronData('xor');
    expect(neuronAccuracy(and, { w1: 1, w2: 1, b: -1.5 })).toBe(1);
    expect(neuronAccuracy(or, { w1: 1, w2: 1, b: -0.5 })).toBe(1);
    expect(neuronAccuracy(xor, { w1: 1, w2: 1, b: -0.5 })).toBeLessThan(1);
  });

  it('the datasets are well formed and fixed', () => {
    for (const name of NEURON_DATASETS) {
      const d = neuronData(name);
      expect(d.x).toHaveLength(d.count * 2);
      expect(d.y).toHaveLength(d.count);
      expect(new Set(d.y)).toEqual(new Set([0, 1]));
      expect(neuronData(name).x).toEqual(d.x);
    }
    expect(neuronData('xor').y).toEqual(Uint8Array.from([0, 1, 1, 0]));
  });
});

describe('the best a single line can do', () => {
  it('reaches 100% on every dataset except XOR, where 75% is the limit', () => {
    expect(bestLine(neuronData('and')).accuracy).toBe(1);
    expect(bestLine(neuronData('or')).accuracy).toBe(1);
    expect(bestLine(neuronData('blobs')).accuracy).toBe(1);
    expect(bestLine(neuronData('xor')).accuracy).toBe(0.75);
  });

  it('returns weights that really reach the accuracy it reports', () => {
    for (const name of NEURON_DATASETS) {
      const data = neuronData(name);
      const { accuracy, params } = bestLine(data);
      expect(neuronAccuracy(data, params)).toBe(accuracy);
    }
  });
});

describe('the loss and its gradient', () => {
  it('match finite differences', () => {
    for (const name of NEURON_DATASETS) {
      const data = neuronData(name);
      const p = { w1: 0.7, w2: -1.3, b: 0.4 };
      const { grad } = neuronLoss(data, p);
      const h = 1e-6;
      for (const key of ['w1', 'w2', 'b'] as const) {
        const up = neuronLoss(data, { ...p, [key]: p[key] + h }).loss;
        const down = neuronLoss(data, { ...p, [key]: p[key] - h }).loss;
        expect(grad[key]).toBeCloseTo((up - down) / (2 * h), 6);
      }
    }
  });

  it('is ln 2 for a neuron that knows nothing, and XOR cannot be pushed below it', () => {
    const xor = neuronData('xor');
    expect(neuronLoss(xor, { w1: 0, w2: 0, b: 0 }).loss).toBeCloseTo(Math.LN2, 12);
    let min = Infinity;
    for (const b of [-6, -3, -1, 0, 1, 3, 6])
      for (const v of lossLandscape(xor, b, 8, 32)) min = Math.min(min, v);
    expect(min).toBeGreaterThan(Math.LN2 - 0.01);
    // while a separable problem goes far below it
    let easy = Infinity;
    for (const b of [-3, -1, 0])
      for (const v of lossLandscape(neuronData('or'), b, 8, 32)) easy = Math.min(easy, v);
    expect(easy).toBeLessThan(0.1);
  });

  it('the landscape is laid out row by row from the top with w2 decreasing', () => {
    const data = neuronData('blobs');
    const grid = lossLandscape(data, -3, 8, 4);
    expect(grid).toHaveLength(16);
    // top-left cell centre = (-6, 6), bottom-right = (6, -6)
    expect(grid[0]).toBeCloseTo(neuronLoss(data, { w1: -6, w2: 6, b: -3 }).loss, 12);
    expect(grid[15]).toBeCloseTo(neuronLoss(data, { w1: 6, w2: -6, b: -3 }).loss, 12);
  });
});

describe('gradient descent on the landscape', () => {
  const stepsToReach = (d: Descent, target: number, max = 400) => {
    for (let i = 1; i <= max; i++) if (d.step().loss < target) return i;
    return Infinity;
  };

  it('rolls downhill: the loss falls monotonically for plain SGD with a moderate rate', () => {
    const d = new Descent({ ...DEFAULT_DESCENT, lr: 0.5 });
    let previous = d.here().loss;
    for (let i = 0; i < 100; i++) {
      const loss = d.step().loss;
      expect(loss).toBeLessThanOrEqual(previous + 1e-12);
      previous = loss;
    }
    expect(previous).toBeLessThan(0.3);
    expect(d.path).toHaveLength(101);
  });

  it('a tiny learning rate crawls and a bigger one arrives quickly', () => {
    expect(stepsToReach(new Descent({ ...DEFAULT_DESCENT, lr: 0.01 }), 0.3)).toBe(Infinity);
    expect(stepsToReach(new Descent({ ...DEFAULT_DESCENT, lr: 0.5 }), 0.3)).toBeLessThan(40);
  });

  it('momentum arrives much sooner than plain SGD at the same learning rate', () => {
    const plain = stepsToReach(new Descent({ ...DEFAULT_DESCENT, lr: 0.1 }), 0.3);
    const momentum = stepsToReach(
      new Descent({ ...DEFAULT_DESCENT, lr: 0.1, optimizer: 'momentum' }),
      0.3,
    );
    expect(plain).toBeGreaterThan(80);
    expect(momentum).toBeLessThan(30);
  });

  it('on XOR the ball settles on the plateau at ln 2 and stays there', () => {
    const d = new Descent({ ...DEFAULT_DESCENT, dataset: 'xor', lr: 0.5, bias: 0 });
    for (let i = 0; i < 400; i++) d.step();
    expect(d.here().loss).toBeGreaterThan(Math.LN2 - 0.01);
    expect(d.gradientNorm()).toBeLessThan(0.01);
  });

  it('can be placed anywhere (clamped to the window), and changes of the surface reset the path', () => {
    const d = new Descent(DEFAULT_DESCENT);
    d.step();
    d.step();
    d.place(100, -100);
    expect(d.position.w1).toBe(DESCENT_LIMITS.range);
    expect(d.position.w2).toBe(-DESCENT_LIMITS.range);
    expect(d.path).toHaveLength(1);
    d.step();
    d.update({ lr: 1 }); // the learning rate does not move the surface
    expect(d.path).toHaveLength(2);
    d.update({ bias: -1 });
    expect(d.path).toHaveLength(1);
    d.step();
    d.update({ dataset: 'or' });
    expect(d.path).toHaveLength(1);
    d.step();
    d.update({ optimizer: 'adam' });
    expect(d.path).toHaveLength(1);
  });

  it('does not run away to infinity even with an absurd learning rate, and validates its settings', () => {
    const d = new Descent({ ...DEFAULT_DESCENT, lr: 20 });
    for (let i = 0; i < 200; i++) d.step();
    expect(Number.isFinite(d.here().loss)).toBe(true);
    expect(Math.abs(d.position.w1)).toBeLessThanOrEqual(DESCENT_LIMITS.range * 4);
    expect(() => validateDescent({ ...DEFAULT_DESCENT, lr: 0 })).toThrow(RangeError);
    expect(() => validateDescent({ ...DEFAULT_DESCENT, bias: 99 })).toThrow(RangeError);
    expect(() => validateDescent({ ...DEFAULT_DESCENT, optimizer: 'x' as 'sgd' })).toThrow(
      RangeError,
    );
    expect(() => validateDescent({ ...DEFAULT_DESCENT, dataset: 'x' as 'and' })).toThrow(
      RangeError,
    );
    expect(() => new Descent(DEFAULT_DESCENT).update({ lr: -1 })).toThrow(RangeError);
  });
});
