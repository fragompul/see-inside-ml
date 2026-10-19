import { PlaygroundSession } from '@sim/deep';
import { describe, expect, it } from 'vitest';
import { PRESETS, presetConfig } from './presets';

const run = (id: string, epochs: number) => {
  const preset = PRESETS.find((p) => p.id === id);
  if (!preset) throw new Error(`missing preset ${id}`);
  const session = new PlaygroundSession(presetConfig(preset));
  const history = session.step(epochs);
  return history[history.length - 1]!;
};

describe('presets', () => {
  it('have unique ids and valid, complete configurations', () => {
    expect(new Set(PRESETS.map((p) => p.id)).size).toBe(PRESETS.length);
    for (const preset of PRESETS) {
      expect(() => new PlaygroundSession(presetConfig(preset)), preset.id).not.toThrow();
      expect(preset.lesson.length).toBeGreaterThan(40);
    }
  });

  // The lessons make factual claims; make sure they stay true.
  it('"Linear can’t do XOR" stalls near chance', () => {
    expect(run('xor-linear', 300).testMetric).toBeLessThan(0.7);
  });

  it('an engineered x₁·x₂ feature solves XOR with a linear model', () => {
    expect(run('xor-feature', 300).testMetric).toBeGreaterThan(0.95);
  });

  it('a small hidden layer solves XOR on its own', () => {
    expect(run('xor-hidden', 600).testMetric).toBeGreaterThan(0.9);
  });

  it('a sigmoid stack barely learns while its ReLU twin does', () => {
    expect(run('vanishing', 400).trainMetric).toBeLessThan(0.7);
    expect(run('relu-fix', 400).trainMetric).toBeGreaterThan(0.95);
  });

  it('overfitting opens a train/test loss gap that L2 narrows', () => {
    const plain = run('overfit', 1200);
    const l2 = run('regularize', 1200);
    expect(plain.testLoss - plain.trainLoss).toBeGreaterThan(0.2);
    expect(l2.testLoss - l2.trainLoss).toBeLessThan(plain.testLoss - plain.trainLoss);
    expect(l2.testLoss).toBeLessThan(plain.testLoss);
  });
}, 60_000);
