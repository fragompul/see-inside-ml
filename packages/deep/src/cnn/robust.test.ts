import { describe, expect, it } from 'vitest';
import { DEFAULT_ROBUST, robustnessCurve, trainRobust } from './robust';

const small = { ...DEFAULT_ROBUST, epochs: 3, perClass: 15 };

describe('adversarial training', () => {
  it('is deterministic for a seed and learns something in a few epochs', () => {
    const a = trainRobust(small);
    const b = trainRobust(small);
    expect(a.accuracy).toEqual(b.accuracy);
    expect(a.accuracy[a.accuracy.length - 1]!).toBeGreaterThan(0.5);
  });

  it('the robustness curve starts at the clean accuracy and an attack never helps on average', () => {
    const run = trainRobust(small);
    const curve = robustnessCurve(run, [0, 0.1, 0.3], 'fgsm');
    expect(curve[0]).toBe(run.accuracy[run.accuracy.length - 1]);
    expect(curve[2]!).toBeLessThanOrEqual(curve[0]!);
  });

  it('training against attacks gives a network that is harder to attack at the budget it saw', () => {
    const ordinary = trainRobust({ ...small, epochs: 6 });
    const robust = trainRobust({ ...small, epochs: 6, epsilon: 0.2, method: 'fgsm' });
    const at = (run: typeof ordinary) => robustnessCurve(run, [0.15], 'fgsm')[0] as number;
    expect(at(robust)).toBeGreaterThan(at(ordinary));
  });
});
