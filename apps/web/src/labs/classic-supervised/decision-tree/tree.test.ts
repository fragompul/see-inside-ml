import type { DecisionTree } from '@sim/classic';
import { leafCount } from '@sim/classic';
import { describe, expect, it } from 'vitest';
import { runClassifierExperiment } from '@/shared/lab-kit';
import { treeSpec } from './treeSpec';

const run = (id: string) =>
  runClassifierExperiment(
    treeSpec,
    treeSpec.experiments.find((e) => e.id === id)!,
  );
const leaves = (id: string) => leafCount((run(id).model as DecisionTree).root);

describe('decision tree lab', () => {
  for (const exp of treeSpec.experiments) {
    it(`experiment "${exp.title}" holds`, () => {
      expect(exp.claim(runClassifierExperiment(treeSpec, exp))).toBe(true);
    });
  }

  it('limits and pruning shrink the tree compared with growing it freely', () => {
    expect(leaves('leaf-size')).toBeLessThan(leaves('memorise'));
    expect(leaves('pruned')).toBeLessThan(leaves('memorise'));
  });

  it('pruning improves held-out accuracy over memorising noisy data', () => {
    expect(run('pruned').scores.testAccuracy).toBeGreaterThanOrEqual(
      run('memorise').scores.testAccuracy as number,
    );
  });
});
