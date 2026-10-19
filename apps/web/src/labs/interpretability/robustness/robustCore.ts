import {
  DEFAULT_ROBUST,
  IMAGE_SIZE,
  argmax,
  attack,
  robustnessCurve,
  trainRobust,
  type AttackMethod,
  type RobustRun,
} from '@sim/deep';

export interface RobustScenario {
  /** Budget of the attack used during training (0 = ordinary training). */
  trainEpsilon: number;
  trainMethod: AttackMethod;
  /** Attack the trained network is tested with, and the budget of the pictures shown. */
  testMethod: AttackMethod;
  shownEpsilon: number;
}

export const DEFAULT_ROBUSTNESS: RobustScenario = {
  trainEpsilon: 0.2,
  trainMethod: 'fgsm',
  testMethod: 'pgd',
  shownEpsilon: 0.1,
};

export const BUDGETS = [0, 0.05, 0.1, 0.15, 0.2, 0.3];

export interface RobustnessResult {
  scenario: RobustScenario;
  /** Accuracy on the test digits under each attack at each budget, for the network trained as asked and for an ordinary one. */
  chosen: { fgsm: number[]; pgd: number[]; clean: number };
  ordinary: { fgsm: number[]; pgd: number[]; clean: number };
  /** Examples: the digit, the attacked digit at the shown budget, and the answers of the chosen network. */
  examples: {
    original: number[];
    attacked: number[];
    label: number;
    before: number;
    after: number;
  }[];
}

const models = new Map<string, RobustRun>();
const curves = new Map<string, { fgsm: number[]; pgd: number[]; clean: number }>();

function modelFor(epsilon: number, method: AttackMethod): { run: RobustRun; key: string } {
  const key = `${epsilon}|${epsilon === 0 ? 'fgsm' : method}`;
  let run = models.get(key);
  if (!run) {
    run = trainRobust({ ...DEFAULT_ROBUST, epsilon, method: epsilon === 0 ? 'fgsm' : method });
    models.set(key, run);
  }
  return { run, key };
}

function curveFor(epsilon: number, method: AttackMethod) {
  const { run, key } = modelFor(epsilon, method);
  let c = curves.get(key);
  if (!c) {
    const fgsm = robustnessCurve(run, BUDGETS, 'fgsm');
    const pgd = robustnessCurve(run, BUDGETS, 'pgd');
    c = { fgsm, pgd, clean: fgsm[0] as number };
    curves.set(key, c);
  }
  return c;
}

export function runRobustness(s: RobustScenario): RobustnessResult {
  const { run } = modelFor(s.trainEpsilon, s.trainMethod);
  const px = IMAGE_SIZE * IMAGE_SIZE;
  const examples = [3, 40, 77, 120, 160, 200].map((i) => {
    const original = run.test.images.slice(i * px, (i + 1) * px);
    const label = run.test.labels[i] as number;
    const result = attack(run.net, original, label, {
      method: s.testMethod,
      epsilon: s.shownEpsilon,
      steps: 10,
    });
    return {
      original: Array.from(original),
      attacked: Array.from(result.adversarial),
      label,
      before: argmax(run.net.predict(original)),
      after: result.after.label,
    };
  });
  return {
    scenario: s,
    chosen: curveFor(s.trainEpsilon, s.trainMethod),
    ordinary: curveFor(0, 'fgsm'),
    examples,
  };
}
