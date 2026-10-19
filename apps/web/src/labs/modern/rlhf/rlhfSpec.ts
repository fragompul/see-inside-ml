import type { ParamSpec } from '@/shared/lab-kit';
import { BEST_OF, runRlhf, type RlhfResult, type RlhfScenario } from './rlhfCore';

export const rlhfParams: ParamSpec[] = [
  {
    key: 'pairs',
    label: 'Preference pairs the reward model learns from',
    kind: 'ladder',
    values: [20, 100, 600],
    format: (v) => String(v),
  },
  {
    key: 'noise',
    label: 'Unreliability of the human choices',
    kind: 'ladder',
    values: [0.3, 0.8, 2],
    format: (v) => String(v),
    hint: 'Higher means people disagree with the true quality more often.',
  },
];

export interface RlhfExperiment {
  id: string;
  title: string;
  text: string;
  scenario: Partial<RlhfScenario>;
  claim: (run: RlhfResult) => boolean;
}

const sibling = (r: RlhfResult, change: Partial<RlhfScenario>) =>
  runRlhf({ ...r.scenario, ...change });
const increasing = (v: number[]) => v.every((x, i) => i === 0 || x > (v[i - 1] as number) - 1e-12);

export const rlhfExperiments: RlhfExperiment[] = [
  {
    id: 'data',
    title: 'More feedback, a better reward model',
    text: 'Fitted to 20 preference pairs the reward model agrees with new human choices 59% of the time; with 100 pairs 68%, with 600 pairs 70%. It never reaches the reliability of the people, whose choices are themselves noisy.',
    scenario: { pairs: 20 },
    claim: (r) => {
      const hundred = sibling(r, { pairs: 100 });
      const many = sibling(r, { pairs: 600 });
      return hundred.accuracy > r.accuracy + 0.05 && many.accuracy >= hundred.accuracy - 0.005;
    },
  },
  {
    id: 'hacking',
    title: 'Optimising the reward model too hard backfires',
    text: 'With a reward model learned from 20 pairs, the policy that follows it ever more closely keeps raising the reward model’s score. The true reward rises at first, peaks, and then falls to well below where the original policy started (−1.5 against −0.4): the policy has found responses the reward model is wrong about.',
    scenario: { pairs: 20 },
    claim: (r) =>
      increasing(r.proxy) &&
      r.fall > 1.5 &&
      r.peakTruth > 0.3 &&
      (r.truth[r.truth.length - 1] as number) < r.baseline - 0.5,
  },
  {
    id: 'penalty',
    title: 'A penalty for moving away keeps the gain',
    text: 'The best true reward, 0.5, is reached when the policy has moved a moderate distance from the original (KL 3). Beyond that, extra distance only exploits the reward model. Choosing how strong the penalty is means choosing where on this curve to stop.',
    scenario: { pairs: 20 },
    claim: (r) =>
      r.peakTruth - r.baseline > 0.8 &&
      r.peakKl < (r.kl[r.kl.length - 1] as number) - 2 &&
      r.peakKl > 1,
  },
  {
    id: 'more-feedback-safer',
    title: 'Better feedback pushes the failure away',
    text: 'With 100 preference pairs the true reward keeps rising all the way to the sharpest policy (3.3), with no fall at all, and 600 pairs do slightly better. Reward hacking is less a property of the method than of how wrong the reward model is.',
    scenario: { pairs: 100 },
    claim: (r) =>
      r.fall < 0.05 &&
      (r.truth[r.truth.length - 1] as number) > 3 &&
      sibling(r, { pairs: 600 }).peakTruth > r.peakTruth,
  },
  {
    id: 'noisy-people',
    title: 'Unreliable people, the same trouble',
    text: 'If the human choices are very noisy (100 pairs, but an unreliability of 2) the reward model is little better than with 20 clean pairs and the policy hacks it again: the true reward peaks at 1.35 and falls to 0.4.',
    scenario: { pairs: 100, noise: 2 },
    claim: (r) => r.fall > 0.5 && r.peakTruth < sibling(r, { noise: 0.8 }).peakTruth,
  },
  {
    id: 'best-of-n',
    title: 'Best-of-n has the same problem',
    text: 'Instead of training, draw n responses and keep the one the reward model likes best. The true reward of the chosen response rises up to a few hundred draws and then falls as the search starts finding the reward model’s mistakes: with 20 pairs the pool of 3,000 is worse to search completely than a few hundred.',
    scenario: { pairs: 20 },
    claim: (r) => {
      const t = r.bestOfTruth;
      const peak = Math.max(...t);
      const last = t[t.length - 1] as number;
      return (
        peak > 0.3 &&
        t.indexOf(peak) > 4 &&
        t.indexOf(peak) < BEST_OF.length - 1 &&
        last < peak - 0.5
      );
    },
  },
];
