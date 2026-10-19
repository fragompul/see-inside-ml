import type { ParamSpec } from '@/shared/lab-kit';
import { TARGETS, type McmcRun, type McmcScenario, type TargetId } from './mcmcCore';

export const mcmcParams: ParamSpec[] = [
  {
    key: 'target',
    label: 'Distribution to explore',
    kind: 'segmented',
    options: (Object.keys(TARGETS) as TargetId[]).map((id) => ({
      value: id,
      label: TARGETS[id].label,
    })),
  },
  {
    key: 'sampler',
    label: 'Sampler',
    kind: 'segmented',
    options: [
      { value: 'metropolis', label: 'Random-walk Metropolis' },
      { value: 'gibbs', label: 'Gibbs' },
      { value: 'hmc', label: 'Hamiltonian' },
    ],
    hint: 'Gibbs needs the exact conditionals, which are only at hand for the Gaussian.',
  },
  {
    key: 'proposalSd',
    label: 'Jump size',
    kind: 'ladder',
    values: [0.05, 0.1, 0.3, 0.5, 0.8, 1.5, 3, 6],
    format: (v) => String(v),
    showIf: (v) => v.sampler === 'metropolis',
    hint: 'Standard deviation of the random step.',
  },
  {
    key: 'stepSize',
    label: 'Leapfrog step size',
    kind: 'ladder',
    values: [0.01, 0.05, 0.1, 0.15, 0.2, 0.4, 0.5],
    format: (v) => String(v),
    showIf: (v) => v.sampler === 'hmc',
  },
  {
    key: 'leapfrog',
    label: 'Leapfrog steps',
    kind: 'slider',
    min: 1,
    max: 40,
    step: 1,
    showIf: (v) => v.sampler === 'hmc',
    hint: 'Length of each simulated trajectory.',
  },
  {
    key: 'steps',
    label: 'Steps per chain',
    kind: 'ladder',
    values: [100, 300, 1000, 1500, 3000, 5000],
    format: (v) => String(v),
  },
  {
    key: 'burnIn',
    label: 'Burn-in discarded',
    kind: 'slider',
    min: 0,
    max: 1000,
    step: 50,
    hint: 'The first steps remember the starting point, not the distribution.',
  },
];

export interface McmcExperiment {
  id: string;
  title: string;
  text: string;
  scenario: Partial<McmcScenario>;
  claim: (run: McmcRun) => boolean;
}

export const mcmcExperiments: McmcExperiment[] = [
  {
    id: 'tuned',
    title: 'A well-tuned walk',
    text: 'Jumps of the right size are accepted about a third of the time: big enough to travel, small enough to be accepted. The four chains agree (R-hat near 1) and the samples are worth many times more than with the default step.',
    scenario: { proposalSd: 0.8 },
    claim: (r) => r.acceptance > 0.2 && r.acceptance < 0.4 && r.rHat < 1.05 && r.ess > 100,
  },
  {
    id: 'timid',
    title: 'Timid steps',
    text: 'Tiny jumps are almost always accepted but go nowhere: successive samples are nearly identical, the four chains never meet (R-hat far above 1) and the estimate of the mean is poor.',
    scenario: { proposalSd: 0.05 },
    claim: (r) => r.acceptance > 0.85 && r.rHat > 2,
  },
  {
    id: 'reckless',
    title: 'Reckless steps',
    text: 'Huge jumps land far outside the ridge almost every time and are rejected: the chain sits still for long stretches.',
    scenario: { proposalSd: 5 },
    claim: (r) => r.acceptance < 0.05,
  },
  {
    id: 'ridge',
    title: 'Walking a narrow ridge',
    text: 'On the correlated ridge a random walk with a fixed step zig-zags across it. After thousands of steps the chains are worth only a few dozen independent samples.',
    scenario: {},
    claim: (r) => r.ess < 100 && r.ess / r.kept < 0.02,
  },
  {
    id: 'gibbs',
    title: 'Gibbs never rejects',
    text: 'Gibbs sampling draws each coordinate from its exact conditional, so every move is accepted, but it can only move along the axes, so it also crawls along the diagonal ridge.',
    scenario: { sampler: 'gibbs' },
    claim: (r) => r.acceptance === 1 && r.ess > 150 && r.ess < 500 && r.rHat < 1.05,
  },
  {
    id: 'hmc',
    title: 'Momentum helps',
    text: 'Hamiltonian Monte Carlo rolls a ball along the surface of minus log p, using the gradient to follow the ridge. Nearly every draw is an independent sample: dozens of times more efficient than the random walk.',
    scenario: { sampler: 'hmc' },
    claim: (r) => r.ess > 2000 && r.rHat < 1.05 && r.meanError < 0.1,
  },
  {
    id: 'hmc-unstable',
    title: 'Steps that are too long',
    text: 'If the leapfrog step is too large the simulated ball does not conserve energy; the proposal is far worse than where it started and is always rejected.',
    scenario: { sampler: 'hmc', stepSize: 0.5 },
    claim: (r) => r.acceptance < 0.05,
  },
  {
    id: 'banana-walk',
    title: 'The banana defeats a random walk',
    text: 'A curved ridge needs a different step direction at every point. The random walk cannot follow it: the chains disagree and the mean is off.',
    scenario: { target: 'banana', proposalSd: 0.3 },
    claim: (r) => r.rHat > 1.5 && r.meanError > 0.8,
  },
  {
    id: 'banana-hmc',
    title: 'Hamiltonian follows the curve',
    text: 'The gradient bends the trajectory along the banana, so the four chains agree and the samples are plentiful.',
    scenario: { target: 'banana', sampler: 'hmc', stepSize: 0.15, leapfrog: 15 },
    claim: (r) => r.ess > 500 && r.rHat < 1.05,
  },
  {
    id: 'stuck',
    title: 'Stuck in one mode',
    text: 'Two chains start in each blob and the valley between them has almost no probability: with modest steps a chain never crosses. The chains disagree, and the share of samples in the right-hand blob is nowhere near the true one half.',
    scenario: { target: 'modes', proposalSd: 0.5 },
    claim: (r) => r.rHat > 3 && r.rightShare < 0.35,
  },
  {
    id: 'jumping',
    title: 'Big jumps cross the valley',
    text: 'A jump as large as the gap occasionally lands in the other blob and is accepted, so the chains visit both and agree (at the price of rejecting most proposals).',
    scenario: { target: 'modes', proposalSd: 6 },
    claim: (r) => r.rHat < 1.1 && r.rightShare > 0.35 && r.rightShare < 0.65,
  },
  {
    id: 'short',
    title: 'Not enough steps',
    text: 'After 300 steps from dispersed starting points the chains have not yet met: R-hat above 1.3 says the answer cannot be trusted.',
    scenario: { steps: 300, burnIn: 0 },
    claim: (r) => r.rHat > 1.3,
  },
];
