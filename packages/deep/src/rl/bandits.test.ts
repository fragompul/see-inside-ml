import { describe, expect, it } from 'vitest';
import { BANDIT_STRATEGIES, meanRegret, runBandit, type BanditProblem } from './bandits';

const bernoulli: BanditProblem = { means: [0.2, 0.4, 0.5, 0.6], noise: 0 };
const gaussian: BanditProblem = { means: [0, 0.5, 1, 0.8], noise: 1 };

describe('bandit runs', () => {
  it.each(BANDIT_STRATEGIES)('%s: regret never decreases and counts add up', (strategy) => {
    const run = runBandit(bernoulli, strategy, { steps: 500, seed: 1 });
    for (let i = 1; i < run.regret.length; i++)
      expect(run.regret[i]!).toBeGreaterThanOrEqual(run.regret[i - 1]! - 1e-12);
    expect(run.counts.reduce((a, b) => a + b, 0)).toBe(500);
    expect(run.pulls).toHaveLength(500);
  });

  it('is deterministic for a seed', () => {
    const a = runBandit(gaussian, 'ucb', { steps: 200, seed: 7 });
    const b = runBandit(gaussian, 'ucb', { steps: 200, seed: 7 });
    expect(a.pulls).toEqual(b.pulls);
  });

  it('pseudo-regret equals the gap of the arms that were pulled', () => {
    const run = runBandit(bernoulli, 'epsilon', { steps: 300, seed: 3 });
    const expected = run.counts.reduce((s, c, i) => s + c * (0.6 - bernoulli.means[i]!), 0);
    expect(run.regret[299]).toBeCloseTo(expected, 9);
  });

  it('empirical estimates converge to the true means for arms that are pulled a lot', () => {
    const run = runBandit(gaussian, 'epsilon', { steps: 5000, seed: 2, epsilon: 0.3 });
    run.estimates.forEach((m, i) => expect(Math.abs(m - gaussian.means[i]!)).toBeLessThan(0.25));
  });
});

describe('exploration strategies', () => {
  const steps = 2000;
  const runs = 40;
  it('UCB and Thompson sampling beat a fixed epsilon in the long run, which regrets linearly', () => {
    const eps = meanRegret(bernoulli, 'epsilon', { steps, seed: 1, epsilon: 0.1 }, runs);
    const ucb = meanRegret(bernoulli, 'ucb', { steps, seed: 1 }, runs);
    const ts = meanRegret(bernoulli, 'thompson', { steps, seed: 1 }, runs);
    expect(ucb[steps - 1]!).toBeLessThan(eps[steps - 1]!);
    expect(ts[steps - 1]!).toBeLessThan(eps[steps - 1]!);
    // a fixed epsilon keeps paying about eps * mean gap per step: its regret keeps growing linearly
    const slopeEps = (eps[steps - 1]! - eps[steps / 2 - 1]!) / (steps / 2);
    const slopeTs = (ts[steps - 1]! - ts[steps / 2 - 1]!) / (steps / 2);
    expect(slopeEps).toBeGreaterThan(0.01);
    expect(slopeTs).toBeLessThan(slopeEps / 2);
  });

  it('pure greedy can lock onto a bad arm: its regret is sometimes linear', () => {
    const regrets = Array.from(
      { length: 60 },
      (_, s) => runBandit(bernoulli, 'greedy', { steps: 1000, seed: s + 1 }).regret[999]!,
    );
    expect(Math.max(...regrets)).toBeGreaterThan(50);
  });

  it('optimism makes early exploration systematic: every arm is tried', () => {
    const run = runBandit(bernoulli, 'optimistic', { steps: 100, seed: 4, optimism: 5 });
    for (const c of run.counts) expect(c).toBeGreaterThan(0);
  });

  it('the decaying epsilon explores less and less', () => {
    const run = runBandit(bernoulli, 'decaying', { steps: 4000, seed: 5 });
    const best = bernoulli.means.indexOf(Math.max(...bernoulli.means));
    const early = run.pulls.slice(0, 200).filter((a) => a === best).length / 200;
    const late = run.pulls.slice(3000).filter((a) => a === best).length / 1000;
    expect(late).toBeGreaterThan(early);
    expect(late).toBeGreaterThan(0.8);
  });

  it('the gradient bandit learns to prefer the best arm', () => {
    const run = runBandit(gaussian, 'gradient', { steps: 6000, seed: 6, learningRate: 0.1 });
    expect(run.counts.indexOf(Math.max(...run.counts))).toBe(2);
  });
});
