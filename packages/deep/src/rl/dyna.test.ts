import { describe, expect, it } from 'vitest';
import { DEFAULT_DYNA, averageSteps, trainDyna } from './dyna';
import { DEFAULT_REWARDS, GRID_PRESETS, GridWorld, parseGrid } from './grid';
import { valueIteration } from './planning';

const maze = new GridWorld(
  parseGrid((GRID_PRESETS.maze as { rows: string[] }).rows),
  DEFAULT_REWARDS,
  0,
);

describe('Dyna-Q', () => {
  it('without planning it is plain Q-learning: no imagined updates', () => {
    const r = trainDyna(maze, { ...DEFAULT_DYNA, planning: 0, episodes: 5 });
    expect(r.imaginedSteps).toBe(0);
    expect(r.realSteps).toBe(r.steps.reduce((a, b) => a + b, 0));
  });

  it('planning does n imagined updates per real step', () => {
    const r = trainDyna(maze, { ...DEFAULT_DYNA, planning: 10, episodes: 5 });
    expect(r.imaginedSteps).toBe(10 * r.realSteps);
  });

  it('planning learns the maze in far fewer real steps', () => {
    const none = averageSteps(maze, { ...DEFAULT_DYNA, planning: 0, episodes: 15 }, 20);
    const plan = averageSteps(maze, { ...DEFAULT_DYNA, planning: 30, episodes: 15 }, 20);
    const total = (v: number[]) => v.reduce((a, b) => a + b, 0);
    expect(total(plan)).toBeLessThan(total(none) / 2);
  });

  it('converges to the optimal policy: following it reaches the goal in the optimal number of steps', () => {
    const r = trainDyna(maze, {
      ...DEFAULT_DYNA,
      planning: 30,
      episodes: 60,
      epsilon: 0.1,
      seed: 4,
    });
    const optimal = valueIteration(maze, DEFAULT_DYNA.gamma);
    let s = maze.start;
    let steps = 0;
    while (!maze.isTerminal(s) && steps < 100) {
      s = maze.move(s, r.policy[s] as number);
      steps++;
    }
    expect(maze.kind(s)).toBe('goal');
    // the optimal value of the start state tells the length of the shortest path: 10 - (steps-1) with step cost -1 (undiscounted ordering)
    let t = maze.start;
    let shortest = 0;
    while (!maze.isTerminal(t) && shortest < 100) {
      const q = Array.from({ length: 4 }, (_, a) => optimal.q[t * 4 + a] as number);
      t = maze.move(t, q.indexOf(Math.max(...q)));
      shortest++;
    }
    expect(steps).toBe(shortest);
  });

  it('is deterministic for a seed', () => {
    expect(trainDyna(maze, { ...DEFAULT_DYNA, seed: 3 }).steps).toEqual(
      trainDyna(maze, { ...DEFAULT_DYNA, seed: 3 }).steps,
    );
  });
});
