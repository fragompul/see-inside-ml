import { DEFAULT_RL_LAB, GRID_PRESETS } from '@sim/deep';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { resetRlSession, useRlStore } from './store';

const state = () => useRlStore.getState();

async function waitFor(condition: () => boolean, timeout = 8000) {
  const start = Date.now();
  while (!condition()) {
    if (Date.now() - start > timeout) throw new Error('waitFor timed out');
    await new Promise((resolve) => setTimeout(resolve, 10));
  }
}

describe('rl store', () => {
  beforeEach(() => {
    resetRlSession();
    state().init();
  });
  afterEach(() => state().setRunning(false));

  it('starts with the cliff, an untrained agent on the start cell and the optimal values ready', () => {
    const snap = state().snapshot!;
    expect(state().config.grid).toEqual(DEFAULT_RL_LAB.grid);
    expect(snap.q).toHaveLength(12 * 4 * 4);
    expect(snap.q.every((v) => v === 0)).toBe(true);
    expect(snap.agent).toBe(36); // bottom-left
    expect(snap.optimal[36]).not.toBe(0);
    expect(state().episode).toBe(0);
    expect(state().epsilon).toBe(DEFAULT_RL_LAB.epsilon);
  });

  it('steps training one environment step at a time', () => {
    state().stepOnce(1);
    expect(state().totalSteps).toBe(1);
    expect(state().running).toBe(false);
    state().stepOnce(500);
    expect(state().totalSteps).toBe(501);
    expect(state().snapshot!.visits.reduce((a, b) => a + b, 0)).toBe(501);
  });

  it('fast-forwards whole episodes and records them', () => {
    state().episodes(25);
    expect(state().episode).toBe(25);
    expect(state().history).toHaveLength(25);
    expect(state().epsilon).toBeLessThan(DEFAULT_RL_LAB.epsilon);
    expect(state().snapshot!.q.some((v) => v !== 0)).toBe(true);
  });

  it('painting a cell rebuilds the world; the start moves rather than duplicates', () => {
    state().episodes(5);
    state().paint(13, 'wall');
    expect(state().config.grid[1]?.[1]).toBe('#');
    expect(state().episode).toBe(0); // learning restarted
    state().paint(5, 'start');
    const starts = state().config.grid.join('').split('S').length - 1;
    expect(starts).toBe(1);
    expect(state().config.grid[0]?.[5]).toBe('S');
    expect(state().config.grid[3]?.[0]).toBe('.');
    expect(state().snapshot!.agent).toBe(5);
    state().paint(5, 'start'); // painting the same thing again changes nothing
    expect(state().notice).toBeNull();
  });

  it('refuses to erase the only goal, with a notice, and keeps the layout', () => {
    const before = [...state().config.grid];
    state().paint(47, 'empty'); // the goal at the bottom-right
    expect(state().notice).toMatch(/at least one goal/);
    expect(state().config.grid).toEqual(before);
    state().paint(46, 'goal'); // add another goal first, then erasing the first is fine
    state().paint(47, 'empty');
    expect(state().notice).toBeNull();
  });

  it('loads presets and ignores unknown ones', () => {
    state().loadPreset('maze');
    expect(state().config.grid).toEqual(GRID_PRESETS.maze!.rows);
    state().loadPreset('nope');
    expect(state().config.grid).toEqual(GRID_PRESETS.maze!.rows);
    expect(state().snapshot!.q).toHaveLength(8 * 6 * 4);
  });

  it('applies learning settings live and rebuilds for environment settings', () => {
    state().episodes(10);
    state().patch({ alpha: 0.2, gamma: 0.9, algorithm: 'sarsa' });
    expect(state().episode).toBe(10);
    state().patch({ slip: 0.2 });
    expect(state().episode).toBe(0);
    state().patch({ alpha: 0 });
    expect(state().notice).toMatch(/Learning rate/);
    expect(state().config.alpha).toBe(0.2);
  });

  it('inspecting a cell is remembered until the layout changes', () => {
    state().inspect(5);
    expect(state().inspected).toBe(5);
    state().loadPreset('open');
    expect(state().inspected).toBeNull();
    state().inspect(null);
    expect(state().inspected).toBeNull();
  });

  it('runs continuously, counting steps, and stops when paused', async () => {
    state().setSpeed(100);
    state().setRunning(true);
    await waitFor(() => state().totalSteps > 2000);
    state().setRunning(false);
    const settled = state().totalSteps;
    await new Promise((resolve) => setTimeout(resolve, 150));
    expect(state().totalSteps).toBe(settled);
  });

  it('reset forgets what was learned', () => {
    state().episodes(30);
    state().reset();
    expect(state().episode).toBe(0);
    expect(state().history).toHaveLength(0);
    expect(state().snapshot!.q.every((v) => v === 0)).toBe(true);
  });
});
