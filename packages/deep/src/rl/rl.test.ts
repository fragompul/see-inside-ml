import { describe, expect, it } from 'vitest';
import { createRng } from '@sim/core';
import { GRID_PRESETS, GridWorld, gridRows, parseGrid, type GridRewards } from './grid';
import { valueIteration } from './planning';
import {
  DEFAULT_RL_LAB,
  RL_ALGORITHMS,
  RlSession,
  validateRlLab,
  type RlLabConfig,
} from './session';

const REWARDS: GridRewards = { step: -1, goal: 10, pit: -100 };
const make = (rows: string[], slip = 0) => new GridWorld(parseGrid(rows), REWARDS, slip);

describe('grid parsing', () => {
  it('reads the layout, finds the start and round-trips', () => {
    const layout = parseGrid(['S.#', '.XG', '...']);
    expect(layout.width).toBe(3);
    expect(layout.height).toBe(3);
    expect(layout.start).toBe(0);
    expect(layout.kinds[2]).toBe('wall');
    expect(layout.kinds[4]).toBe('pit');
    expect(layout.kinds[5]).toBe('goal');
    expect(gridRows(layout)).toEqual(['S.#', '.XG', '...']);
  });

  it('rejects invalid layouts with a useful message', () => {
    expect(() => parseGrid(['S.G', '...', '..'])).toThrow(/same length/);
    expect(() => parseGrid(['...', '..G', '...'])).toThrow(/exactly one start/);
    expect(() => parseGrid(['S.S', '..G', '...'])).toThrow(/exactly one start/);
    expect(() => parseGrid(['S..', '...', '...'])).toThrow(/at least one goal/);
    expect(() => parseGrid(['S?G', '...', '...'])).toThrow(/Unknown cell/);
    expect(() => parseGrid(['SG', '..'])).toThrow(RangeError);
    expect(() => parseGrid([])).toThrow(RangeError);
  });

  it('every preset is valid', () => {
    for (const preset of Object.values(GRID_PRESETS))
      expect(() => parseGrid(preset.rows)).not.toThrow();
  });
});

describe('dynamics', () => {
  it('moves, and bumps into walls and the edge', () => {
    const w = make(['S.#', '...', '..G']);
    expect(w.move(0, 1)).toBe(1); // right
    expect(w.move(0, 0)).toBe(0); // up from the top row: stays
    expect(w.move(0, 3)).toBe(0); // left from the left column: stays
    expect(w.move(1, 1)).toBe(1); // right into a wall: stays
    expect(w.move(1, 2)).toBe(4); // down
  });

  it('pays step, goal and pit rewards and ends the episode at the terminals', () => {
    const w = make(['SXG', '...', '...']);
    const rng = createRng(1);
    expect(w.step(0, 2, rng)).toEqual({ next: 3, reward: -1, done: false });
    expect(w.step(0, 1, rng)).toEqual({ next: 1, reward: -100, done: true });
    expect(make(['S.G', '...', '...']).step(1, 1, rng)).toEqual({
      next: 2,
      reward: 10,
      done: true,
    });
    expect(w.isTerminal(1)).toBe(true);
    expect(w.isTerminal(0)).toBe(false);
  });

  it('transitions are a probability distribution; slip spreads the intended action', () => {
    const w = make(['S.G', '...', '...'], 0.4);
    for (let s = 0; s < 9; s++)
      for (let a = 0; a < 4; a++) {
        const total = w.transitions(s, a).reduce((sum, t) => sum + t.probability, 0);
        expect(total).toBeCloseTo(1, 12);
      }
    const right = w.transitions(0, 1).filter((t) => t.result.next === 1);
    expect(right.reduce((s, t) => s + t.probability, 0)).toBeCloseTo(0.6 + 0.1, 12);
    const sure = make(['S.G', '...', '...']).transitions(0, 1);
    expect(sure).toHaveLength(1);
    expect(sure[0]?.probability).toBe(1);
  });

  it('slipping happens at the requested rate', () => {
    const w = make(['...', '.S.', '..G'], 0.5);
    const rng = createRng(2);
    let moved = 0;
    const n = 10_000;
    for (let i = 0; i < n; i++) if (w.step(4, 1, rng).next === 5) moved += 1;
    // chosen: 0.5 + 0.5 · 1/4 = 0.625 of the time the agent goes right
    expect(moved / n).toBeGreaterThan(0.6);
    expect(moved / n).toBeLessThan(0.65);
    expect(() => make(['S.G', '...', '...'], 1.5)).toThrow(RangeError);
  });
});

describe('value iteration', () => {
  it('matches hand-computed values on a corridor', () => {
    const w = make(['S.G', '###', '###']);
    const { values, best } = valueIteration(w, 0.9);
    expect(values[1]).toBeCloseTo(10, 6); // one step from the goal: just the goal reward
    expect(values[0]).toBeCloseTo(-1 + 0.9 * 10, 6);
    expect(best[0]).toEqual([false, true, false, false]); // only "right" is optimal at the start
    expect(values[2]).toBe(0); // terminal
    expect(values[3]).toBe(0); // wall
  });

  it('values fall with the distance to the goal and a pit lowers its neighbours', () => {
    const open = make(GRID_PRESETS.open!.rows);
    const v = valueIteration(open, 0.95).values;
    const at = (x: number, y: number) => v[y * 6 + x] as number;
    expect(at(4, 5)).toBeGreaterThan(at(3, 5));
    expect(at(0, 0)).toBeLessThan(at(5, 4));
    const slippery = valueIteration(make(GRID_PRESETS.cliff!.rows, 0.2), 0.95).values;
    const dry = valueIteration(make(GRID_PRESETS.cliff!.rows), 0.95).values;
    // beside the cliff a slippery floor is worth much less than a dry one
    expect(slippery[2 * 12 + 5] as number).toBeLessThan((dry[2 * 12 + 5] as number) - 3);
  });

  it('satisfies the Bellman optimality equation everywhere', () => {
    const w = make(GRID_PRESETS.maze!.rows, 0.1);
    const { values, q } = valueIteration(w, 0.9);
    for (let s = 0; s < w.states; s++) {
      if (w.isTerminal(s) || w.kind(s) === 'wall') continue;
      const top = Math.max(...[0, 1, 2, 3].map((a) => q[s * 4 + a] as number));
      expect(values[s]).toBeCloseTo(top, 8);
    }
  });
});

describe('RlSession', () => {
  const config = (patch: Partial<RlLabConfig> = {}): RlLabConfig => ({
    ...DEFAULT_RL_LAB,
    ...patch,
  });

  it('is deterministic for a seed and chunk-independent', () => {
    const a = new RlSession(config({ grid: GRID_PRESETS.open!.rows }));
    const b = new RlSession(config({ grid: GRID_PRESETS.open!.rows }));
    a.train(500);
    b.train(120);
    b.train(380);
    expect(a.q).toEqual(b.q);
    expect(a.history).toEqual(b.history);
    expect(a.totalSteps).toBe(500);
  });

  it.each(RL_ALGORITHMS)('%s learns the optimal route in an open room', (algorithm) => {
    const s = new RlSession(
      config({
        grid: GRID_PRESETS.open!.rows,
        algorithm,
        epsilonMin: 0.02,
        epsilon: 0.3,
        epsilonDecay: 0.99,
      }),
    );
    s.trainEpisodes(1200);
    const path = s.greedyPath();
    expect(path.outcome).toBe('goal');
    expect(path.cells.length - 1).toBe(10); // 5 right + 5 down: the shortest route
    expect(s.recentReturn(100)).toBeGreaterThan(-3);
  });

  it('Q-learning converges close to the optimal values and policy (known dynamics)', () => {
    const s = new RlSession(config({ grid: GRID_PRESETS.open!.rows }));
    s.trainEpisodes(1500);
    expect(s.agreement()).toBeGreaterThan(0.9);
    expect(s.valueError()).toBeLessThan(2);
    const fresh = new RlSession(config({ grid: GRID_PRESETS.open!.rows }));
    expect(fresh.valueError()).toBeGreaterThan(s.valueError() * 3);
  });

  it('an agent with no discount cannot plan: its learned route fails and its returns are poor', () => {
    const patient = new RlSession(config({ grid: GRID_PRESETS.open!.rows }));
    patient.trainEpisodes(600);
    const myopic = new RlSession(config({ grid: GRID_PRESETS.open!.rows, gamma: 0 }));
    myopic.trainEpisodes(600);
    expect(patient.greedyPath().outcome).toBe('goal');
    expect(myopic.greedyPath().outcome).not.toBe('goal');
    expect(patient.recentReturn(100)).toBeGreaterThan(-3);
    expect(myopic.recentReturn(100)).toBeLessThan(-30);
  });

  it('on the cliff, Q-learning hugs the edge while SARSA keeps its distance', () => {
    const rowsOf = (algorithm: 'qlearning' | 'sarsa') => {
      const s = new RlSession(config({ algorithm }));
      s.trainEpisodes(1500);
      const path = s.greedyPath();
      return {
        path,
        top: Math.min(...path.cells.map((c) => Math.floor(c / 12))),
        online: s.recentReturn(100),
      };
    };
    const q = rowsOf('qlearning');
    const sarsa = rowsOf('sarsa');
    expect(q.path.outcome).toBe('goal');
    expect(q.path.cells.length - 1).toBe(13); // the shortest route, along the cliff edge
    expect(q.top).toBe(2);
    expect(sarsa.path.outcome).toBe('goal');
    expect(sarsa.path.cells.length - 1).toBeGreaterThan(13);
    expect(sarsa.top).toBeLessThanOrEqual(1);
    expect(sarsa.online).toBeGreaterThan(q.online + 5); // fewer falls while still exploring
  });

  it('records episodes with their outcome, and the exploration rate decays to its floor', () => {
    const s = new RlSession(
      config({ grid: GRID_PRESETS.cliff!.rows, epsilon: 0.5, epsilonMin: 0.1, epsilonDecay: 0.9 }),
    );
    s.trainEpisodes(60);
    expect(s.history).toHaveLength(60);
    expect(s.history.map((e) => e.episode)).toEqual(Array.from({ length: 60 }, (_, i) => i + 1));
    expect(s.history.every((e) => ['goal', 'pit', 'timeout'].includes(e.outcome))).toBe(true);
    expect(s.history[0]?.epsilon).toBe(0.5);
    expect(s.epsilon).toBeCloseTo(0.1, 12);
    expect(s.history.every((e) => e.steps >= 1 && e.steps <= s.config.maxSteps)).toBe(true);
  });

  it('cuts episodes off at the maximum number of steps', () => {
    const s = new RlSession(
      config({ grid: GRID_PRESETS.open!.rows, maxSteps: 20, epsilon: 1, epsilonMin: 1 }),
    );
    s.trainEpisodes(30);
    expect(s.history.every((e) => e.steps <= 20)).toBe(true);
    expect(s.history.some((e) => e.outcome === 'timeout')).toBe(true);
  });

  it('applies learning settings live and rebuilds when the world changes', () => {
    const s = new RlSession(config({ grid: GRID_PRESETS.open!.rows }));
    s.trainEpisodes(20);
    const q = Float64Array.from(s.q);
    expect(s.update({ alpha: 0.1, algorithm: 'sarsa', epsilonDecay: 0.9, gamma: 0.8 })).toBe(
      'live',
    );
    expect(s.q).toEqual(q);
    expect(s.history).toHaveLength(20);
    expect(s.optimal.values[0]).not.toBe(valueIteration(s.world, 0.95).values[0]); // recomputed for γ = 0.8
    expect(s.update({ slip: 0.2 })).toBe('rebuilt');
    expect(s.history).toHaveLength(0);
    expect(s.update({ grid: GRID_PRESETS.maze!.rows })).toBe('rebuilt');
    expect(s.world.layout.width).toBe(8);
    expect(s.update({ stepReward: -0.5 })).toBe('rebuilt');
  });

  it('rejects invalid settings and keeps the previous ones', () => {
    const s = new RlSession(config());
    expect(() => s.update({ alpha: 0 })).toThrow();
    expect(() => s.update({ gamma: 1 })).toThrow();
    expect(() => s.update({ slip: 0.9 })).toThrow();
    expect(() => s.update({ epsilonDecay: 0 })).toThrow();
    expect(() => s.update({ epsilonMin: 0.9, epsilon: 0.3 })).toThrow();
    expect(() => s.update({ grid: ['S.G'] })).toThrow();
    expect(() => s.update({ algorithm: 'x' as 'sarsa' })).toThrow();
    expect(() => validateRlLab({ ...DEFAULT_RL_LAB, maxSteps: 5 })).toThrow();
    expect(s.config.alpha).toBe(DEFAULT_RL_LAB.alpha);
  });

  it('reset forgets everything; greedy path detects loops; reachable set excludes walled-off cells', () => {
    const s = new RlSession(config({ grid: ['S.#G', '..#.', '..#.'] }));
    expect(s.greedyPath().outcome).toBe('loop'); // an untrained agent pushes into the same wall forever
    s.train(300);
    s.reset();
    expect(s.totalSteps).toBe(0);
    expect(s.history).toHaveLength(0);
    expect(s.q.every((v) => v === 0)).toBe(true);
    // the goal side of the wall can never be reached
    expect(s.reachable.has(3)).toBe(false);
    expect(s.reachable.has(0)).toBe(true);
    expect(s.agreement()).toBeGreaterThanOrEqual(0);
  });
});
