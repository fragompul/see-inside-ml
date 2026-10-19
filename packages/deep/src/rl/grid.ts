import { type Rng } from '@sim/core';

/**
 * A small gridworld. The agent starts on `S`, moves up / right / down / left, and every step costs a
 * little. Reaching `G` ends the episode with a reward; falling into an `X` pit ends it with a big
 * penalty; `#` are walls (moving into one, or off the edge, leaves the agent where it is).
 * With probability `slip` the chosen action is replaced by a random one (a slippery floor).
 */
export const ACTIONS = ['up', 'right', 'down', 'left'] as const;
export type Action = 0 | 1 | 2 | 3;
export const ACTION_COUNT = 4;
const DX = [0, 1, 0, -1] as const;
const DY = [-1, 0, 1, 0] as const;

export type CellKind = 'empty' | 'wall' | 'goal' | 'pit' | 'start';
const CHAR_KIND: Record<string, CellKind> = {
  '.': 'empty',
  '#': 'wall',
  G: 'goal',
  X: 'pit',
  S: 'start',
};
export const KIND_CHAR: Record<CellKind, string> = {
  empty: '.',
  wall: '#',
  goal: 'G',
  pit: 'X',
  start: 'S',
};

export interface GridRewards {
  /** Reward of an ordinary step (usually a small negative number). */
  step: number;
  goal: number;
  pit: number;
}

export const DEFAULT_REWARDS: GridRewards = { step: -1, goal: 10, pit: -100 };

export const GRID_LIMITS = { width: [3, 14], height: [3, 10] } as const;

/** Named layouts, as rows of characters. */
export const GRID_PRESETS: Record<string, { label: string; note: string; rows: string[] }> = {
  open: {
    label: 'Open room',
    note: 'No obstacles: the shortest path is obvious, so this is a clean test of the algorithms.',
    rows: ['S.....', '......', '......', '......', '......', '.....G'],
  },
  cliff: {
    label: 'Cliff',
    note: 'The classic: the shortest route runs along the edge of a cliff (X). A single wrong step costs −100.',
    rows: ['............', '............', '............', 'SXXXXXXXXXXG'],
  },
  maze: {
    label: 'Maze',
    note: 'Walls force detours; the agent must discover the way round.',
    rows: ['S..#....', '.#.#.##.', '.#...#..', '.####.#.', '......#G', '.####...'],
  },
  detour: {
    label: 'Pit by the gap',
    note: 'The short way through the wall passes right beside a pit; the long way round is safe.',
    rows: ['S.......', '.######.', '.#XG.#..', '.#...#..', '.##.##..', '........'],
  },
};

export interface GridLayout {
  width: number;
  height: number;
  kinds: CellKind[];
  start: number;
}

/** Parses rows of characters; throws a `RangeError` with a helpful message if the layout is invalid. */
export function parseGrid(rows: readonly string[]): GridLayout {
  const height = rows.length;
  const width = rows[0]?.length ?? 0;
  const [wLo, wHi] = GRID_LIMITS.width;
  const [hLo, hHi] = GRID_LIMITS.height;
  if (width < wLo || width > wHi) throw new RangeError(`Grid width must be in [${wLo}, ${wHi}]`);
  if (height < hLo || height > hHi) throw new RangeError(`Grid height must be in [${hLo}, ${hHi}]`);
  const kinds: CellKind[] = [];
  let start = -1;
  let goals = 0;
  rows.forEach((row, y) => {
    if (row.length !== width) throw new RangeError('All rows must have the same length');
    [...row].forEach((ch, x) => {
      const kind = CHAR_KIND[ch];
      if (!kind) throw new RangeError(`Unknown cell "${ch}"`);
      if (kind === 'start') {
        if (start >= 0) throw new RangeError('There must be exactly one start cell');
        start = y * width + x;
      }
      if (kind === 'goal') goals += 1;
      kinds.push(kind);
    });
  });
  if (start < 0) throw new RangeError('There must be exactly one start cell');
  if (goals < 1) throw new RangeError('There must be at least one goal');
  return { width, height, kinds, start };
}

export const gridRows = (layout: GridLayout): string[] =>
  Array.from({ length: layout.height }, (_, y) =>
    layout.kinds
      .slice(y * layout.width, (y + 1) * layout.width)
      .map((k) => KIND_CHAR[k])
      .join(''),
  );

export interface StepResult {
  next: number;
  reward: number;
  done: boolean;
}

export class GridWorld {
  readonly states: number;

  constructor(
    readonly layout: GridLayout,
    readonly rewards: GridRewards = DEFAULT_REWARDS,
    readonly slip = 0,
  ) {
    if (!(slip >= 0 && slip <= 1)) throw new RangeError('Slip must be in [0, 1]');
    this.states = layout.width * layout.height;
  }

  get start(): number {
    return this.layout.start;
  }

  kind(state: number): CellKind {
    return this.layout.kinds[state] as CellKind;
  }

  isTerminal(state: number): boolean {
    const k = this.kind(state);
    return k === 'goal' || k === 'pit';
  }

  /** Where moving in `action` from `state` leads (ignoring slipping, and not counting rewards). */
  move(state: number, action: number): number {
    const { width, height } = this.layout;
    const x = (state % width) + (DX[action] as number);
    const y = Math.floor(state / width) + (DY[action] as number);
    if (x < 0 || y < 0 || x >= width || y >= height) return state;
    const next = y * width + x;
    return this.kind(next) === 'wall' ? state : next;
  }

  /** The outcome of arriving in `next`. */
  private outcome(next: number): StepResult {
    const kind = this.kind(next);
    if (kind === 'goal') return { next, reward: this.rewards.goal, done: true };
    if (kind === 'pit') return { next, reward: this.rewards.pit, done: true };
    return { next, reward: this.rewards.step, done: false };
  }

  /** One step with the real (possibly slippery) dynamics. */
  step(state: number, action: number, rng: Rng): StepResult {
    const actual = this.slip > 0 && rng.next() < this.slip ? Math.floor(rng.next() * 4) : action;
    return this.outcome(this.move(state, actual));
  }

  /** The exact distribution of outcomes of an action: `[probability, result]` pairs. */
  transitions(state: number, action: number): { probability: number; result: StepResult }[] {
    const out: { probability: number; result: StepResult }[] = [];
    for (let a = 0; a < 4; a++) {
      const probability = (a === action ? 1 - this.slip : 0) + this.slip / 4;
      if (probability > 0) out.push({ probability, result: this.outcome(this.move(state, a)) });
    }
    return out;
  }
}
