import { mcts, optimalMoves, playGame, type Board, type Cell, type MctsResult } from '@sim/deep';

export type PositionId = 'empty' | 'win' | 'block' | 'fork';

const b = (s: string): Board => [...s].map((c) => (c === 'X' ? 1 : c === 'O' ? 2 : 0)) as Cell[];

export const POSITIONS: Record<PositionId, { label: string; note: string; board: Board }> = {
  empty: {
    label: 'Empty board',
    note: 'X to move. Every game is a draw with perfect play, but not every opening is equally safe.',
    board: b('.........'),
  },
  win: {
    label: 'Win in one',
    note: 'X has two in a row at the top and can complete the line.',
    board: b('XX.OO....'),
  },
  block: {
    label: 'Must block',
    note: 'X threatens to win at the top; O must block it.',
    board: b('XX..O....'),
  },
  fork: {
    label: 'Avoid the trap',
    note: 'X has two opposite corners; O must not take a corner or X wins by force, only an edge draws.',
    board: b('X...O...X'),
  },
};

export interface MctsScenario {
  position: PositionId;
  simulations: number;
  exploration: number;
  seed: number;
}

export const DEFAULT_MCTS: MctsScenario = {
  position: 'block',
  simulations: 200,
  exploration: 1.4,
  seed: 1,
};

export interface Tournament {
  wins: number;
  draws: number;
  losses: number;
}

export interface MctsRun {
  scenario: MctsScenario;
  result: MctsResult;
  optimal: number[];
  /** Share of 20 independent searches (different random seeds) that pick an optimal move. */
  reliability: number;
  /** The searcher against uniformly random play, taking both sides, and as O against a perfect X. */
  vsRandom: Tournament;
  vsPerfect: Tournament;
}

const GAMES = 20;

export function runMcts(s: MctsScenario): MctsRun {
  const board = POSITIONS[s.position].board;
  const result = mcts(board, s.simulations, s.exploration, s.seed);
  const optimal = optimalMoves(board);
  let right = 0;
  for (let k = 0; k < 20; k++)
    if (optimal.includes(mcts(board, s.simulations, s.exploration, 100 + k).move)) right++;
  const me = { kind: 'mcts' as const, simulations: s.simulations, exploration: s.exploration };
  const vsRandom: Tournament = { wins: 0, draws: 0, losses: 0 };
  const vsPerfect: Tournament = { wins: 0, draws: 0, losses: 0 };
  const add = (t: Tournament, outcome: 0 | 1 | 2, mine: 1 | 2) => {
    if (outcome === 0) t.draws++;
    else if (outcome === mine) t.wins++;
    else t.losses++;
  };
  for (let g = 0; g < GAMES; g++) {
    add(vsRandom, playGame(me, { kind: 'random' }, g + 1), 1);
    add(vsRandom, playGame({ kind: 'random' }, me, g + 1), 2);
    add(vsPerfect, playGame({ kind: 'perfect' }, me, g + 1), 2);
  }
  return { scenario: s, result, optimal, reliability: right / 20, vsRandom, vsPerfect };
}
