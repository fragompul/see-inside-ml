import { describe, expect, it } from 'vitest';
import {
  EMPTY_BOARD,
  legalMoves,
  mcts,
  minimaxValue,
  optimalMoves,
  play,
  playGame,
  turnOf,
  winner,
  type Board,
  type Cell,
} from './mcts';

const board = (s: string): Board =>
  [...s].map((c) => (c === 'X' ? 1 : c === 'O' ? 2 : 0)) as Cell[];

describe('the game', () => {
  it('detects wins in rows, columns and diagonals and counts whose turn it is', () => {
    expect(winner(board('XXX......'))).toBe(1);
    expect(winner(board('O..O..O..'))).toBe(2);
    expect(winner(board('X...X...X'))).toBe(1);
    expect(winner(board('XOXOXOOXO'))).toBe(0);
    expect(turnOf(EMPTY_BOARD)).toBe(1);
    expect(turnOf(board('X........'))).toBe(2);
    expect(legalMoves(board('XXX......'))).toEqual([]);
  });

  it('perfect play from the empty board is a draw, and the corner opening does not lose', () => {
    expect(minimaxValue(EMPTY_BOARD)).toBe(0);
    expect(minimaxValue(play(EMPTY_BOARD, 0, 1))).toBe(0);
  });

  it('the optimal move takes a win when there is one and blocks a threat otherwise', () => {
    expect(optimalMoves(board('XX.OO....'))).toEqual([2]);
    expect(optimalMoves(board('XX..O....'))).toContain(2);
    expect(optimalMoves(board('XX..O....'))).toHaveLength(1);
  });
});

describe('Monte Carlo tree search', () => {
  it('finds a win in one and a forced block with enough simulations', () => {
    expect(mcts(board('XX.OO....'), 800, 1.4, 1).move).toBe(2);
    expect(mcts(board('XX..O....'), 1500, 1.4, 2).move).toBe(2);
  });

  it('visit counts add up to the number of simulations and the chosen move is the most visited', () => {
    const r = mcts(board('X...O....'), 500, 1.4, 3);
    expect(r.root.reduce((a, c) => a + c.visits, 0)).toBe(500);
    const top = r.root.reduce((a, b) => (b.visits > a.visits ? b : a));
    expect(r.move).toBe(top.move);
  });

  it('is deterministic for a seed', () => {
    expect(mcts(EMPTY_BOARD, 300, 1.4, 5).root).toEqual(mcts(EMPTY_BOARD, 300, 1.4, 5).root);
  });

  it('more simulations play stronger: the search with many never loses to random play, one with few sometimes does', () => {
    let lossesStrong = 0;
    let lossesWeak = 0;
    for (let g = 0; g < 30; g++) {
      if (
        playGame(
          { kind: 'random' },
          { kind: 'mcts', simulations: 400, exploration: 1.4 },
          g + 1,
        ) === 1
      )
        lossesStrong++;
      if (
        playGame({ kind: 'random' }, { kind: 'mcts', simulations: 3, exploration: 1.4 }, g + 1) ===
        1
      )
        lossesWeak++;
    }
    expect(lossesStrong).toBe(0);
    expect(lossesWeak).toBeGreaterThan(lossesStrong);
  });
});
