import { createRng, type Rng } from '@sim/core';

/**
 * Monte Carlo tree search on noughts and crosses. The search grows a tree of positions: select a promising
 * leaf with UCB, add a child, finish the game with random moves (a rollout) and back the result up the path.
 * Perfect play is known (minimax), so the quality of the search can be measured.
 */
export type Cell = 0 | 1 | 2; // empty, X (player 1), O (player 2)
export type Board = readonly Cell[];

export const EMPTY_BOARD: Board = Array.from({ length: 9 }, () => 0 as Cell);
const LINES = [
  [0, 1, 2],
  [3, 4, 5],
  [6, 7, 8],
  [0, 3, 6],
  [1, 4, 7],
  [2, 5, 8],
  [0, 4, 8],
  [2, 4, 6],
] as const;

export const winner = (b: Board): 0 | 1 | 2 => {
  for (const [a, c, d] of LINES)
    if (b[a] !== 0 && b[a] === b[c] && b[a] === b[d]) return b[a] as 1 | 2;
  return 0;
};
export const legalMoves = (b: Board): number[] =>
  winner(b) !== 0 ? [] : b.flatMap((c, i) => (c === 0 ? [i] : []));
export const play = (b: Board, move: number, player: 1 | 2): Board =>
  b.map((c, i) => (i === move ? player : c)) as Board;
/** Whose turn it is: X moves when the counts are equal. */
export const turnOf = (b: Board): 1 | 2 =>
  b.filter((c) => c === 1).length === b.filter((c) => c === 2).length ? 1 : 2;
export const isOver = (b: Board): boolean => winner(b) !== 0 || b.every((c) => c !== 0);

const memo = new Map<string, number>();
/** Minimax value for player 1 (X): +1 X wins, 0 draw, -1 O wins, with perfect play by both. */
export function minimaxValue(b: Board): number {
  const key = b.join('');
  const hit = memo.get(key);
  if (hit !== undefined) return hit;
  const w = winner(b);
  let v: number;
  if (w === 1) v = 1;
  else if (w === 2) v = -1;
  else if (b.every((c) => c !== 0)) v = 0;
  else {
    const turn = turnOf(b);
    const values = legalMoves(b).map((m) => minimaxValue(play(b, m, turn)));
    v = turn === 1 ? Math.max(...values) : Math.min(...values);
  }
  memo.set(key, v);
  return v;
}

/** The moves that keep the best achievable result for the player to move. */
export function optimalMoves(b: Board): number[] {
  const turn = turnOf(b);
  const scored = legalMoves(b).map((m) => ({
    m,
    v: minimaxValue(play(b, m, turn)) * (turn === 1 ? 1 : -1),
  }));
  const best = Math.max(...scored.map((s) => s.v));
  return scored.filter((s) => s.v === best).map((s) => s.m);
}

function rollout(b: Board, rng: Rng): 0 | 1 | 2 {
  let cur = b;
  while (!isOver(cur)) {
    const moves = legalMoves(cur);
    cur = play(cur, moves[Math.floor(rng.next() * moves.length)] as number, turnOf(cur));
  }
  return winner(cur);
}

interface Node {
  board: Board;
  parent: Node | null;
  move: number;
  /** The player who made the move leading here. */
  mover: 1 | 2;
  visits: number;
  /** Sum of results from the mover's point of view (1 win, 0.5 draw, 0 loss). */
  wins: number;
  children: Node[];
  untried: number[];
}

const makeNode = (board: Board, parent: Node | null, move: number): Node => ({
  board,
  parent,
  move,
  mover: parent ? turnOf(parent.board) : turnOf(board) === 1 ? 2 : 1,
  visits: 0,
  wins: 0,
  children: [],
  untried: legalMoves(board),
});

export interface MctsResult {
  move: number;
  /** Statistics of the moves at the root. */
  root: { move: number; visits: number; winRate: number }[];
  simulations: number;
}

export function mcts(
  board: Board,
  simulations: number,
  exploration: number,
  seed: number,
): MctsResult {
  const rng = createRng(seed);
  const root = makeNode(board, null, -1);
  for (let sim = 0; sim < simulations; sim++) {
    let node = root;
    // selection
    while (node.untried.length === 0 && node.children.length > 0) {
      const parent = node;
      node = parent.children.reduce((best, c) => {
        const score =
          c.wins / c.visits + exploration * Math.sqrt(Math.log(parent.visits) / c.visits);
        const bestScore =
          best.wins / best.visits + exploration * Math.sqrt(Math.log(parent.visits) / best.visits);
        return score > bestScore ? c : best;
      });
    }
    // expansion
    if (node.untried.length > 0) {
      const i = Math.floor(rng.next() * node.untried.length);
      const move = node.untried.splice(i, 1)[0] as number;
      const child = makeNode(play(node.board, move, turnOf(node.board)), node, move);
      node.children.push(child);
      node = child;
    }
    // rollout and backup
    const result = rollout(node.board, rng);
    for (let n: Node | null = node; n; n = n.parent) {
      n.visits += 1;
      if (result === 0) n.wins += 0.5;
      else if (result === n.mover) n.wins += 1;
    }
  }
  const stats = root.children.map((c) => ({
    move: c.move,
    visits: c.visits,
    winRate: c.wins / c.visits,
  }));
  const best = stats.reduce(
    (a, b) => (b.visits > a.visits ? b : a),
    stats[0] as (typeof stats)[number],
  );
  return { move: best.move, root: stats, simulations };
}

export type Player =
  | { kind: 'random' }
  | { kind: 'perfect' }
  | { kind: 'mcts'; simulations: number; exploration: number };

export function chooseMove(board: Board, player: Player, rng: Rng, seed: number): number {
  const moves = legalMoves(board);
  if (player.kind === 'random') return moves[Math.floor(rng.next() * moves.length)] as number;
  if (player.kind === 'perfect') {
    const best = optimalMoves(board);
    return best[Math.floor(rng.next() * best.length)] as number;
  }
  return mcts(board, player.simulations, player.exploration, seed).move;
}

/** Plays one game; returns 1 if the first player (X) wins, 2 if O wins, 0 for a draw. */
export function playGame(x: Player, o: Player, seed: number): 0 | 1 | 2 {
  const rng = createRng(seed);
  let b: Board = EMPTY_BOARD;
  let s = seed * 31;
  while (!isOver(b)) {
    const turn = turnOf(b);
    b = play(b, chooseMove(b, turn === 1 ? x : o, rng, (s += 7)), turn);
  }
  return winner(b);
}
