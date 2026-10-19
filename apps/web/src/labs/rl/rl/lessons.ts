import { DEFAULT_RL_LAB, GRID_PRESETS, RlSession, type RlLabConfig } from '@sim/deep';

/**
 * One-click experiments. `claim.check` verifies, on a session trained for `claim.episodes` episodes
 * with the lesson's configuration, the statement the text makes; `lessons.test.ts` runs them all.
 */
export interface Lesson {
  id: string;
  title: string;
  config: Partial<RlLabConfig>;
  scrollTo: string;
  text: string;
  claim: { episodes: number; check: (session: RlSession) => boolean };
}

const rows = (id: string): string[] => [...(GRID_PRESETS[id] as { rows: string[] }).rows];

/** A sibling session with the same settings but some changes, trained for `episodes`. */
function sibling(s: RlSession, patch: Partial<RlLabConfig>, episodes: number): RlSession {
  const other = new RlSession({ ...DEFAULT_RL_LAB, ...s.config, ...patch });
  other.trainEpisodes(episodes);
  return other;
}

/** Highest row (smallest index) the greedy route reaches: the cliff is on the last row. */
const topRow = (s: RlSession): number =>
  Math.min(...s.greedyPath().cells.map((c) => Math.floor(c / s.world.layout.width)));

export const LESSONS: readonly Lesson[] = [
  {
    id: 'qlearning-cliff',
    title: 'Q-learning on the cliff',
    config: { grid: rows('cliff'), algorithm: 'qlearning', slip: 0, gamma: 0.95, alpha: 0.5 },
    scrollTo: 'rl-world',
    text: 'Q-learning learns the value of the best action, so its final route is the shortest one: right along the edge of the cliff. But while it is still exploring it keeps stepping off, and its returns during training are poor.',
    claim: {
      episodes: 1500,
      check: (s) => {
        const path = s.greedyPath();
        const sarsa = sibling(s, { algorithm: 'sarsa' }, 1500);
        return (
          path.outcome === 'goal' &&
          path.cells.length - 1 === 13 &&
          topRow(s) === 2 &&
          sarsa.recentReturn(100) > s.recentReturn(100) + 5
        );
      },
    },
  },
  {
    id: 'sarsa-cliff',
    title: 'SARSA takes the safe road',
    config: { grid: rows('cliff'), algorithm: 'sarsa', slip: 0, gamma: 0.95, alpha: 0.5 },
    scrollTo: 'rl-world',
    text: 'SARSA learns the value of what it actually does, exploratory slips included. Near the cliff those slips are costly, so it prefers the longer route far from the edge: worse on paper, better in practice.',
    claim: {
      episodes: 1500,
      check: (s) => {
        const path = s.greedyPath();
        return path.outcome === 'goal' && path.cells.length - 1 > 13 && topRow(s) <= 1;
      },
    },
  },
  {
    id: 'myopic',
    title: 'An agent with no foresight',
    config: { grid: rows('open'), algorithm: 'qlearning', gamma: 0, slip: 0 },
    scrollTo: 'rl-train',
    text: 'With a discount of 0 the agent only cares about the very next reward. Far from the goal every move looks equally bad, so it never learns a route: its returns stay terrible. Raise the discount in section 03.',
    claim: {
      episodes: 600,
      check: (s) =>
        s.greedyPath().outcome !== 'goal' &&
        s.recentReturn(100) < -30 &&
        sibling(s, { gamma: 0.95 }, 600).recentReturn(100) > -3,
    },
  },
  {
    id: 'slow',
    title: 'Learning too slowly',
    config: { grid: rows('open'), algorithm: 'qlearning', alpha: 0.01, gamma: 0.95, slip: 0 },
    scrollTo: 'rl-train',
    text: 'A learning rate of 0.01 changes each value by 1% of the error per visit, so after hundreds of episodes the table is still mostly wrong. With 0.5 the same agent is nearly perfect.',
    claim: {
      episodes: 300,
      check: (s) => s.agreement() < 0.85 && sibling(s, { alpha: 0.5 }, 300).agreement() > 0.9,
    },
  },
  {
    id: 'maze',
    title: 'Finding a way through a maze',
    config: { grid: rows('maze'), algorithm: 'qlearning', alpha: 0.5, gamma: 0.95, slip: 0 },
    scrollTo: 'rl-world',
    text: 'The agent knows nothing about walls or where the goal is. By trying moves and seeing what happens it works out, cell by cell, which way leads out: watch the arrows turn to follow the corridors.',
    claim: {
      episodes: 2000,
      check: (s) => s.greedyPath().outcome === 'goal' && s.agreement() > 0.9,
    },
  },
];
