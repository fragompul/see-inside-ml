import type { ParamSpec } from '@/shared/lab-kit';
import { POSITIONS, runMcts, type MctsRun, type MctsScenario, type PositionId } from './mctsCore';

export const mctsParams: ParamSpec[] = [
  {
    key: 'position',
    label: 'Position to search',
    kind: 'select',
    options: (Object.keys(POSITIONS) as PositionId[]).map((id) => ({
      value: id,
      label: POSITIONS[id].label,
    })),
  },
  {
    key: 'simulations',
    label: 'Simulated games per move',
    kind: 'ladder',
    values: [3, 10, 30, 100, 400, 1500],
    format: (v) => String(v),
  },
  {
    key: 'exploration',
    label: 'Exploration constant c',
    kind: 'ladder',
    values: [0, 0.3, 1.4, 5],
    format: (v) => String(v),
    hint: 'How strongly the search is drawn to moves it has tried little. 0 = pure greed.',
  },
];

export interface MctsExperiment {
  id: string;
  title: string;
  text: string;
  scenario: Partial<MctsScenario>;
  claim: (run: MctsRun) => boolean;
}

const same = (r: MctsRun, over: Partial<MctsScenario>) => runMcts({ ...r.scenario, ...over });

export const mctsExperiments: MctsExperiment[] = [
  {
    id: 'win',
    title: 'Seeing a win',
    text: 'With 30 random playouts per move the search always finds the winning move; with 3 it picks it in about one case out of five. A random game from a position where a win is available rarely finds it, so it takes some repetitions for the evidence to show.',
    scenario: { position: 'win', simulations: 30 },
    claim: (r) => r.reliability === 1 && same(r, { simulations: 3 }).reliability < 0.5,
  },
  {
    id: 'block',
    title: 'Seeing a threat',
    text: 'Blocking is harder to see than winning, because the evidence is a lot of lost playouts for every other move. It takes about a hundred simulations to find the only move reliably.',
    scenario: { position: 'block', simulations: 100 },
    claim: (r) => r.reliability === 1 && same(r, { simulations: 10 }).reliability < 0.5,
  },
  {
    id: 'trap',
    title: 'A trap that fools short searches',
    text: 'After X takes two opposite corners, only an edge move saves O. Random playouts make corners look good, so short searches walk into the trap about a third of the time. With hundreds of simulations the statistics become reliable.',
    scenario: { position: 'fork', simulations: 100 },
    claim: (r) => r.reliability < 0.8 && same(r, { simulations: 1500 }).reliability === 1,
  },
  {
    id: 'weak',
    title: 'A handful of simulations',
    text: 'With 3 simulations per move the player is hardly better than random: it loses about as many games as it wins against random play, and a perfect opponent beats it almost every time.',
    scenario: { simulations: 3 },
    claim: (r) => r.vsRandom.losses >= 10 && r.vsPerfect.losses >= 17,
  },
  {
    id: 'strong',
    title: 'Never losing to random play',
    text: 'With 30 or more simulations the search does not lose a single game to a random opponent, whichever side it plays, and wins nearly all of them.',
    scenario: { simulations: 100 },
    claim: (r) => r.vsRandom.losses === 0 && r.vsRandom.wins >= 34,
  },
  {
    id: 'perfect',
    title: 'Holding a perfect player to a draw',
    text: 'Noughts and crosses is a draw with perfect play. Given 1,500 simulations per move the search, playing second, draws every game against a perfect opponent; with 100 it loses most of them.',
    scenario: { simulations: 1500 },
    claim: (r) =>
      r.vsPerfect.losses === 0 &&
      r.vsPerfect.draws === 20 &&
      same(r, { simulations: 100 }).vsPerfect.losses > 10,
  },
  {
    id: 'exploration',
    title: 'Some exploration helps',
    text: 'With c = 0 the search commits to the first move that looks good and never revisits the others; a small c = 0.3 spreads its attention and halves the number of games lost to the perfect player at 100 simulations.',
    scenario: { simulations: 100, exploration: 0.3 },
    claim: (r) => r.vsPerfect.losses < same(r, { exploration: 0 }).vsPerfect.losses * 0.7,
  },
];
