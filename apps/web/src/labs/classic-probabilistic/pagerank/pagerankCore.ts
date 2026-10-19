import {
  convergenceRate,
  googleMatrix,
  pageRank,
  pageRankSteps,
  simulateVisits,
  totalVariation,
  type LinkGraph,
} from '@sim/classic';

export type WebId = 'popular' | 'trap' | 'deadend';

export interface Web {
  id: WebId;
  label: string;
  note: string;
  links: LinkGraph;
}

/** Eight pages; `links[i]` are the pages that page i links to. */
export const WEBS: Record<WebId, Web> = {
  popular: {
    id: 'popular',
    label: 'A small web',
    note: 'Most pages link to page C, and C links back to A: a little hub that everyone cites.',
    links: [[1, 2], [2], [0], [2, 4], [2, 5], [2, 6], [2, 7], [2]],
  },
  trap: {
    id: 'trap',
    label: 'A link farm',
    note: 'Pages E to H link only to each other (and E collects all of it): a spider trap that soaks up rank. One honest page, D, happens to link in.',
    links: [[1], [2], [3], [0, 4], [5, 6, 7], [4], [4], [4]],
  },
  deadend: {
    id: 'deadend',
    label: 'A dead end',
    note: 'Page D links nowhere. A surfer reaching it must teleport, so it neither hoards nor loses rank.',
    links: [[1, 2], [3], [3], [], [3, 5], [4], [5, 3], [6]],
  },
};

export const PAGE_NAMES = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'];

export interface PageRankScenario {
  web: WebId;
  /** Chance of following a link instead of teleporting. */
  damping: number;
  /** Power-iteration steps shown. */
  iterations: number;
  /** Length of the random-surfer simulation. */
  surfers: number;
  seed: number;
}

export const DEFAULT_PAGERANK: PageRankScenario = {
  web: 'popular',
  damping: 0.85,
  iterations: 10,
  surfers: 5000,
  seed: 3,
};

export interface PageRankRun {
  scenario: PageRankScenario;
  web: Web;
  exact: number[];
  /** Rank vector after each power-iteration step (entry 0 = uniform start). */
  history: number[][];
  /** The rank vector at the chosen number of iterations. */
  shown: number[];
  /** Total-variation distance to the exact answer after each step (to plot). */
  error: number[];
  /** Steps needed to get within 1e-3 of the exact ranks. */
  stepsToConverge: number;
  /** Per-step shrink rate of the error (second eigenvalue of the Google matrix). */
  rate: number;
  /** Share of time a simulated random surfer spent on each page. */
  surfed: number[];
  /** Distance between the surfer's shares and the exact ranks. */
  surferGap: number;
}

export function runPageRank(s: PageRankScenario): PageRankRun {
  const web = WEBS[s.web];
  const exact = pageRank(web.links, s.damping);
  const horizon = Math.max(s.iterations, 200);
  const history = pageRankSteps(web.links, s.damping, horizon);
  const error = history.map((h) => totalVariation(h, exact));
  const stepsToConverge = error.findIndex((e) => e < 1e-3);
  const matrix = googleMatrix(web.links, s.damping);
  const surfed = simulateVisits(matrix, 0, s.surfers, s.seed);
  return {
    scenario: s,
    web,
    exact,
    history,
    shown: history[s.iterations] as number[],
    error: error.slice(0, Math.max(s.iterations, 40) + 1),
    stepsToConverge: stepsToConverge < 0 ? Infinity : stepsToConverge,
    rate: convergenceRate(matrix),
    surfed,
    surferGap: totalVariation(surfed, exact),
  };
}
