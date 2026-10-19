import { turnOf } from '@sim/deep';
import { useEffect, useMemo, useState } from 'react';
import { Badge, Panel } from '@/design-system';
import { usePalette } from '@/shared/canvas/colors';
import { ExperimentBar } from '@/shared/components/ExperimentBar';
import {
  LabHeader,
  MetricsStrip,
  ParamPanel,
  StageNav,
  pct,
  type Metric,
  type ParamValue,
  type ParamValues,
} from '@/shared/lab-kit';
import {
  DEFAULT_MCTS,
  POSITIONS,
  runMcts,
  type MctsRun,
  type MctsScenario,
  type Tournament,
} from './mctsCore';
import { mctsExperiments, mctsParams, type MctsExperiment } from './mctsSpec';

const toScenario = (p: ParamValues): MctsScenario => ({
  ...DEFAULT_MCTS,
  position: p.position as MctsScenario['position'],
  simulations: Number(p.simulations),
  exploration: Number(p.exploration),
});

const fromScenario = (s: MctsScenario): ParamValues => ({
  position: s.position,
  simulations: s.simulations,
  exploration: s.exploration,
});

function BoardView({ run }: { run: MctsRun }) {
  const palette = usePalette();
  const board = POSITIONS[run.scenario.position].board;
  const cell = 110;
  const maxVisits = Math.max(...run.result.root.map((r) => r.visits), 1);
  const turn = turnOf(board);
  return (
    <svg
      viewBox={`0 0 ${cell * 3} ${cell * 3}`}
      className="w-full max-w-sm rounded-xl border border-line"
      role="img"
      aria-label="The position, with how often the search tried each move"
    >
      {board.map((c, i) => {
        const x = (i % 3) * cell;
        const y = Math.floor(i / 3) * cell;
        const stat = run.result.root.find((r) => r.move === i);
        const optimal = run.optimal.includes(i) && c === 0;
        return (
          <g key={i}>
            <rect
              x={x}
              y={y}
              width={cell}
              height={cell}
              fill="none"
              stroke={palette.css.muted}
              strokeOpacity={0.35}
            />
            {c === 0 && stat && (
              <>
                <rect
                  x={x + 8}
                  y={y + cell - 26}
                  width={((cell - 16) * stat.visits) / maxVisits}
                  height={8}
                  rx={4}
                  fill={palette.css.brand}
                  opacity={0.8}
                />
                <text
                  x={x + cell / 2}
                  y={y + cell / 2 - 4}
                  textAnchor="middle"
                  className="fill-fg font-mono text-[15px]"
                >
                  {pct(stat.winRate)}
                </text>
                <text
                  x={x + cell / 2}
                  y={y + cell / 2 + 14}
                  textAnchor="middle"
                  className="fill-faint font-mono text-[10px]"
                >
                  {stat.visits} tries
                </text>
              </>
            )}
            {optimal && <circle cx={x + 14} cy={y + 14} r={5} fill={palette.css.pos} />}
            {run.result.move === i && c === 0 && (
              <rect
                x={x + 2}
                y={y + 2}
                width={cell - 4}
                height={cell - 4}
                fill="none"
                stroke={palette.css.neg}
                strokeWidth={2.4}
                rx={6}
              />
            )}
            {c === 1 && (
              <g stroke={palette.css.fg} strokeWidth={5} strokeLinecap="round">
                <line x1={x + 28} y1={y + 28} x2={x + cell - 28} y2={y + cell - 28} />
                <line x1={x + cell - 28} y1={y + 28} x2={x + 28} y2={y + cell - 28} />
              </g>
            )}
            {c === 2 && (
              <circle
                cx={x + cell / 2}
                cy={y + cell / 2}
                r={cell / 2 - 28}
                fill="none"
                stroke={palette.css.brand}
                strokeWidth={5}
              />
            )}
          </g>
        );
      })}
      <text x={6} y={cell * 3 - 6} className="fill-faint text-[10px]">
        {turn === 1 ? 'X' : 'O'} to move
      </text>
    </svg>
  );
}

function Score({ title, t, games }: { title: string; t: Tournament; games: number }) {
  return (
    <div className="space-y-1.5">
      <p className="eyebrow">{title}</p>
      <div
        className="flex h-5 overflow-hidden rounded-md border border-line text-[0.68rem]"
        role="img"
        aria-label={`${t.wins} wins, ${t.draws} draws, ${t.losses} losses`}
      >
        <div
          className="bg-pos/60 text-center text-[#0a0a14]"
          style={{ width: `${(t.wins / games) * 100}%` }}
        >
          {t.wins || ''}
        </div>
        <div
          className="bg-muted/40 text-center text-fg"
          style={{ width: `${(t.draws / games) * 100}%` }}
        >
          {t.draws || ''}
        </div>
        <div
          className="bg-neg/60 text-center text-[#0a0a14]"
          style={{ width: `${(t.losses / games) * 100}%` }}
        >
          {t.losses || ''}
        </div>
      </div>
      <p className="font-mono text-[0.72rem] text-faint">
        {t.wins} won · {t.draws} drawn · {t.losses} lost
      </p>
    </div>
  );
}

export default function MctsPage() {
  const [params, setParams] = useState<ParamValues>(() => fromScenario(DEFAULT_MCTS));
  const [activeId, setActiveId] = useState<string | null>(null);
  const [run, setRun] = useState<MctsRun | null>(null);
  const scenario = useMemo(() => toScenario(params), [params]);

  useEffect(() => {
    const timer = window.setTimeout(() => setRun(runMcts(scenario)), 30);
    return () => window.clearTimeout(timer);
  }, [scenario]);

  const setParam = (key: string, value: ParamValue) => setParams((p) => ({ ...p, [key]: value }));
  const apply = (exp: MctsExperiment) => {
    setParams(fromScenario({ ...DEFAULT_MCTS, ...exp.scenario }));
    setActiveId(exp.id);
  };
  const stillActive = (() => {
    const exp = mctsExperiments.find((e) => e.id === activeId);
    return (
      !!exp &&
      JSON.stringify(fromScenario({ ...DEFAULT_MCTS, ...exp.scenario })) === JSON.stringify(params)
    );
  })();

  const metrics: Metric[] = run
    ? [
        {
          label: 'Move chosen',
          value: String(run.result.move + 1),
          hint: 'cell number, 1 to 9, reading left to right then down',
        },
        {
          label: 'Is it optimal?',
          value: run.optimal.includes(run.result.move) ? 'yes' : 'no',
          tone: run.optimal.includes(run.result.move) ? 'good' : 'bad',
        },
        {
          label: 'Searches that pick an optimal move',
          value: pct(run.reliability),
          testId: 'reliability',
          hint: 'out of 20 searches with different random numbers',
        },
      ]
    : [];

  const stages: [string, string][] = [
    ['lab-board', 'The search'],
    ['lab-games', 'Playing real games'],
    ['lab-how', 'How it works'],
    ['lab-controls', 'Settings'],
  ];

  return (
    <div className="mx-auto max-w-6xl px-4 pt-8 pb-20 sm:px-6 lg:px-8">
      <LabHeader
        eyebrow="Reinforcement learning"
        title="Monte Carlo tree search"
        accent="deciding by playing it out"
        blurb="No evaluation function and no rules of thumb: just play many random games from each candidate move and prefer the ones that win more, while still giving the others a chance. This is the idea behind the programs that mastered Go."
        badges={
          <>
            <Badge>{POSITIONS[scenario.position].label}</Badge>
            <Badge>{scenario.simulations} playouts per move</Badge>
          </>
        }
      />
      <StageNav stages={stages} />
      <ExperimentBar
        lessons={mctsExperiments}
        activeId={stillActive ? activeId : null}
        onApply={apply}
      />

      <div className="flex flex-col gap-5">
        <Panel id="lab-board" className="scroll-mt-24" title="The search" eyebrow="01">
          {run ? (
            <div className="space-y-6">
              <p className="text-[0.8rem] text-muted">{POSITIONS[scenario.position].note}</p>
              <MetricsStrip metrics={metrics} className="lg:grid-cols-3" />
              <div className="grid gap-8 lg:grid-cols-[auto_minmax(0,1fr)]">
                <BoardView run={run} />
                <p className="text-sm text-muted" data-testid="board-note">
                  Each empty cell shows the share of random games from that move that the player to
                  move won or drew (a draw counts as half) and how many times the search tried it
                  (the bar). The <span className="text-pos">green dot</span> marks moves that are
                  optimal with perfect play; the <span className="text-neg">red frame</span> is the
                  move the search picks: the most-tried one. A good search puts most of its tries on
                  the right move.
                </p>
              </div>
            </div>
          ) : (
            <p className="text-sm text-muted">Searching…</p>
          )}
        </Panel>
        <Panel id="lab-games" className="scroll-mt-24" title="Playing real games" eyebrow="02">
          {run && (
            <div className="space-y-5">
              <div className="grid gap-6 sm:grid-cols-2">
                <Score
                  title="Against random play (20 games each side)"
                  t={run.vsRandom}
                  games={40}
                />
                <Score title="As O against a perfect X (20 games)" t={run.vsPerfect} games={20} />
              </div>
              <p className="text-sm text-muted">
                The search can only be as good as its number of playouts allows. A perfect opponent
                never loses, so the best the search can do is to draw; every loss is a mistake of
                the search, which becomes rarer as playouts increase.
              </p>
            </div>
          )}
        </Panel>
        <Panel id="lab-how" className="scroll-mt-24" title="How it works" eyebrow="03">
          <div className="space-y-3 text-sm text-muted">
            <p>
              Each simulation has four steps. <strong className="text-fg">Select</strong>: starting
              at the current position, descend through the tree by repeatedly taking the child with
              the highest <em>UCB</em> score, its win rate plus c·√(ln N / n), where N and n are the
              visit counts of the parent and the child. <strong className="text-fg">Expand</strong>:
              add one untried move. <strong className="text-fg">Roll out</strong>: play random moves
              to the end. <strong className="text-fg">Back up</strong>: add the result to every
              position on the path. After the budget is spent the most-visited move is played.
            </p>
            <p>
              The exploration constant c balances trying the best-looking move against giving the
              others a chance to prove themselves. The tests check the game logic, minimax (perfect
              play) and that more playouts play stronger.
            </p>
          </div>
        </Panel>
        <Panel id="lab-controls" className="scroll-mt-24" title="Settings" eyebrow="04">
          <ParamPanel specs={mctsParams} values={params} onChange={setParam} columns={3} />
        </Panel>
      </div>
    </div>
  );
}
