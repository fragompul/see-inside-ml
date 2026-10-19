import { BANDIT_STRATEGIES, type BanditStrategy } from '@sim/deep';
import { useMemo, useState } from 'react';
import { Badge, Panel, Segmented, Switch } from '@/design-system';
import { usePalette } from '@/shared/canvas/colors';
import { ExperimentBar } from '@/shared/components/ExperimentBar';
import { MiniChart } from '@/shared/components/MiniChart';
import {
  LabHeader,
  MetricsStrip,
  ParamPanel,
  StageNav,
  fix,
  type Metric,
  type ParamValue,
  type ParamValues,
} from '@/shared/lab-kit';
import {
  DEFAULT_BANDITS,
  PROBLEMS,
  RUNS,
  STRATEGY_LABELS,
  STRATEGY_NOTES,
  runBandits,
  type BanditsRun,
  type BanditScenario,
} from './banditsCore';
import { banditExperiments, banditParams, type BanditExperiment } from './banditsSpec';

const toScenario = (p: ParamValues, strategies: BanditStrategy[]): BanditScenario => ({
  ...DEFAULT_BANDITS,
  problem: p.problem as BanditScenario['problem'],
  steps: Number(p.steps),
  epsilon: Number(p.epsilon),
  strategies,
});

const fromScenario = (s: BanditScenario): ParamValues => ({
  problem: s.problem,
  steps: s.steps,
  epsilon: s.epsilon,
});

function Detail({ run }: { run: BanditsRun }) {
  const palette = usePalette();
  const { means } = run.problem;
  const best = means.indexOf(Math.max(...means));
  const maxCount = Math.max(...run.single.counts);
  const top = Math.max(...means, ...run.single.estimates);
  const bottom = Math.min(0, ...means, ...run.single.estimates);
  return (
    <div className="space-y-4">
      <div className="overflow-x-auto">
        <div
          className="flex min-w-[24rem] items-end gap-3"
          style={{ height: 190 }}
          role="img"
          aria-label="How often each machine was pulled in this run, with its estimated and true payout"
        >
          {means.map((m, i) => {
            const c = run.single.counts[i] as number;
            const e = run.single.estimates[i] as number;
            const pos = (v: number) => `${((v - bottom) / (top - bottom || 1)) * 100}%`;
            return (
              <div key={i} className="flex h-full flex-1 flex-col items-center justify-end gap-1">
                <div className="relative w-full flex-1">
                  <div
                    className="absolute inset-x-1 bottom-0 rounded-t"
                    style={{
                      height: `${(c / maxCount) * 100}%`,
                      background: i === best ? palette.css.pos : palette.css.brand,
                      opacity: 0.55,
                    }}
                    title={`${c} pulls`}
                  />
                  <div
                    className="absolute inset-x-0 h-0.5"
                    style={{ bottom: pos(m), background: palette.css.fg }}
                    title={`true payout ${m}`}
                  />
                  <div
                    className="absolute left-1/2 size-2.5 -translate-x-1/2 translate-y-1/2 rounded-full"
                    style={{ bottom: pos(e), background: palette.css.neg }}
                    title={`estimate ${fix(e, 2)}`}
                  />
                </div>
                <span className="font-mono text-[0.68rem] text-faint">{c}</span>
                <span className="text-[0.72rem] text-muted">#{i + 1}</span>
              </div>
            );
          })}
        </div>
      </div>
      <p className="text-sm text-muted" data-testid="detail-note">
        Bars: how many times each machine was pulled in this one run (green = the best machine). The
        black line is each machine&apos;s true average payout and the red dot what the strategy
        believes it is. A good strategy ends with a tall green bar and red dots near the lines of
        the machines it has played often; the ones it rarely played can stay badly wrong, and that
        is the price of not wasting pulls on them.
      </p>
    </div>
  );
}

export default function BanditsPage() {
  const palette = usePalette();
  const [params, setParams] = useState<ParamValues>(() => fromScenario(DEFAULT_BANDITS));
  const [strategies, setStrategies] = useState<BanditStrategy[]>([...BANDIT_STRATEGIES]);
  const [detail, setDetail] = useState<BanditStrategy>('thompson');
  const [activeId, setActiveId] = useState<string | null>(null);
  const scenario = useMemo(() => toScenario(params, strategies), [params, strategies]);
  const run = useMemo(() => runBandits(scenario, detail), [scenario, detail]);
  const setParam = (key: string, value: ParamValue) => setParams((p) => ({ ...p, [key]: value }));

  const apply = (exp: BanditExperiment) => {
    setParams(fromScenario({ ...DEFAULT_BANDITS, ...exp.scenario }));
    setStrategies([...BANDIT_STRATEGIES]);
    setActiveId(exp.id);
  };
  const stillActive = (() => {
    const exp = banditExperiments.find((e) => e.id === activeId);
    return (
      !!exp &&
      strategies.length === BANDIT_STRATEGIES.length &&
      JSON.stringify(fromScenario({ ...DEFAULT_BANDITS, ...exp.scenario })) ===
        JSON.stringify(params)
    );
  })();
  const toggle = (st: BanditStrategy) =>
    setStrategies((cur) =>
      cur.includes(st)
        ? cur.filter((x) => x !== st)
        : BANDIT_STRATEGIES.filter((x) => x === st || cur.includes(x)),
    );

  const ranked = [...run.race].sort((a, b) => a.final - b.final);
  const metrics: Metric[] = [
    { label: 'Best strategy here', value: ranked[0] ? STRATEGY_LABELS[ranked[0].strategy] : '–' },
    {
      label: 'Its average regret',
      value: ranked[0] ? fix(ranked[0].final, 1) : '–',
      testId: 'best-regret',
      hint: `total reward lost against always pulling the best machine, averaged over ${RUNS} runs`,
    },
    {
      label: 'Worst strategy',
      value: ranked.length ? STRATEGY_LABELS[ranked[ranked.length - 1]!.strategy] : '–',
    },
  ];

  const stages: [string, string][] = [
    ['lab-regret', 'Regret'],
    ['lab-board', 'Scoreboard'],
    ['lab-detail', 'One run in detail'],
    ['lab-how', 'How it works'],
    ['lab-controls', 'Settings'],
  ];

  return (
    <div className="mx-auto max-w-6xl px-4 pt-8 pb-20 sm:px-6 lg:px-8">
      <LabHeader
        eyebrow="Reinforcement learning"
        title="Multi-armed bandits"
        accent="explore or exploit?"
        blurb="Several slot machines pay at unknown rates. Every pull both earns a reward and teaches you something, so you must balance trying what you know is good against finding out whether something is better. Eight strategies, compared by the reward they lose."
        badges={
          <>
            <Badge>{PROBLEMS[scenario.problem].label}</Badge>
            <Badge>{scenario.steps} pulls</Badge>
            <Badge>{RUNS} runs averaged</Badge>
          </>
        }
      />
      <StageNav stages={stages} />
      <ExperimentBar
        lessons={banditExperiments}
        activeId={stillActive ? activeId : null}
        onApply={apply}
      />

      <div className="flex flex-col gap-5">
        <Panel id="lab-regret" className="scroll-mt-24" title="Regret" eyebrow="01">
          <div className="space-y-6">
            <p className="text-[0.8rem] text-muted">{PROBLEMS[scenario.problem].note}</p>
            <MetricsStrip metrics={metrics} className="lg:grid-cols-3" />
            <div className="grid gap-8 lg:grid-cols-[minmax(0,1fr)_15rem]">
              {run.race.length > 0 ? (
                <MiniChart
                  title="Average cumulative regret"
                  x={run.race[0]!.curve.map((_, i) => i + 1)}
                  xLabel="pull"
                  series={run.race.map((r) => ({
                    name: STRATEGY_LABELS[r.strategy],
                    color: palette.css.classes[
                      BANDIT_STRATEGIES.indexOf(r.strategy) % palette.css.classes.length
                    ] as string,
                    values: r.curve,
                  }))}
                  height={260}
                />
              ) : (
                <p className="text-sm text-muted">Switch on at least one strategy.</p>
              )}
              <div className="space-y-3">
                <p className="eyebrow">Strategies</p>
                {BANDIT_STRATEGIES.map((st, i) => (
                  <div key={st} className="space-y-0.5">
                    <div className="flex items-center gap-2">
                      <span
                        className="inline-block size-2.5 rounded-sm"
                        style={{ background: palette.css.classes[i % palette.css.classes.length] }}
                      />
                      <div className="flex-1">
                        <Switch
                          label={STRATEGY_LABELS[st]}
                          checked={strategies.includes(st)}
                          onCheckedChange={() => toggle(st)}
                        />
                      </div>
                    </div>
                    <p className="pl-[1.15rem] text-[0.68rem] text-faint">{STRATEGY_NOTES[st]}</p>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </Panel>
        <Panel id="lab-board" className="scroll-mt-24" title="Scoreboard" eyebrow="02">
          <div className="space-y-2">
            {ranked.map((r) => {
              const max = Math.max(...ranked.map((x) => x.final), 1);
              return (
                <div
                  key={r.strategy}
                  className="grid grid-cols-[10rem_minmax(0,1fr)_5.5rem] items-center gap-3 text-[0.82rem]"
                >
                  <span>{STRATEGY_LABELS[r.strategy]}</span>
                  <div className="h-3 rounded-full bg-raised">
                    <div
                      className="h-3 rounded-full"
                      style={{
                        width: `${(r.final / max) * 100}%`,
                        background:
                          palette.css.classes[
                            BANDIT_STRATEGIES.indexOf(r.strategy) % palette.css.classes.length
                          ],
                        opacity: 0.85,
                      }}
                    />
                  </div>
                  <span
                    className="text-right font-mono text-faint"
                    data-testid={`final-${r.strategy}`}
                  >
                    {fix(r.final, 1)}
                  </span>
                </div>
              );
            })}
            <p className="pt-2 text-sm text-muted">
              Final average regret (shorter is better). A curve that flattens means the strategy has
              found the best machine and stopped losing; a straight rising line means it keeps
              paying a constant price per pull, as a strategy that explores at random for ever must.
            </p>
          </div>
        </Panel>
        <Panel id="lab-detail" className="scroll-mt-24" title="One run in detail" eyebrow="03">
          <div className="space-y-5">
            <Segmented<BanditStrategy>
              label="Strategy"
              size="sm"
              fit
              value={detail}
              onValueChange={setDetail}
              options={BANDIT_STRATEGIES.map((s) => ({ value: s, label: STRATEGY_LABELS[s] }))}
            />
            <Detail run={run} />
          </div>
        </Panel>
        <Panel id="lab-how" className="scroll-mt-24" title="How it works" eyebrow="04">
          <div className="space-y-3 text-sm text-muted">
            <p>
              <strong className="text-fg">Regret</strong> is the reward you would have gained by
              always pulling the best machine, minus what you got. Any strategy must pull inferior
              machines sometimes to find out they are inferior; the question is how fast regret
              grows. A strategy that explores at a constant rate (ε-greedy) has regret growing
              linearly; good strategies make it grow like the logarithm of the number of pulls.
            </p>
            <p>
              <strong className="text-fg">UCB</strong> adds an uncertainty bonus proportional to the
              square root of log t over the number of pulls of an arm.{' '}
              <strong className="text-fg">Thompson sampling</strong> keeps a posterior over each
              arm&apos;s payout (a Beta distribution for coins) and pulls the arm with the best
              sample, so arms are tried in proportion to the probability that they are the best. The{' '}
              <strong className="text-fg">gradient bandit</strong> keeps a preference per arm and
              follows the gradient of the expected reward: exactly the REINFORCE rule of policy
              gradients, applied to a one-step problem.
            </p>
          </div>
        </Panel>
        <Panel id="lab-controls" className="scroll-mt-24" title="Settings" eyebrow="05">
          <ParamPanel specs={banditParams} values={params} onChange={setParam} columns={3} />
        </Panel>
      </div>
    </div>
  );
}
