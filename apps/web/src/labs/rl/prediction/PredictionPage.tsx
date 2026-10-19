import { PREDICTION_METHODS } from '@sim/deep';
import { useMemo, useState } from 'react';
import { Badge, Panel } from '@/design-system';
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
  DEFAULT_PREDICTION,
  METHOD_LABELS,
  RUNS,
  runPrediction,
  type PredictionScenario,
} from './predictionCore';
import {
  predictionExperiments,
  predictionParams,
  type PredictionExperiment,
} from './predictionSpec';

const toScenario = (p: ParamValues): PredictionScenario => ({
  ...DEFAULT_PREDICTION,
  states: Number(p.states),
  alpha: Number(p.alpha),
  n: Number(p.n),
  lambda: Number(p.lambda),
  episodes: Number(p.episodes),
});

const fromScenario = (s: PredictionScenario): ParamValues => ({
  states: s.states,
  alpha: s.alpha,
  n: s.n,
  lambda: s.lambda,
  episodes: s.episodes,
});

export default function PredictionPage() {
  const palette = usePalette();
  const [params, setParams] = useState<ParamValues>(() => fromScenario(DEFAULT_PREDICTION));
  const [activeId, setActiveId] = useState<string | null>(null);
  const scenario = useMemo(() => toScenario(params), [params]);
  const run = useMemo(() => runPrediction(scenario), [scenario]);
  const setParam = (key: string, value: ParamValue) => setParams((p) => ({ ...p, [key]: value }));

  const apply = (exp: PredictionExperiment) => {
    setParams(fromScenario({ ...DEFAULT_PREDICTION, ...exp.scenario }));
    setActiveId(exp.id);
  };
  const stillActive = (() => {
    const exp = predictionExperiments.find((e) => e.id === activeId);
    return (
      !!exp &&
      JSON.stringify(fromScenario({ ...DEFAULT_PREDICTION, ...exp.scenario })) ===
        JSON.stringify(params)
    );
  })();

  const ranked = [...PREDICTION_METHODS].sort((a, b) => run.area[a] - run.area[b]);
  const metrics: Metric[] = [
    { label: 'Lowest average error', value: METHOD_LABELS[ranked[0]!] },
    { label: 'Its error', value: fix(run.area[ranked[0]!], 3), testId: 'best-error' },
    { label: 'Highest average error', value: METHOD_LABELS[ranked[ranked.length - 1]!] },
  ];
  const colors = palette.css.classes;
  const stages: [string, string][] = [
    ['lab-curves', 'Error while learning'],
    ['lab-values', 'What was learned'],
    ['lab-how', 'How it works'],
    ['lab-controls', 'Settings'],
  ];

  return (
    <div className="mx-auto max-w-6xl px-4 pt-8 pb-20 sm:px-6 lg:px-8">
      <LabHeader
        eyebrow="Reinforcement learning"
        title="Monte Carlo and temporal-difference learning"
        accent="learning the value of a state"
        blurb="A random walk along a row of states ends at either side, with a reward of one on the right. How good is each state? Monte Carlo waits for the end of the walk, TD learns at every step, and n-step TD and TD(λ) look in between. The true answer is known, so each method's error can be read off."
        badges={
          <>
            <Badge>{scenario.states} states</Badge>
            <Badge>α = {scenario.alpha}</Badge>
            <Badge>{RUNS} runs averaged</Badge>
          </>
        }
      />
      <StageNav stages={stages} />
      <ExperimentBar
        lessons={predictionExperiments}
        activeId={stillActive ? activeId : null}
        onApply={apply}
      />

      <div className="flex flex-col gap-5">
        <Panel id="lab-curves" className="scroll-mt-24" title="Error while learning" eyebrow="01">
          <div className="space-y-6">
            <MetricsStrip metrics={metrics} className="lg:grid-cols-3" />
            <MiniChart
              title="Root-mean-square error of the value estimates, averaged over runs"
              x={run.curves.mc.map((_, i) => i)}
              xLabel="episode"
              series={PREDICTION_METHODS.map((m, i) => ({
                name: METHOD_LABELS[m],
                color: colors[i % colors.length] as string,
                values: run.curves[m],
              }))}
              height={240}
            />
          </div>
        </Panel>
        <Panel id="lab-values" className="scroll-mt-24" title="What was learned" eyebrow="02">
          <div className="space-y-3">
            <MiniChart
              title="Learned value of each state after the last episode (one run)"
              x={run.truth.map((_, i) => i + 1)}
              xLabel="state"
              yRange={{ lo: 0, hi: 1 }}
              series={[
                { name: 'true value', color: palette.css.fg, values: run.truth, dashed: true },
                ...PREDICTION_METHODS.map((m, i) => ({
                  name: METHOD_LABELS[m],
                  color: colors[i % colors.length] as string,
                  values: run.values[m],
                })),
              ]}
              height={220}
            />
            <p className="text-sm text-muted">
              The true value of a state is the chance that the walk ends on the right, which grows
              evenly from left to right. Monte Carlo estimates are jagged (each is an average of a
              few noisy outcomes); TD estimates are smoother because neighbouring states keep
              pulling on each other.
            </p>
          </div>
        </Panel>
        <Panel id="lab-how" className="scroll-mt-24" title="How it works" eyebrow="03">
          <div className="space-y-3 text-sm text-muted">
            <p>
              <strong className="text-fg">Monte Carlo</strong>: V(s) ← V(s) + α (G − V(s)), where G
              is the return that actually followed. <strong className="text-fg">TD(0)</strong>: V(s)
              ← V(s) + α (r + V(s′) − V(s)): the target is one real step plus the current estimate
              of the next state, so learning happens during the episode but starts from guesses
              (bias). <strong className="text-fg">n-step TD</strong> uses n real rewards before
              bootstrapping. <strong className="text-fg">TD(λ)</strong> averages all of these with
              weights λ^k and is implemented with <em>eligibility traces</em>: every state keeps a
              fading memory of having been visited and shares in each TD error accordingly.
            </p>
            <p>
              The trade-off is bias against variance: the more real rewards in the target, the less
              bias and the more noise. The tests check that all methods converge to the exact values
              i/(n+1), that n-step TD with a huge n is Monte Carlo, and that λ = 0 is TD(0).
            </p>
          </div>
        </Panel>
        <Panel id="lab-controls" className="scroll-mt-24" title="Settings" eyebrow="04">
          <ParamPanel specs={predictionParams} values={params} onChange={setParam} columns={3} />
        </Panel>
      </div>
    </div>
  );
}
