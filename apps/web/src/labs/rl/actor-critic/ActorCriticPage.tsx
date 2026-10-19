import { useMemo, useState } from 'react';
import { Badge, Panel } from '@/design-system';
import { usePalette } from '@/shared/canvas/colors';
import { useCompute } from '@/shared/compute/useCompute';
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
import { DEFAULT_AC, LAMBDAS, SEEDS, type AcRun, type AcScenario } from './acCore';
import { acExperiments, acParams, type AcExperiment } from './acSpec';

const toScenario = (p: ParamValues): AcScenario => ({
  ...DEFAULT_AC,
  layout: p.layout as AcScenario['layout'],
  slip: Number(p.slip),
  lambda: Number(p.lambda),
  actorRate: Number(p.actorRate),
});

const fromScenario = (s: AcScenario): ParamValues => ({
  layout: s.layout,
  slip: s.slip,
  lambda: s.lambda,
  actorRate: s.actorRate,
});

function Results({ run }: { run: Omit<AcRun, 'world' | 'single'> }) {
  const palette = usePalette();
  const colors = [palette.css.neg, palette.css.muted, palette.css.brand, palette.css.pos];
  const metrics: Metric[] = [
    {
      label: 'Return, λ = 0 (A2C)',
      value: fix(run.final['0'] as number, 2),
      testId: 'final-a2c',
      hint: `best possible ${fix(run.optimal, 2)}`,
    },
    { label: 'Return, λ = 0.9', value: fix(run.final['0.9'] as number, 2) },
    { label: 'Return, λ = 1', value: fix(run.final['1'] as number, 2), hint: 'whole returns' },
    {
      label: 'Gradient noise, λ = 1 over λ = 0',
      value: `${fix((run.variance['1'] as number) / (run.variance['0'] as number), 1)}×`,
    },
  ];
  return (
    <>
      <Panel id="lab-race" className="scroll-mt-24" title="The race" eyebrow="01">
        <div className="space-y-6">
          <MetricsStrip metrics={metrics} className="lg:grid-cols-4" />
          <MiniChart
            title={`Average return per episode (moving average over 20, ${SEEDS} runs per curve)`}
            x={run.curves['0']?.map((_, i) => i + 1) ?? []}
            xLabel="episode"
            series={LAMBDAS.map((l, i) => ({
              name: `λ = ${l}`,
              color: colors[i] as string,
              values: run.curves[String(l)] as number[],
            }))}
            height={250}
          />
          <p className="text-sm text-muted">
            Episodes until the average passes 70% of the best possible return:{' '}
            {LAMBDAS.map((l) => {
              const n = run.speed[String(l)] as number;
              return `λ = ${l}: ${Number.isFinite(n) ? n : 'not within the run'}`;
            }).join(', ')}
            .
          </p>
        </div>
      </Panel>
      <Panel id="lab-tradeoff" className="scroll-mt-24" title="Noise against bias" eyebrow="02">
        <div className="grid gap-6 lg:grid-cols-2">
          <MiniChart
            title="Variance of the gradient estimate"
            x={LAMBDAS}
            xLabel="λ"
            series={[
              {
                name: 'variance',
                color: palette.css.neg,
                values: LAMBDAS.map((l) => run.variance[String(l)] as number),
              },
            ]}
            height={200}
          />
          <MiniChart
            title="Distance of its mean from the unbiased estimate (λ = 1)"
            x={LAMBDAS}
            xLabel="λ"
            series={[
              {
                name: 'bias',
                color: palette.css.brand,
                values: LAMBDAS.map((l) => run.bias[String(l)] as number),
              },
            ]}
            height={200}
          />
        </div>
      </Panel>
    </>
  );
}

export default function ActorCriticPage() {
  const [params, setParams] = useState<ParamValues>(() => fromScenario(DEFAULT_AC));
  const [activeId, setActiveId] = useState<string | null>(null);
  const scenario = useMemo(() => toScenario(params), [params]);
  const { result: run } = useCompute('actor-critic', scenario);
  const setParam = (key: string, value: ParamValue) => setParams((p) => ({ ...p, [key]: value }));

  const apply = (exp: AcExperiment) => {
    setParams(fromScenario({ ...DEFAULT_AC, ...exp.scenario }));
    setActiveId(exp.id);
  };
  const stillActive = (() => {
    const exp = acExperiments.find((e) => e.id === activeId);
    return (
      !!exp &&
      JSON.stringify(fromScenario({ ...DEFAULT_AC, ...exp.scenario })) === JSON.stringify(params)
    );
  })();

  const stages: [string, string][] = [
    ['lab-race', 'The race'],
    ['lab-tradeoff', 'Noise against bias'],
    ['lab-how', 'How it works'],
    ['lab-controls', 'Settings'],
  ];

  return (
    <div className="mx-auto max-w-6xl px-4 pt-8 pb-20 sm:px-6 lg:px-8">
      <LabHeader
        eyebrow="Reinforcement learning"
        title="Actor-critic and advantage estimation"
        accent="a critic to cut the noise"
        blurb="A policy-gradient agent learns from how much better an action turned out than expected. Waiting for the whole outcome is unbiased but noisy; asking a learned critic after one step is calm but can be wrong. One dial, λ, slides between the two. See the trade-off, and when each end wins."
        badges={
          <>
            <Badge>λ = {scenario.lambda} in the shown run</Badge>
            <Badge>{SEEDS} runs per curve</Badge>
          </>
        }
      />
      <StageNav stages={stages} />
      <ExperimentBar
        lessons={acExperiments}
        activeId={stillActive ? activeId : null}
        onApply={apply}
      />

      <div className="flex flex-col gap-5">
        {run ? (
          <Results run={run} />
        ) : (
          <Panel id="lab-race" className="scroll-mt-24" title="The race" eyebrow="01">
            <p className="text-sm text-muted" role="status">
              Training the agents…
            </p>
          </Panel>
        )}
        <Panel id="lab-how" className="scroll-mt-24" title="How it works" eyebrow="03">
          <div className="space-y-3 text-sm text-muted">
            <p>
              The actor is a table of preferences for each move in each cell; the critic is a table
              of how good each cell is. After an episode, each step gets an advantage: how much
              better the outcome was than the critic expected. With λ = 0 that is just the surprise
              of the very next step, with λ = 1 it is the surprise of everything that followed, and
              in between the surprises of later steps count for less and less. The actor moves
              towards moves with positive advantage; the critic moves towards what it observed.
            </p>
            <p>
              The tests check two identities: λ = 0 gives exactly the one-step error and λ = 1 gives
              exactly the discounted return minus the value, and with perfect values every advantage
              is zero. The noise and bias shown above are measured at one fixed mid-training agent
              from 800 fresh episodes for each λ.
            </p>
          </div>
        </Panel>
        <Panel id="lab-controls" className="scroll-mt-24" title="Settings" eyebrow="04">
          <ParamPanel specs={acParams} values={params} onChange={setParam} columns={2} />
        </Panel>
      </div>
    </div>
  );
}
