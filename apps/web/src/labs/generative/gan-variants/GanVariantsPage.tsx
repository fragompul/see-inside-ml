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
import {
  DEFAULT_GAN_LAB,
  KIND_LABEL,
  MODE_CENTERS,
  type GanLabRun,
  type GanScenario,
} from './ganCore';
import { ganExperiments, ganParams, type GanExperiment } from './ganSpec';

const toScenario = (p: ParamValues): GanScenario => ({
  kind: p.kind as GanScenario['kind'],
  steps: Number(p.steps),
  seed: Number(p.seed),
});

const fromScenario = (s: GanScenario): ParamValues => ({ ...s });

const SIZE = 340;
const RANGE = 3.2;
const px = (v: number) => ((v + RANGE) / (2 * RANGE)) * SIZE;

function Scatter({ run }: { run: GanLabRun }) {
  const palette = usePalette();
  const conditional = run.scenario.kind === 'cgan';
  const hue = (k: number) => `hsl(${(k * 45) % 360} 70% 50%)`;
  return (
    <svg
      viewBox={`0 0 ${SIZE} ${SIZE}`}
      className="w-full max-w-[26rem] rounded-xl border border-line bg-surface"
      role="img"
      aria-label="Generated points over the eight real clusters"
    >
      {MODE_CENTERS.map((c, k) => (
        <circle
          key={k}
          cx={px(c[0])}
          cy={SIZE - px(c[1])}
          r={px(0.5) - px(0)}
          fill={palette.css.muted}
          opacity={0.12}
        />
      ))}
      {Array.from({ length: run.real.length / 2 }, (_, i) => (
        <circle
          key={`r${i}`}
          cx={px(run.real[2 * i] as number)}
          cy={SIZE - px(run.real[2 * i + 1] as number)}
          r={1.6}
          fill={palette.css.muted}
          opacity={0.45}
        />
      ))}
      {Array.from({ length: run.samples.length / 2 }, (_, i) => (
        <circle
          key={`g${i}`}
          cx={px(run.samples[2 * i] as number)}
          cy={SIZE - px(run.samples[2 * i + 1] as number)}
          r={2.4}
          fill={conditional ? hue(run.labels[i] as number) : palette.css.brand}
          opacity={0.85}
        />
      ))}
    </svg>
  );
}

export default function GanVariantsPage() {
  const palette = usePalette();
  const [params, setParams] = useState<ParamValues>(() => fromScenario(DEFAULT_GAN_LAB));
  const [activeId, setActiveId] = useState<string | null>(null);
  const scenario = useMemo(() => toScenario(params), [params]);
  const { result: run, error } = useCompute('gan-variants', scenario);
  const setParam = (key: string, value: ParamValue) => setParams((p) => ({ ...p, [key]: value }));

  const apply = (exp: GanExperiment) => {
    setParams(fromScenario({ ...DEFAULT_GAN_LAB, ...exp.scenario }));
    setActiveId(exp.id);
  };
  const stillActive = (() => {
    const exp = ganExperiments.find((e) => e.id === activeId);
    return (
      !!exp &&
      JSON.stringify(fromScenario({ ...DEFAULT_GAN_LAB, ...exp.scenario })) ===
        JSON.stringify(params)
    );
  })();

  const metrics: Metric[] = run
    ? [
        {
          label: 'Clusters covered',
          value: `${run.modes} of 8`,
          testId: 'modes',
          hint: 'with at least 2% of the points',
        },
        {
          label: 'Distance from the data',
          value: fix(run.distance, 2),
          hint: 'sliced Wasserstein, lower is better',
        },
        {
          label: run.classAccuracy === null ? 'Points on real data' : 'In the cluster asked for',
          value:
            run.classAccuracy === null
              ? `${fix(run.nearData * 100, 0)}%`
              : `${fix(run.classAccuracy * 100, 0)}%`,
        },
        { label: 'Critic updates', value: run.criticUpdates.toLocaleString('en-US') },
      ]
    : [];

  const stages: [string, string][] = [
    ['lab-points', 'The points'],
    ['lab-history', 'During training'],
    ['lab-how', 'How it works'],
    ['lab-controls', 'Settings'],
  ];

  return (
    <div className="mx-auto max-w-6xl px-4 pt-8 pb-20 sm:px-6 lg:px-8">
      <LabHeader
        eyebrow="Generative models"
        title="GAN recipes compared"
        accent="the same game, three ways"
        blurb="A generator and a critic play against each other. The rules of the game change what the generator learns and how reliably. Compare the original game with a Wasserstein critic and a gradient penalty, and with a conditional version that is told which cluster to draw, on eight Gaussian clusters."
        badges={
          <>
            <Badge>{KIND_LABEL[scenario.kind]}</Badge>
            <Badge>start #{scenario.seed}</Badge>
          </>
        }
      />
      <StageNav stages={stages} />
      <ExperimentBar
        lessons={ganExperiments}
        activeId={stillActive ? activeId : null}
        onApply={apply}
      />

      <div className="flex flex-col gap-5">
        <Panel id="lab-points" className="scroll-mt-24" title="The points" eyebrow="01">
          {run ? (
            <div className="space-y-6">
              <MetricsStrip metrics={metrics} className="lg:grid-cols-4" />
              <Scatter run={run} />
              <p className="text-sm text-muted">
                Grey points are real data, in eight clusters. Coloured points are what the generator
                makes after training
                {run.classAccuracy !== null ? ', coloured by the cluster it was asked for' : ''}.
              </p>
            </div>
          ) : (
            <p className="text-sm text-muted" role="status">
              {error ? `Something went wrong: ${error}` : 'Training the generator and the critic…'}
            </p>
          )}
        </Panel>
        <Panel id="lab-history" className="scroll-mt-24" title="During training" eyebrow="02">
          {run ? (
            <div className="grid gap-6 lg:grid-cols-2">
              <MiniChart
                title="Clusters covered"
                x={run.history.map((h) => h.step)}
                xLabel="generator updates"
                series={[
                  {
                    name: 'clusters',
                    color: palette.css.brand,
                    values: run.history.map((h) => h.modes),
                  },
                ]}
                height={200}
              />
              <MiniChart
                title="Distance from the data (lower is better)"
                x={run.history.map((h) => h.step)}
                xLabel="generator updates"
                series={[
                  {
                    name: 'distance',
                    color: palette.css.neg,
                    values: run.history.map((h) => h.distance),
                  },
                ]}
                height={200}
              />
            </div>
          ) : (
            <p className="text-sm text-muted">Training…</p>
          )}
        </Panel>
        <Panel id="lab-how" className="scroll-mt-24" title="How it works" eyebrow="03">
          <div className="space-y-3 text-sm text-muted">
            <p>
              In the <strong className="text-fg">standard GAN</strong> the discriminator outputs the
              probability that a point is real and is trained as a classifier, while the generator
              is trained to fool it. The <strong className="text-fg">WGAN-GP</strong> critic outputs
              an unbounded score; it is trained to give real points higher scores than generated
              ones, and a penalty on the size of its gradient at points between real and generated
              ones keeps the scores from growing without limit. The penalty needs the gradient of a
              gradient; for this critic, whose units are piecewise linear, that has a closed form,
              checked in the tests against finite differences.
            </p>
            <p>
              The <strong className="text-fg">conditional GAN</strong> adds the cluster number to
              the inputs of both players. Distances are sliced Wasserstein distances between 400
              generated and 400 real points. The same training run shows large differences from one
              random start to the next, so the experiments above name the start they use.
            </p>
          </div>
        </Panel>
        <Panel id="lab-controls" className="scroll-mt-24" title="Settings" eyebrow="04">
          <ParamPanel specs={ganParams} values={params} onChange={setParam} columns={2} />
        </Panel>
      </div>
    </div>
  );
}
