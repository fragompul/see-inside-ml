import { useMemo, useState } from 'react';
import { BLOBS, BLOB_STD, blobCentre } from '@sim/deep';
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
import { DEFAULT_GUIDANCE, STEP_COUNTS, WEIGHTS, type GuidanceScenario } from './guidanceCore';
import { guidanceExperiments, guidanceParams, type GuidanceExperiment } from './guidanceSpec';

const toScenario = (p: ParamValues): GuidanceScenario => ({
  ...DEFAULT_GUIDANCE,
  mode: p.mode as GuidanceScenario['mode'],
  weight: Number(p.weight),
  asked: Number(p.asked),
  steps: Number(p.steps),
});

const fromScenario = (s: GuidanceScenario): ParamValues => ({
  mode: s.mode,
  weight: s.weight,
  asked: s.asked,
  steps: s.steps,
});

const SIZE = 300;
const RANGE = 3.4;
const at = (v: number) => ((v + RANGE) / (2 * RANGE)) * SIZE;

function Blobs({
  real,
  drawn,
  highlight,
  color,
  label,
}: {
  real: number[];
  drawn: number[];
  highlight?: number;
  color: string;
  label: string;
}) {
  const palette = usePalette();
  return (
    <svg
      viewBox={`0 0 ${SIZE} ${SIZE}`}
      className="w-full max-w-[20rem] rounded-xl border border-line bg-surface"
      role="img"
      aria-label={label}
    >
      {Array.from({ length: BLOBS }, (_, k) => {
        const [cx, cy] = blobCentre(k);
        return (
          <circle
            key={k}
            cx={at(cx)}
            cy={SIZE - at(cy)}
            r={at(2 * BLOB_STD) - at(0)}
            fill={k === highlight ? palette.css.brand : palette.css.muted}
            opacity={k === highlight ? 0.2 : 0.1}
          />
        );
      })}
      {Array.from({ length: real.length / 2 }, (_, i) => (
        <circle
          key={`r${i}`}
          cx={at(real[2 * i] as number)}
          cy={SIZE - at(real[2 * i + 1] as number)}
          r={1.3}
          fill={palette.css.muted}
          opacity={0.4}
        />
      ))}
      {Array.from({ length: drawn.length / 2 }, (_, i) => {
        const x = drawn[2 * i] as number;
        const y = drawn[2 * i + 1] as number;
        if (Math.abs(x) > RANGE || Math.abs(y) > RANGE) return null;
        return (
          <circle key={`d${i}`} cx={at(x)} cy={SIZE - at(y)} r={2.2} fill={color} opacity={0.8} />
        );
      })}
    </svg>
  );
}

export default function GuidancePage() {
  const palette = usePalette();
  const [params, setParams] = useState<ParamValues>(() => fromScenario(DEFAULT_GUIDANCE));
  const [activeId, setActiveId] = useState<string | null>(null);
  const scenario = useMemo(() => toScenario(params), [params]);
  const { result: run, error } = useCompute('guidance', scenario);
  const setParam = (key: string, value: ParamValue) => setParams((p) => ({ ...p, [key]: value }));

  const apply = (exp: GuidanceExperiment) => {
    setParams(fromScenario({ ...DEFAULT_GUIDANCE, ...exp.scenario }));
    setActiveId(exp.id);
  };
  const stillActive = (() => {
    const exp = guidanceExperiments.find((e) => e.id === activeId);
    return (
      !!exp &&
      JSON.stringify(fromScenario({ ...DEFAULT_GUIDANCE, ...exp.scenario })) ===
        JSON.stringify(params)
    );
  })();

  const wi = WEIGHTS.indexOf(scenario.weight);
  const si = STEP_COUNTS.indexOf(scenario.steps);
  const metrics: Metric[] = run
    ? scenario.mode === 'guidance'
      ? [
          {
            label: 'Points in the class asked for',
            value: `${fix((run.accuracy[wi] as number) * 100, 0)}%`,
            testId: 'accuracy',
            hint: `weight ${scenario.weight}, all six classes`,
          },
          { label: 'Mean distance from the class centre', value: fix(run.spread[wi] as number, 2) },
          {
            label: 'Same for real data',
            value: fix(run.realSpread, 2),
            hint: 'how varied the data is',
          },
        ]
      : [
          {
            label: 'Flow matching, distance from the data',
            value: fix(run.flowDistance[si] as number, 2),
            testId: 'flow-distance',
            hint: `${scenario.steps} steps, lower is better`,
          },
          {
            label: 'DDIM diffusion, same steps',
            value: fix(run.diffusionDistance[si] as number, 2),
          },
          {
            label: 'DDIM with 32 steps',
            value: fix(run.diffusionDistance[STEP_COUNTS.indexOf(32)] as number, 2),
          },
        ]
    : [];

  const stages: [string, string][] = [
    ['lab-view', scenario.mode === 'guidance' ? 'Drawing a class' : 'Few steps'],
    ['lab-sweep', scenario.mode === 'guidance' ? 'Across guidance weights' : 'Across step counts'],
    ['lab-how', 'How it works'],
    ['lab-controls', 'Settings'],
  ];

  return (
    <div className="mx-auto max-w-6xl px-4 pt-8 pb-20 sm:px-6 lg:px-8">
      <LabHeader
        eyebrow="Generative models"
        title="Guidance and flow matching"
        accent="steering and shortening diffusion"
        blurb="Two refinements of diffusion. Classifier-free guidance pushes a conditional model further from its unconditional guess to follow the prompt more closely, at a price in variety. Flow matching learns a velocity field along straight paths from noise to data, which takes a handful of steps instead of dozens."
        badges={
          <>
            <Badge>
              {scenario.mode === 'guidance' ? 'Classifier-free guidance' : 'Flow matching'}
            </Badge>
            <Badge>six overlapping classes</Badge>
          </>
        }
      />
      <StageNav stages={stages} />
      <ExperimentBar
        lessons={guidanceExperiments}
        activeId={stillActive ? activeId : null}
        onApply={apply}
      />

      <div className="flex flex-col gap-5">
        <Panel
          id="lab-view"
          className="scroll-mt-24"
          title={scenario.mode === 'guidance' ? 'Drawing a class' : 'Few steps'}
          eyebrow="01"
        >
          {run ? (
            <div className="space-y-6">
              <MetricsStrip metrics={metrics} className="lg:grid-cols-3" />
              {scenario.mode === 'guidance' ? (
                <div className="space-y-2">
                  <Blobs
                    real={run.real}
                    drawn={run.drawn}
                    highlight={scenario.asked}
                    color={palette.css.brand}
                    label="Points drawn for the class asked for, over the six real classes"
                  />
                  <p className="text-sm text-muted">
                    200 points drawn for class {scenario.asked + 1} (highlighted) at guidance weight{' '}
                    {scenario.weight}; the grey dots are real data. Points flung outside the frame
                    are not drawn.
                  </p>
                </div>
              ) : (
                <div className="space-y-2">
                  <div className="flex flex-wrap gap-6">
                    <figure className="space-y-1">
                      <Blobs
                        real={run.real}
                        drawn={run.flowPoints}
                        color={palette.css.pos}
                        label="Flow matching samples"
                      />
                      <figcaption className="text-xs text-muted">
                        Flow matching, {scenario.steps} Euler steps
                      </figcaption>
                    </figure>
                    <figure className="space-y-1">
                      <Blobs
                        real={run.real}
                        drawn={run.diffusionPoints}
                        color={palette.css.neg}
                        label="DDIM diffusion samples"
                      />
                      <figcaption className="text-xs text-muted">
                        DDIM diffusion, {scenario.steps} steps
                      </figcaption>
                    </figure>
                  </div>
                </div>
              )}
            </div>
          ) : (
            <p className="text-sm text-muted" role="status">
              {error
                ? `Something went wrong: ${error}`
                : 'Training the models (about half a minute the first time)…'}
            </p>
          )}
        </Panel>
        <Panel
          id="lab-sweep"
          className="scroll-mt-24"
          title={scenario.mode === 'guidance' ? 'Across guidance weights' : 'Across step counts'}
          eyebrow="02"
        >
          {run ? (
            scenario.mode === 'guidance' ? (
              <div className="grid gap-6 lg:grid-cols-2">
                <MiniChart
                  title="Share of points in the class asked for"
                  x={WEIGHTS}
                  xLabel="guidance weight"
                  series={[{ name: 'accuracy', color: palette.css.brand, values: run.accuracy }]}
                  height={200}
                />
                <MiniChart
                  title="Mean distance from the class centre (log scale)"
                  x={WEIGHTS}
                  xLabel="guidance weight"
                  log
                  series={[
                    { name: 'samples', color: palette.css.neg, values: run.spread },
                    {
                      name: 'real data',
                      color: palette.css.muted,
                      values: WEIGHTS.map(() => run.realSpread),
                      dashed: true,
                    },
                  ]}
                  height={200}
                />
              </div>
            ) : (
              <MiniChart
                title="Distance from the data by number of steps (log scale, lower is better)"
                x={STEP_COUNTS}
                xLabel="sampling steps"
                log
                series={[
                  { name: 'flow matching', color: palette.css.pos, values: run.flowDistance },
                  { name: 'DDIM diffusion', color: palette.css.neg, values: run.diffusionDistance },
                ]}
                height={230}
              />
            )
          ) : (
            <p className="text-sm text-muted">Training…</p>
          )}
        </Panel>
        <Panel id="lab-how" className="scroll-mt-24" title="How it works" eyebrow="03">
          <div className="space-y-3 text-sm text-muted">
            <p>
              <strong className="text-fg">Guidance.</strong> One network is trained to predict the
              noise given the point, the time and the class, and in 15% of the examples the class is
              hidden. It therefore knows two predictions: with the class (c) and without (u). When
              drawing, the prediction used is c plus w times (c minus u): the difference points
              towards &ldquo;more like this class&rdquo;, and w says how far to follow it. Small w
              sharpens; very large w overshoots the data altogether.
            </p>
            <p>
              <strong className="text-fg">Flow matching.</strong> A noise sample and a data sample
              are joined by a straight line, along which a point moves at constant velocity. A
              network learns that velocity from the position and time, and drawing is following the
              field from noise to data in a few Euler steps. The diffusion model it is compared with
              has the same size, data and training steps. The tests check the network&rsquo;s
              gradients, that the exact field of a single point lands exactly on it, and the claims
              above.
            </p>
          </div>
        </Panel>
        <Panel id="lab-controls" className="scroll-mt-24" title="Settings" eyebrow="04">
          <ParamPanel specs={guidanceParams} values={params} onChange={setParam} columns={2} />
        </Panel>
      </div>
    </div>
  );
}
