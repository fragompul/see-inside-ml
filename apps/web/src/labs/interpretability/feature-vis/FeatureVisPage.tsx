import { useMemo, useState } from 'react';
import { IMAGE_SIZE } from '@sim/deep';
import { Badge, Panel } from '@/design-system';
import { useCompute } from '@/shared/compute/useCompute';
import { ExperimentBar } from '@/shared/components/ExperimentBar';
import { PixelCanvas } from '@/shared/components/PixelCanvas';
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
import { DEFAULT_VIS, REGULARISER_LABEL, type VisResult, type VisScenario } from './visCore';
import { visExperiments, visParams, type VisExperiment } from './visSpec';

const toScenario = (p: ParamValues): VisScenario => ({
  ...DEFAULT_VIS,
  kind: p.kind as VisScenario['kind'],
  regulariser: p.regulariser as VisScenario['regulariser'],
  steps: Number(p.steps),
});

const fromScenario = (s: VisScenario): ParamValues => ({
  kind: s.kind,
  regulariser: s.regulariser,
  steps: s.steps,
});

function Results({ run }: { run: VisResult }) {
  const classes = run.scenario.kind === 'class';
  const metrics: Metric[] = [
    {
      label: 'Average score of the pictures',
      value: fix(run.meanScore, 1),
      testId: 'score',
      hint: `real digits: ${fix(run.realScore, 1)}`,
    },
    {
      label: 'How noisy (total variation)',
      value: fix(run.meanNoise, 0),
      hint: `real digits: ${fix(run.realNoise, 0)}`,
    },
    ...(classes
      ? [
          {
            label: 'Like the average real digit',
            value: fix(run.meanLikeness, 2),
            hint: 'correlation',
          },
          {
            label: 'Read as the class asked for',
            value: `${fix(run.hitRate * 100, 0)}%`,
          },
        ]
      : []),
  ];
  return (
    <Panel
      id="lab-pictures"
      className="scroll-mt-24"
      title={
        classes ? 'What each digit looks like to the network' : 'What each feature map prefers'
      }
      eyebrow="01"
    >
      <div className="space-y-6">
        <MetricsStrip metrics={metrics} className="lg:grid-cols-4" />
        <div className="flex flex-wrap gap-3">
          {run.images.map((img, k) => (
            <figure key={k} className="space-y-1 text-center">
              <PixelCanvas
                values={img}
                side={IMAGE_SIZE}
                size={84}
                ariaLabel={classes ? `Picture for digit ${k}` : `Picture for feature map ${k + 1}`}
                className="rounded-lg"
              />
              <figcaption className="font-mono text-[11px] text-muted">
                {classes ? `digit ${k}` : `map ${k + 1}`} · {fix(run.scores[k] as number, 1)}
                {classes && run.predicted[k] !== k ? ` (read as ${run.predicted[k]})` : ''}
              </figcaption>
            </figure>
          ))}
        </div>
        <p className="text-sm text-muted">
          Each picture was grown from grey with a little noise by repeatedly changing the pixels in
          the direction that raises the unit&rsquo;s score. The number is that score. Real digits
          score about {fix(run.realScore, 1)} for their own class.
        </p>
      </div>
    </Panel>
  );
}

export default function FeatureVisPage() {
  const [params, setParams] = useState<ParamValues>(() => fromScenario(DEFAULT_VIS));
  const [activeId, setActiveId] = useState<string | null>(null);
  const scenario = useMemo(() => toScenario(params), [params]);
  const { result: run, error } = useCompute('feature-vis', scenario);
  const setParam = (key: string, value: ParamValue) => setParams((p) => ({ ...p, [key]: value }));

  const apply = (exp: VisExperiment) => {
    setParams(fromScenario({ ...DEFAULT_VIS, ...exp.scenario }));
    setActiveId(exp.id);
  };
  const stillActive = (() => {
    const exp = visExperiments.find((e) => e.id === activeId);
    return (
      !!exp &&
      JSON.stringify(fromScenario({ ...DEFAULT_VIS, ...exp.scenario })) === JSON.stringify(params)
    );
  })();

  const stages: [string, string][] = [
    [
      'lab-pictures',
      scenario.kind === 'class'
        ? 'What each digit looks like to the network'
        : 'What each feature map prefers',
    ],
    ['lab-how', 'How it works'],
    ['lab-controls', 'Settings'],
  ];

  return (
    <div className="mx-auto max-w-6xl px-4 pt-8 pb-20 sm:px-6 lg:px-8">
      <LabHeader
        eyebrow="Interpretability"
        title="Feature visualisation"
        accent="what a unit would draw"
        blurb="Instead of asking which of your pictures excites a unit, ask the network to draw the picture that excites it most, by gradient ascent on the pixels. What comes out, with and without help to keep it natural, says what the network has learned to look for and how little that resembles what we would show it."
        badges={
          <>
            <Badge>{scenario.kind === 'class' ? 'digit classes' : 'feature maps'}</Badge>
            <Badge>{REGULARISER_LABEL[scenario.regulariser]}</Badge>
          </>
        }
      />
      <StageNav stages={stages} />
      <ExperimentBar
        lessons={visExperiments}
        activeId={stillActive ? activeId : null}
        onApply={apply}
      />

      <div className="flex flex-col gap-5">
        {run ? (
          <Results run={run} />
        ) : (
          <Panel
            id="lab-pictures"
            className="scroll-mt-24"
            title={
              scenario.kind === 'class'
                ? 'What each digit looks like to the network'
                : 'What each feature map prefers'
            }
            eyebrow="01"
          >
            <p className="text-sm text-muted" role="status">
              {error
                ? `Something went wrong: ${error}`
                : 'Training the network and growing the pictures…'}
            </p>
          </Panel>
        )}
        <Panel id="lab-how" className="scroll-mt-24" title="How it works" eyebrow="02">
          <div className="space-y-3 text-sm text-muted">
            <p>
              The score of a class is the network&rsquo;s output for it before the softmax; the
              score of a feature map is its average activation. Starting from a grey picture with
              faint noise, each step moves every pixel a little in the direction of the gradient of
              the score and keeps it between 0 and 1. Because the network was trained only on real
              digits, nothing stops this from finding patterns that no real digit has.
            </p>
            <p>
              Two ways of keeping the picture natural are compared: removing a small fraction of it
              at every step, and blurring it slightly. The tests check the gradient of a feature map
              against finite differences, that the score really rises and that blurring lowers the
              noise.
            </p>
          </div>
        </Panel>
        <Panel id="lab-controls" className="scroll-mt-24" title="Settings" eyebrow="03">
          <ParamPanel specs={visParams} values={params} onChange={setParam} columns={2} />
        </Panel>
      </div>
    </div>
  );
}
