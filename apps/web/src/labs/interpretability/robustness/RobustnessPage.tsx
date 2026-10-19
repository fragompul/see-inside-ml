import { useMemo, useState } from 'react';
import { IMAGE_SIZE } from '@sim/deep';
import { Badge, Panel, cn } from '@/design-system';
import { usePalette } from '@/shared/canvas/colors';
import { useCompute } from '@/shared/compute/useCompute';
import { ExperimentBar } from '@/shared/components/ExperimentBar';
import { MiniChart } from '@/shared/components/MiniChart';
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
import {
  BUDGETS,
  DEFAULT_ROBUSTNESS,
  type RobustScenario,
  type RobustnessResult,
} from './robustCore';
import { robustExperiments, robustParams, type RobustExperiment } from './robustSpec';

const toScenario = (p: ParamValues): RobustScenario => ({
  trainEpsilon: Number(p.trainEpsilon),
  trainMethod: p.trainMethod as RobustScenario['trainMethod'],
  testMethod: p.testMethod as RobustScenario['testMethod'],
  shownEpsilon: Number(p.shownEpsilon),
});

const fromScenario = (s: RobustScenario): ParamValues => ({ ...s });

const pct = (v: number) => `${fix(v * 100, 0)}%`;

function Results({ run }: { run: RobustnessResult }) {
  const palette = usePalette();
  const s = run.scenario;
  const shown = BUDGETS.indexOf(s.shownEpsilon);
  const metrics: Metric[] = [
    { label: 'Ordinary network, untouched', value: pct(run.ordinary.clean), testId: 'clean' },
    {
      label: `Ordinary under ${s.testMethod.toUpperCase()}, budget ${s.shownEpsilon}`,
      value: pct(run.ordinary[s.testMethod][shown] as number),
    },
    {
      label: `Trained network under ${s.testMethod.toUpperCase()}, budget ${s.shownEpsilon}`,
      value: pct(run.chosen[s.testMethod][shown] as number),
      hint:
        s.trainEpsilon === 0
          ? 'same as the ordinary one'
          : `trained with ${s.trainMethod.toUpperCase()} ${s.trainEpsilon}`,
    },
  ];
  const label = s.trainEpsilon === 0 ? 'ordinary' : `trained at ${s.trainEpsilon}`;
  return (
    <>
      <Panel id="lab-curves" className="scroll-mt-24" title="Accuracy under attack" eyebrow="01">
        <div className="space-y-6">
          <MetricsStrip metrics={metrics} className="lg:grid-cols-3" />
          <div className="grid gap-6 lg:grid-cols-2">
            {(['fgsm', 'pgd'] as const).map((method) => (
              <MiniChart
                key={method}
                title={`${method === 'fgsm' ? 'One-step attack (FGSM)' : 'Iterated attack (PGD)'}`}
                x={BUDGETS}
                xLabel="attack budget (largest change per pixel)"
                series={[
                  { name: 'ordinary', color: palette.css.neg, values: run.ordinary[method] },
                  ...(s.trainEpsilon > 0
                    ? [{ name: label, color: palette.css.pos, values: run.chosen[method] }]
                    : []),
                ]}
                height={210}
              />
            ))}
          </div>
        </div>
      </Panel>
      <Panel id="lab-examples" className="scroll-mt-24" title="What the attack does" eyebrow="02">
        <div className="space-y-3">
          <div className="flex flex-wrap gap-4">
            {run.examples.map((e, i) => (
              <figure key={i} className="space-y-1">
                <div className="flex gap-1">
                  <PixelCanvas
                    values={e.original}
                    side={IMAGE_SIZE}
                    size={76}
                    ariaLabel={`Original digit ${i + 1}`}
                    className="rounded"
                  />
                  <PixelCanvas
                    values={e.attacked}
                    side={IMAGE_SIZE}
                    size={76}
                    ariaLabel={`Attacked digit ${i + 1}`}
                    className={cn('rounded', e.after !== e.label && 'ring-2 ring-neg')}
                  />
                </div>
                <figcaption className="font-mono text-[11px] text-muted">
                  {e.before} → {e.after}
                  {e.after !== e.label ? ' fooled' : ''}
                </figcaption>
              </figure>
            ))}
          </div>
          <p className="text-sm text-muted">
            Each pair is a digit and the same digit after a {s.testMethod.toUpperCase()} attack of
            budget {s.shownEpsilon} against the {label} network. The numbers are its answer before
            and after; a red ring marks a digit it now gets wrong.
          </p>
        </div>
      </Panel>
    </>
  );
}

export default function RobustnessPage() {
  const [params, setParams] = useState<ParamValues>(() => fromScenario(DEFAULT_ROBUSTNESS));
  const [activeId, setActiveId] = useState<string | null>(null);
  const scenario = useMemo(() => toScenario(params), [params]);
  const { result: run, error } = useCompute('robustness', scenario);
  const setParam = (key: string, value: ParamValue) => setParams((p) => ({ ...p, [key]: value }));

  const apply = (exp: RobustExperiment) => {
    setParams(fromScenario({ ...DEFAULT_ROBUSTNESS, ...exp.scenario }));
    setActiveId(exp.id);
  };
  const stillActive = (() => {
    const exp = robustExperiments.find((e) => e.id === activeId);
    return (
      !!exp &&
      JSON.stringify(fromScenario({ ...DEFAULT_ROBUSTNESS, ...exp.scenario })) ===
        JSON.stringify(params)
    );
  })();

  const stages: [string, string][] = [
    ['lab-curves', 'Accuracy under attack'],
    ['lab-examples', 'What the attack does'],
    ['lab-how', 'How it works'],
    ['lab-controls', 'Settings'],
  ];

  return (
    <div className="mx-auto max-w-6xl px-4 pt-8 pb-20 sm:px-6 lg:px-8">
      <LabHeader
        eyebrow="Interpretability"
        title="Adversarial robustness"
        accent="small changes, big mistakes"
        blurb="A digit network that is right 98% of the time can be made wrong by changes to each pixel too small to matter to a person. Attack it, then train it against attacks, and measure what protection costs and how far it reaches."
        badges={
          <>
            <Badge>digit CNN</Badge>
            <Badge>
              {scenario.trainEpsilon === 0
                ? 'ordinary training'
                : `trained against ${scenario.trainMethod.toUpperCase()} ${scenario.trainEpsilon}`}
            </Badge>
          </>
        }
      />
      <StageNav stages={stages} />
      <ExperimentBar
        lessons={robustExperiments}
        activeId={stillActive ? activeId : null}
        onApply={apply}
      />

      <div className="flex flex-col gap-5">
        {run ? (
          <Results run={run} />
        ) : (
          <Panel
            id="lab-curves"
            className="scroll-mt-24"
            title="Accuracy under attack"
            eyebrow="01"
          >
            <p className="text-sm text-muted" role="status">
              {error ? `Something went wrong: ${error}` : 'Training the network and attacking it…'}
            </p>
          </Panel>
        )}
        <Panel id="lab-how" className="scroll-mt-24" title="How it works" eyebrow="03">
          <div className="space-y-3 text-sm text-muted">
            <p>
              An attack changes each pixel by at most the budget, in the direction that raises the
              network&rsquo;s loss the most: FGSM takes one step along the sign of the gradient, PGD
              takes several smaller ones and stays within the budget. Adversarial training does the
              same to the training digits while the network learns, and trains on the attacked copy
              as well as on the original.
            </p>
            <p>
              Accuracies are on 300 new digits. The attacked pictures are checked in the tests never
              to differ from the original by more than the budget.
            </p>
          </div>
        </Panel>
        <Panel id="lab-controls" className="scroll-mt-24" title="Settings" eyebrow="04">
          <ParamPanel specs={robustParams} values={params} onChange={setParam} columns={2} />
        </Panel>
      </div>
    </div>
  );
}
