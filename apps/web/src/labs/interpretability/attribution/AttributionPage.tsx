import { useMemo, useState } from 'react';
import { ATTRIBUTION_METHODS, IMAGE_SIZE, type AttributionMethod } from '@sim/deep';
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
  DEFAULT_ATTRIBUTION,
  METHOD_LABEL,
  STEPS,
  type AttributionResult,
  type AttributionScenario,
} from './attributionCore';
import {
  attributionExperiments,
  attributionParams,
  type AttributionExperiment,
} from './attributionSpec';

const toScenario = (p: ParamValues): AttributionScenario => ({
  ...DEFAULT_ATTRIBUTION,
  example: Number(p.example),
  method: p.method as AttributionMethod,
  steps: Number(p.steps),
});

const fromScenario = (s: AttributionScenario): ParamValues => ({
  example: s.example,
  method: s.method,
  steps: s.steps,
});

function Results({ run, onPick }: { run: AttributionResult; onPick: (example: number) => void }) {
  const palette = usePalette();
  const chosen = run.images[run.scenario.example];
  const completeStep = STEPS.indexOf(run.scenario.steps);
  const metrics: Metric[] = [
    {
      label: 'Network accuracy',
      value: `${fix(run.accuracy * 100, 0)}%`,
      testId: 'accuracy',
      hint: 'on new digits',
    },
    {
      label: 'Sum of the attributions',
      value: fix(run.sum, 2),
      hint: `score gap ${fix(run.gap, 2)} (${run.scenario.steps} steps)`,
    },
    {
      label: 'Sum is off by',
      value: `${fix((run.completeness[completeStep] as number) * 100, 1)}%`,
      hint: 'average over 30 digits',
    },
  ];
  return (
    <>
      <Panel id="lab-maps" className="scroll-mt-24" title="Four ways to attribute" eyebrow="01">
        <div className="space-y-6">
          <MetricsStrip metrics={metrics} className="lg:grid-cols-3" />
          <div className="flex flex-wrap gap-2" role="group" aria-label="Choose a digit">
            {run.images.map((img, i) => (
              <button
                key={i}
                type="button"
                aria-pressed={i === run.scenario.example}
                onClick={() => onPick(i)}
                className={cn(
                  'rounded-lg border p-0.5',
                  i === run.scenario.example ? 'border-brand' : 'border-line',
                )}
              >
                <PixelCanvas
                  values={img.pixels}
                  side={IMAGE_SIZE}
                  size={48}
                  ariaLabel={`Digit ${i + 1}`}
                />
              </button>
            ))}
          </div>
          {chosen && (
            <div className="space-y-2">
              <div className="flex flex-wrap items-start gap-4">
                <figure className="space-y-1">
                  <PixelCanvas
                    values={chosen.pixels}
                    side={IMAGE_SIZE}
                    size={120}
                    ariaLabel="The digit"
                    className="rounded-lg"
                  />
                  <figcaption className="text-center text-xs text-muted">
                    the digit (network says {chosen.predicted})
                  </figcaption>
                </figure>
                {ATTRIBUTION_METHODS.map((m) => {
                  const map = run.maps[m];
                  const scale = Math.max(...map.map(Math.abs), 1e-9);
                  return (
                    <figure key={m} className="space-y-1">
                      <PixelCanvas
                        values={map}
                        side={IMAGE_SIZE}
                        size={120}
                        mode="diverging"
                        scale={scale}
                        ariaLabel={METHOD_LABEL[m]}
                        className={cn(
                          'rounded-lg',
                          m === run.scenario.method && 'ring-2 ring-brand',
                        )}
                      />
                      <figcaption className="max-w-[7.5rem] text-center text-xs text-muted">
                        {METHOD_LABEL[m]}
                      </figcaption>
                    </figure>
                  );
                })}
              </div>
              <p className="text-sm text-muted">
                Warm pixels push the score of the predicted class up, cool ones push it down.
                Saliency and SmoothGrad have no sign, so they are all warm.
              </p>
            </div>
          )}
        </div>
      </Panel>
      <Panel
        id="lab-faith"
        className="scroll-mt-24"
        title="Do the maps tell the truth?"
        eyebrow="02"
      >
        <div className="space-y-4">
          <div className="overflow-x-auto">
            <table className="w-full text-left text-sm">
              <thead className="text-muted">
                <tr>
                  <th className="py-2 pr-4 font-medium">Method</th>
                  <th className="py-2 pr-4 font-medium">Score while deleting pixels</th>
                  <th className="py-2 font-medium">Map after scrambling the output layer</th>
                </tr>
              </thead>
              <tbody>
                {ATTRIBUTION_METHODS.map((m) => (
                  <tr key={m} className="border-t border-line">
                    <td className="py-2 pr-4">{METHOD_LABEL[m]}</td>
                    <td className="py-2 pr-4 font-mono">{fix(run.deletion[m], 2)}</td>
                    <td className="py-2 font-mono">correlation {fix(run.sanity[m], 2)}</td>
                  </tr>
                ))}
                <tr className="border-t border-line">
                  <td className="py-2 pr-4">Random order</td>
                  <td className="py-2 pr-4 font-mono">{fix(run.randomDeletion, 2)}</td>
                  <td className="py-2 font-mono">–</td>
                </tr>
              </tbody>
            </table>
          </div>
          <p className="text-sm text-muted">
            Deletion: the pixels are blanked most important first and the class score is averaged
            along the way; lower means the map found what the network really used. Scrambling: the
            maps are recomputed with a random output layer; a map that still looks the same was not
            showing what the network learned.
          </p>
        </div>
      </Panel>
      <Panel id="lab-complete" className="scroll-mt-24" title="Adding up" eyebrow="03">
        <MiniChart
          title="How far the integrated gradients are from the score gap (share of the gap, log scale)"
          x={STEPS}
          xLabel="integration steps"
          log
          series={[{ name: 'error', color: palette.css.brand, values: run.completeness }]}
          height={200}
        />
      </Panel>
    </>
  );
}

export default function AttributionPage() {
  const [params, setParams] = useState<ParamValues>(() => fromScenario(DEFAULT_ATTRIBUTION));
  const [activeId, setActiveId] = useState<string | null>(null);
  const scenario = useMemo(() => toScenario(params), [params]);
  const { result: run, error } = useCompute('attribution', scenario);
  const setParam = (key: string, value: ParamValue) => setParams((p) => ({ ...p, [key]: value }));

  const apply = (exp: AttributionExperiment) => {
    setParams(fromScenario({ ...DEFAULT_ATTRIBUTION, ...exp.scenario }));
    setActiveId(exp.id);
  };
  const stillActive = (() => {
    const exp = attributionExperiments.find((e) => e.id === activeId);
    return (
      !!exp &&
      JSON.stringify(fromScenario({ ...DEFAULT_ATTRIBUTION, ...exp.scenario })) ===
        JSON.stringify(params)
    );
  })();

  const stages: [string, string][] = [
    ['lab-maps', 'Four ways to attribute'],
    ['lab-faith', 'Do the maps tell the truth?'],
    ['lab-complete', 'Adding up'],
    ['lab-how', 'How it works'],
    ['lab-controls', 'Settings'],
  ];

  return (
    <div className="mx-auto max-w-6xl px-4 pt-8 pb-20 sm:px-6 lg:px-8">
      <LabHeader
        eyebrow="Interpretability"
        title="Pixel attribution"
        accent="which pixels was the answer owed to?"
        blurb="Saliency, gradient × input, SmoothGrad and integrated gradients all colour the pixels of an image by how much they mattered to a network's answer, and they disagree. Test them: do the maps add up, do they find the pixels the network really used, and do they change when the network does?"
        badges={
          <>
            <Badge>digit CNN</Badge>
            <Badge>{METHOD_LABEL[scenario.method]}</Badge>
          </>
        }
      />
      <StageNav stages={stages} />
      <ExperimentBar
        lessons={attributionExperiments}
        activeId={stillActive ? activeId : null}
        onApply={apply}
      />

      <div className="flex flex-col gap-5">
        {run ? (
          <Results run={run} onPick={(example) => setParam('example', example)} />
        ) : (
          <Panel id="lab-maps" className="scroll-mt-24" title="Four ways to attribute" eyebrow="01">
            <p className="text-sm text-muted" role="status">
              {error
                ? `Something went wrong: ${error}`
                : 'Training the network and evaluating the methods…'}
            </p>
          </Panel>
        )}
        <Panel id="lab-how" className="scroll-mt-24" title="How it works" eyebrow="04">
          <div className="space-y-3 text-sm text-muted">
            <p>
              The score of a class is the network&rsquo;s output for it before the softmax.{' '}
              <strong className="text-fg">Saliency</strong> is the size of its gradient with respect
              to each pixel: how steeply the score would change if that pixel did.{' '}
              <strong className="text-fg">Gradient × input</strong> multiplies by the pixel itself.{' '}
              <strong className="text-fg">SmoothGrad</strong> averages saliency over noisy copies.{' '}
              <strong className="text-fg">Integrated gradients</strong> averages the gradient along
              the straight line from a blank image to the image, then multiplies by the image; by
              the fundamental theorem of calculus the attributions then sum to the change in score,
              whatever the network.
            </p>
            <p>
              The tests check the gradient against finite differences, that the attributions sum to
              the score gap more exactly with more steps, and that deleting the most important
              pixels first lowers the score more than deleting them last.
            </p>
          </div>
        </Panel>
        <Panel id="lab-controls" className="scroll-mt-24" title="Settings" eyebrow="05">
          <ParamPanel specs={attributionParams} values={params} onChange={setParam} columns={2} />
        </Panel>
      </div>
    </div>
  );
}
