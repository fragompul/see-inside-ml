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
import { DEFAULT_PROBING, type ProbingResult, type ProbingScenario } from './probingCore';
import { probingExperiments, probingParams, type ProbingExperiment } from './probingSpec';

const toScenario = (p: ParamValues): ProbingScenario => ({
  model: p.model as ProbingScenario['model'],
});
const fromScenario = (s: ProbingScenario): ParamValues => ({ model: s.model });
const pct = (v: number) => `${fix(v * 100, 0)}%`;

function Results({ run }: { run: ProbingResult }) {
  const palette = usePalette();
  const colors = [palette.css.brand, palette.css.pos, palette.css.neg];
  const first = run.tasks[0];
  const metrics: Metric[] = [
    {
      label: `${first?.name ?? ''}, first layer`,
      value: pct((first?.test[0] as number) ?? 0),
      testId: 'first-score',
      hint: `chance ${pct(first?.chance ?? 0)}`,
    },
    {
      label: `${first?.name ?? ''}, last layer`,
      value: pct((first?.test[run.layers.length - 1] as number) ?? 0),
    },
    {
      label: 'Control: memorised random labels',
      value: pct(run.controlTrain[0] as number),
      hint: `first layer, ${run.widths[0]} features; held-out ${pct(run.controlTest[0] as number)}`,
    },
  ];
  return (
    <>
      <Panel id="lab-probes" className="scroll-mt-24" title="What each layer knows" eyebrow="01">
        <div className="space-y-6">
          <MetricsStrip metrics={metrics} className="lg:grid-cols-3" />
          <MiniChart
            title="Accuracy of a linear probe on held-out examples, by layer"
            x={run.layers.map((_, i) => i)}
            xLabel={`layer (${run.layers.join(', ')})`}
            series={run.tasks.map((t, i) => ({
              name: t.name,
              color: colors[i % colors.length] as string,
              values: t.test,
            }))}
            height={250}
          />
          <div className="overflow-x-auto">
            <table className="w-full text-left text-sm">
              <thead className="text-muted">
                <tr>
                  <th className="py-2 pr-4 font-medium">Property</th>
                  {run.layers.map((l) => (
                    <th key={l} className="py-2 pr-4 font-medium">
                      {l}
                    </th>
                  ))}
                  <th className="py-2 font-medium">Chance</th>
                </tr>
              </thead>
              <tbody>
                {run.tasks.map((t) => (
                  <tr key={t.name} className="border-t border-line">
                    <td className="py-2 pr-4">{t.name}</td>
                    {t.test.map((v, i) => (
                      <td key={i} className="py-2 pr-4 font-mono">
                        {pct(v)}
                      </td>
                    ))}
                    <td className="py-2 font-mono text-muted">{pct(t.chance)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </Panel>
      <Panel id="lab-control" className="scroll-mt-24" title="The control task" eyebrow="02">
        <div className="space-y-3">
          <MiniChart
            title="A probe asked to learn random labels"
            x={run.layers.map((_, i) => i)}
            xLabel="layer"
            series={[
              {
                name: 'on its training examples',
                color: palette.css.neg,
                values: run.controlTrain,
              },
              {
                name: 'on new examples',
                color: palette.css.muted,
                values: run.controlTest,
                dashed: true,
              },
            ]}
            height={200}
          />
          <p className="text-sm text-muted">
            Random labels cannot be learned, only memorised. When a layer has many features the
            probe memorises them anyway (high on its own examples, chance on new ones), so a high
            score on a real property needs to be judged against this.
          </p>
        </div>
      </Panel>
    </>
  );
}

export default function ProbingPage() {
  const [params, setParams] = useState<ParamValues>(() => fromScenario(DEFAULT_PROBING));
  const [activeId, setActiveId] = useState<string | null>(null);
  const scenario = useMemo(() => toScenario(params), [params]);
  const { result: run, error } = useCompute('probing', scenario);
  const setParam = (key: string, value: ParamValue) => setParams((p) => ({ ...p, [key]: value }));

  const apply = (exp: ProbingExperiment) => {
    setParams(fromScenario({ ...DEFAULT_PROBING, ...exp.scenario }));
    setActiveId(exp.id);
  };
  const stillActive = (() => {
    const exp = probingExperiments.find((e) => e.id === activeId);
    return (
      !!exp &&
      JSON.stringify(fromScenario({ ...DEFAULT_PROBING, ...exp.scenario })) ===
        JSON.stringify(params)
    );
  })();

  const stages: [string, string][] = [
    ['lab-probes', 'What each layer knows'],
    ['lab-control', 'The control task'],
    ['lab-how', 'How it works'],
    ['lab-controls', 'Settings'],
  ];

  return (
    <div className="mx-auto max-w-6xl px-4 pt-8 pb-20 sm:px-6 lg:px-8">
      <LabHeader
        eyebrow="Interpretability"
        title="Probing the layers"
        accent="what does each layer know?"
        blurb="To find out what a layer of a network represents, train the simplest classifier you can on its activations and see what it can read off. Follow facts as they pass through a digit network, and watch a story model move the gender of a name to the full stop that follows it."
        badges={
          <>
            <Badge>{scenario.model === 'digits' ? 'Digit network' : 'Story model'}</Badge>
            <Badge>linear probes</Badge>
          </>
        }
      />
      <StageNav stages={stages} />
      <ExperimentBar
        lessons={probingExperiments}
        activeId={stillActive ? activeId : null}
        onApply={apply}
      />

      <div className="flex flex-col gap-5">
        {run ? (
          <Results run={run} />
        ) : (
          <Panel
            id="lab-probes"
            className="scroll-mt-24"
            title="What each layer knows"
            eyebrow="01"
          >
            <p className="text-sm text-muted" role="status">
              {error ? `Something went wrong: ${error}` : 'Training the network and the probes…'}
            </p>
          </Panel>
        )}
        <Panel id="lab-how" className="scroll-mt-24" title="How it works" eyebrow="03">
          <div className="space-y-3 text-sm text-muted">
            <p>
              A probe is a softmax regression: one weight vector per class, applied to the
              activations of a layer and trained on one set of examples. Its accuracy on a different
              set says how readable the property is there. For the digit network the properties are
              which digit, how much ink and where the digit sits; for the story model they are the
              gender of the name and which object was found, read either at the name or at the full
              stop, and tested on names the probe never saw.
            </p>
            <p>
              A good score does not prove that the network uses the property, only that it is there,
              and a flexible probe can find patterns that are not. The control task, random labels,
              measures the probe&rsquo;s own capacity. The tests check the probe on separable and
              random data, and every sentence on this page.
            </p>
          </div>
        </Panel>
        <Panel id="lab-controls" className="scroll-mt-24" title="Settings" eyebrow="04">
          <ParamPanel specs={probingParams} values={params} onChange={setParam} columns={2} />
        </Panel>
      </div>
    </div>
  );
}
