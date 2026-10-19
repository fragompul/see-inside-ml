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
  DEFAULT_MOE_LAB,
  DESIGN_LABEL,
  GRID,
  REGIONS,
  type Design,
  type MoeResult,
  type MoeScenario,
} from './moeCore';
import { moeExperiments, moeParams, type MoeExperiment } from './moeSpec';

const toScenario = (p: ParamValues): MoeScenario => ({
  ...DEFAULT_MOE_LAB,
  design: p.design as Design,
  seed: Number(p.seed),
});

const fromScenario = (s: MoeScenario): ParamValues => ({ design: s.design, seed: s.seed });

const HUES = [20, 150, 260, 330];
const SIZE = 312;

function Results({ run }: { run: MoeResult }) {
  const palette = usePalette();
  const s = run.scenario;
  const experts = run.share.length;
  const cell = SIZE / GRID;
  const designs = Object.keys(DESIGN_LABEL) as Design[];
  const metrics: Metric[] = [
    {
      label: 'Test error',
      value: fix(run.mse[s.design], 3),
      testId: 'mse',
      hint: DESIGN_LABEL[s.design],
    },
    {
      label: 'Weights used per input',
      value: `${run.active[s.design]} of ${run.total[s.design]}`,
    },
    {
      label: 'Evenness of the load',
      value: experts > 1 ? `${fix(run.entropy, 2)} of ${fix(Math.log(experts), 2)}` : '–',
      hint: 'entropy of the share of inputs per expert',
    },
  ];
  return (
    <>
      <Panel id="lab-routing" className="scroll-mt-24" title="Who handles what" eyebrow="01">
        <div className="space-y-6">
          <MetricsStrip metrics={metrics} className="lg:grid-cols-3" />
          {experts > 1 ? (
            <div className="flex flex-wrap items-start gap-8">
              <figure className="space-y-2">
                <svg
                  viewBox={`0 0 ${SIZE} ${SIZE}`}
                  className="w-full max-w-[20rem] rounded-xl border border-line"
                  role="img"
                  aria-label="Which expert handles each point of the square"
                >
                  {run.map.map((e, i) => (
                    <rect
                      key={i}
                      x={(i % GRID) * cell}
                      y={Math.floor(i / GRID) * cell}
                      width={cell + 0.5}
                      height={cell + 0.5}
                      fill={`hsl(${HUES[e % HUES.length]} 60% 55%)`}
                      opacity={0.75}
                    />
                  ))}
                  <line
                    x1={SIZE / 2}
                    y1={0}
                    x2={SIZE / 2}
                    y2={SIZE}
                    stroke={palette.css.fg}
                    strokeOpacity={0.5}
                    strokeDasharray="4 4"
                  />
                  <line
                    x1={0}
                    y1={SIZE / 2}
                    x2={SIZE}
                    y2={SIZE / 2}
                    stroke={palette.css.fg}
                    strokeOpacity={0.5}
                    strokeDasharray="4 4"
                  />
                </svg>
                <figcaption className="text-xs text-muted">
                  Colour = the expert with the highest gate score. Dashed lines separate the four
                  regions, each with a different function.
                </figcaption>
              </figure>
              <div className="min-w-[14rem] flex-1 space-y-3">
                <p className="text-sm text-fg">Share of inputs per expert</p>
                {run.share.map((v, e) => (
                  <div key={e} className="space-y-1">
                    <div className="flex justify-between font-mono text-xs text-muted">
                      <span>expert {e + 1}</span>
                      <span>{fix(v * 100, 0)}%</span>
                    </div>
                    <div className="h-2 rounded-full bg-line">
                      <div
                        className="h-2 rounded-full"
                        style={{
                          width: `${v * 100}%`,
                          background: `hsl(${HUES[e % HUES.length]} 60% 55%)`,
                        }}
                      />
                    </div>
                  </div>
                ))}
                <p className="pt-2 text-xs text-muted">
                  Favourite expert of each region and its share of that region:{' '}
                  {Array.from(
                    { length: REGIONS },
                    (_, r) =>
                      `${r + 1}: expert ${(run.favourite[r] as number) + 1} (${fix((run.purity[r] as number) * 100, 0)}%)`,
                  ).join(', ')}
                  .
                </p>
              </div>
            </div>
          ) : (
            <p className="text-sm text-muted">
              A single dense network has no routing: every input goes through every unit.
            </p>
          )}
        </div>
      </Panel>
      <Panel id="lab-compare" className="scroll-mt-24" title="Five designs compared" eyebrow="02">
        <div className="space-y-6">
          <div className="overflow-x-auto">
            <table className="w-full text-left text-sm">
              <thead className="text-muted">
                <tr>
                  <th className="py-2 pr-4 font-medium">Design</th>
                  <th className="py-2 pr-4 font-medium">Test error</th>
                  <th className="py-2 pr-4 font-medium">Weights stored</th>
                  <th className="py-2 font-medium">Weights per input</th>
                </tr>
              </thead>
              <tbody>
                {designs.map((d) => (
                  <tr key={d} className="border-t border-line">
                    <td className="py-2 pr-4">{DESIGN_LABEL[d]}</td>
                    <td className="py-2 pr-4 font-mono">{fix(run.mse[d], 3)}</td>
                    <td className="py-2 pr-4 font-mono">{run.total[d]}</td>
                    <td className="py-2 font-mono">{run.active[d]}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <MiniChart
            title="Test error of the model shown during training"
            x={run.history.map((h) => h.step)}
            xLabel="training step"
            series={[
              { name: 'error', color: palette.css.brand, values: run.history.map((h) => h.mse) },
            ]}
            height={200}
          />
        </div>
      </Panel>
    </>
  );
}

export default function MoePage() {
  const [params, setParams] = useState<ParamValues>(() => fromScenario(DEFAULT_MOE_LAB));
  const [activeId, setActiveId] = useState<string | null>(null);
  const scenario = useMemo(() => toScenario(params), [params]);
  const { result: run, error } = useCompute('moe', scenario);
  const setParam = (key: string, value: ParamValue) => setParams((p) => ({ ...p, [key]: value }));

  const apply = (exp: MoeExperiment) => {
    setParams(fromScenario({ ...DEFAULT_MOE_LAB, ...exp.scenario }));
    setActiveId(exp.id);
  };
  const stillActive = (() => {
    const exp = moeExperiments.find((e) => e.id === activeId);
    return (
      !!exp &&
      JSON.stringify(fromScenario({ ...DEFAULT_MOE_LAB, ...exp.scenario })) ===
        JSON.stringify(params)
    );
  })();

  const stages: [string, string][] = [
    ['lab-routing', 'Who handles what'],
    ['lab-compare', 'Five designs compared'],
    ['lab-how', 'How it works'],
    ['lab-controls', 'Settings'],
  ];

  return (
    <div className="mx-auto max-w-6xl px-4 pt-8 pb-20 sm:px-6 lg:px-8">
      <LabHeader
        eyebrow="Modern topics"
        title="Mixture of experts"
        accent="more capacity, same work per input"
        blurb="Instead of sending every input through every weight, a mixture of experts uses a small gate to pick which few of many specialist networks should answer. Capacity grows with the number of experts while the work per input stays flat, provided the gate shares the load and the experts find something to specialise in."
        badges={
          <>
            <Badge>{DESIGN_LABEL[scenario.design]}</Badge>
            <Badge>four regions, four functions</Badge>
          </>
        }
      />
      <StageNav stages={stages} />
      <ExperimentBar
        lessons={moeExperiments}
        activeId={stillActive ? activeId : null}
        onApply={apply}
      />

      <div className="flex flex-col gap-5">
        {run ? (
          <Results run={run} />
        ) : (
          <Panel id="lab-routing" className="scroll-mt-24" title="Who handles what" eyebrow="01">
            <p className="text-sm text-muted" role="status">
              {error ? `Something went wrong: ${error}` : 'Training five models…'}
            </p>
          </Panel>
        )}
        <Panel id="lab-how" className="scroll-mt-24" title="How it works" eyebrow="03">
          <div className="space-y-3 text-sm text-muted">
            <p>
              The task is to predict a number from two inputs; the square of inputs is cut into four
              quadrants and each has its own function. A gate (one linear layer and a softmax)
              scores the four experts, each a tiny network with a few hidden units. Only the best
              one or two run. With two running, their outputs are mixed using the gate probabilities
              renormalised between them; with one, the output is scaled by its gate probability so
              that the gate still receives a signal from the task.
            </p>
            <p>
              A balance term in the loss multiplies the share of inputs sent to each expert by the
              average gate probability of that expert; it is smallest when the load is even. Without
              it the gate tends to favour whichever expert starts ahead, which then trains more and
              pulls further ahead. The tests check the gradients with one, two and all four experts
              running, and the balance term at its two extremes.
            </p>
          </div>
        </Panel>
        <Panel id="lab-controls" className="scroll-mt-24" title="Settings" eyebrow="04">
          <ParamPanel specs={moeParams} values={params} onChange={setParam} columns={2} />
        </Panel>
      </div>
    </div>
  );
}
