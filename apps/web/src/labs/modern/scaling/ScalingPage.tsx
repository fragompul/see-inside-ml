import { useMemo, useState } from 'react';
import { Badge, Panel } from '@/design-system';
import { useCompute } from '@/shared/compute/useCompute';
import { inkOn, mix, usePalette } from '@/shared/canvas/colors';
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
  DATA,
  DEFAULT_SCALING,
  FEATURES,
  type ScalingRun,
  type ScalingScenario,
} from './scalingCore';
import { scalingExperiments, scalingParams, type ScalingExperiment } from './scalingSpec';

const toScenario = (p: ParamValues): ScalingScenario => ({
  noise: Number(p.noise),
  seed: Number(p.seed),
});

const fromScenario = (s: ScalingScenario): ParamValues => ({ ...s });

type PlainLaw = Omit<ScalingRun['dataLaw'], 'predict'>;
type ScalingView = Omit<ScalingRun, 'dataLaw' | 'featureLaw'> & {
  dataLaw: PlainLaw;
  featureLaw: PlainLaw;
};

function Results({ run }: { run: ScalingView }) {
  const palette = usePalette();
  const all = run.loss.flat();
  const lo = Math.min(...all);
  const hi = Math.min(Math.max(...all), 1.2);
  const shade = (v: number) => {
    const t = Math.max(
      0,
      Math.min(1, (Math.log(v) - Math.log(lo)) / (Math.log(hi) - Math.log(lo))),
    );
    return `color-mix(in srgb, ${palette.css.pos} ${Math.round((1 - t) * 70)}%, ${palette.css.surface})`;
  };
  const ink = (v: number) => {
    const t = Math.max(
      0,
      Math.min(1, (Math.log(v) - Math.log(lo)) / (Math.log(hi) - Math.log(lo))),
    );
    return inkOn(mix(palette.surface, palette.pos, (1 - t) * 0.7));
  };

  const metrics: Metric[] = [
    {
      label: 'Data exponent',
      value: fix(run.dataLaw.alpha, 2),
      testId: 'data-exponent',
      hint: `largest model, fit R² ${fix(run.dataLaw.r2, 2)}`,
    },
    {
      label: 'Model-size exponent',
      value: fix(run.featureLaw.alpha, 2),
      hint: `most data, fit R² ${fix(run.featureLaw.r2, 2)}`,
    },
    { label: 'Noise floor', value: fix(run.floor, 3), hint: 'the best any model can do' },
    { label: 'Best loss reached', value: fix(lo, 3) },
  ];

  const lastRow = run.loss[run.loss.length - 1] as number[];
  const sizeRow = run.loss.map((row) => row[row.length - 1] as number);
  return (
    <>
      <Panel id="lab-grid" className="scroll-mt-24" title="Loss by size and data" eyebrow="01">
        <div className="space-y-6">
          <MetricsStrip metrics={metrics} className="lg:grid-cols-4" />
          <div className="overflow-x-auto">
            <table className="w-full min-w-[34rem] border-separate border-spacing-1 text-center font-mono text-xs">
              <caption className="pb-2 text-left font-sans text-sm text-muted">
                Test loss. Rows are model sizes (random features), columns are training examples.
              </caption>
              <thead>
                <tr>
                  <th className="font-normal text-muted" scope="col">
                    features ↓ / examples →
                  </th>
                  {DATA.map((d) => (
                    <th key={d} scope="col" className="font-normal text-muted">
                      {d}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {run.loss.map((row, i) => (
                  <tr key={i}>
                    <th scope="row" className="font-normal text-muted">
                      {FEATURES[i]}
                    </th>
                    {row.map((v, j) => (
                      <td
                        key={j}
                        className="rounded-md py-2"
                        style={{ background: shade(v), color: ink(v) }}
                      >
                        {fix(v, 3)}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </Panel>
      <Panel id="lab-laws" className="scroll-mt-24" title="The two laws" eyebrow="02">
        <div className="grid gap-6 lg:grid-cols-2">
          <MiniChart
            title="Excess loss against examples (log-log), largest model"
            x={DATA}
            xLabel="training examples"
            log
            series={[
              {
                name: 'measured',
                color: palette.css.brand,
                values: lastRow.map((v) => v - run.floor),
              },
              {
                name: 'power law',
                color: palette.css.pos,
                values: DATA.map((d) => lawAt(run.dataLaw, d) - run.floor),
                dashed: true,
              },
            ]}
            height={230}
          />
          <MiniChart
            title="Excess loss against model size (log-log), most data"
            x={FEATURES}
            xLabel="random features"
            log
            series={[
              {
                name: 'measured',
                color: palette.css.brand,
                values: sizeRow.map((v) => v - run.floor),
              },
              {
                name: 'power law',
                color: palette.css.pos,
                values: FEATURES.map((n) => lawAt(run.featureLaw, n) - run.floor),
                dashed: true,
              },
            ]}
            height={230}
          />
        </div>
      </Panel>
      <Panel id="lab-budget" className="scroll-mt-24" title="Spending a budget" eyebrow="03">
        <div className="space-y-3">
          <div className="overflow-x-auto">
            <table className="w-full text-left text-sm">
              <thead className="text-muted">
                <tr>
                  <th className="py-2 pr-4 font-medium">Budget (features × examples)</th>
                  <th className="py-2 pr-4 font-medium">Best model</th>
                  <th className="py-2 pr-4 font-medium">Best data set</th>
                  <th className="py-2 font-medium">Loss</th>
                </tr>
              </thead>
              <tbody>
                {run.frontier.map((f) => (
                  <tr key={f.budget} className="border-t border-line">
                    <td className="py-2 pr-4 font-mono">{f.budget.toLocaleString('en-US')}</td>
                    <td className="py-2 pr-4 font-mono">{f.features} features</td>
                    <td className="py-2 pr-4 font-mono">{f.data} examples</td>
                    <td className="py-2 font-mono">{fix(f.loss, 3)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="text-sm text-muted">
            For each budget, the grid cell with the lowest loss whose features times examples fit
            inside it.
          </p>
        </div>
      </Panel>
    </>
  );
}

const lawAt = (law: PlainLaw, x: number) => law.c + law.a * x ** -law.alpha;

export default function ScalingPage() {
  const [params, setParams] = useState<ParamValues>(() => fromScenario(DEFAULT_SCALING));
  const [activeId, setActiveId] = useState<string | null>(null);
  const scenario = useMemo(() => toScenario(params), [params]);
  const { result: run } = useCompute('scaling', scenario);
  const setParam = (key: string, value: ParamValue) => setParams((p) => ({ ...p, [key]: value }));

  const apply = (exp: ScalingExperiment) => {
    setParams(fromScenario({ ...DEFAULT_SCALING, ...exp.scenario }));
    setActiveId(exp.id);
  };
  const stillActive = (() => {
    const exp = scalingExperiments.find((e) => e.id === activeId);
    return (
      !!exp &&
      JSON.stringify(fromScenario({ ...DEFAULT_SCALING, ...exp.scenario })) ===
        JSON.stringify(params)
    );
  })();

  const stages: [string, string][] = [
    ['lab-grid', 'Loss by size and data'],
    ['lab-laws', 'The two laws'],
    ['lab-budget', 'Spending a budget'],
    ['lab-how', 'How it works'],
    ['lab-controls', 'Settings'],
  ];

  return (
    <div className="mx-auto max-w-6xl px-4 pt-8 pb-20 sm:px-6 lg:px-8">
      <LabHeader
        eyebrow="Modern topics"
        title="Scaling laws"
        accent="predicting the payoff of scale"
        blurb="Across language models, error falls as a smooth power law in the size of the model and the amount of data, and labs plan huge training runs from that. Here the same behaviour appears in a model small enough to solve exactly: fit the laws, see what each limit costs, and decide how to spend a budget."
        badges={
          <>
            <Badge>noise {scenario.noise}</Badge>
            <Badge>{FEATURES.length * DATA.length} models trained</Badge>
          </>
        }
      />
      <StageNav stages={stages} />
      <ExperimentBar
        lessons={scalingExperiments}
        activeId={stillActive ? activeId : null}
        onApply={apply}
      />

      <div className="flex flex-col gap-5">
        {run ? (
          <Results run={run} />
        ) : (
          <Panel id="lab-grid" className="scroll-mt-24" title="Loss by size and data" eyebrow="01">
            <p className="text-sm text-muted" role="status">
              Fitting the models…
            </p>
          </Panel>
        )}
        <Panel id="lab-how" className="scroll-mt-24" title="How it works" eyebrow="04">
          <div className="space-y-3 text-sm text-muted">
            <p>
              Each model is a ridge regression on <em>N</em> random cosine features of five inputs,
              fitted exactly on <em>D</em> noisy examples of a target made of 150 waves whose
              strengths fall as 1/<em>j</em>. The target never ends, so there is always more to
              learn, and the error above the noise floor falls as a power of each of <em>N</em> and{' '}
              <em>D</em>. The laws are fitted in log-log space with the floor, which is known, taken
              out.
            </p>
            <p>
              Real scaling laws have the same form with an unknown floor and a joint dependence on
              model size, data and compute. The tests check the fitting code on exact power laws and
              run every experiment above, including the one about the budget.
            </p>
          </div>
        </Panel>
        <Panel id="lab-controls" className="scroll-mt-24" title="Settings" eyebrow="05">
          <ParamPanel specs={scalingParams} values={params} onChange={setParam} columns={2} />
        </Panel>
      </div>
    </div>
  );
}
