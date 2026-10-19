import { useEffect, useMemo, useState } from 'react';
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
  pct,
  type Metric,
  type ParamValue,
  type ParamValues,
} from '@/shared/lab-kit';
import {
  DEFAULT_TOOLBOX_LAB,
  runToolbox,
  type ToolboxRun,
  type ToolboxScenario,
} from './toolboxCore';
import { toolboxExperiments, toolboxParams, type ToolboxExperiment } from './toolboxSpec';

const toScenario = (p: ParamValues): ToolboxScenario => ({
  ...DEFAULT_TOOLBOX_LAB,
  dataset: p.dataset as ToolboxScenario['dataset'],
  trainSize: Number(p.trainSize),
  noise: Number(p.noise),
  hidden: Number(p.hidden),
  depth: Number(p.depth),
  activation: p.activation as ToolboxScenario['activation'],
  init: p.init as ToolboxScenario['init'],
  dropout: Number(p.dropout),
  batchNorm: Boolean(p.batchNorm),
  weightDecay: Number(p.weightDecay),
  optimizer: p.optimizer as ToolboxScenario['optimizer'],
  learningRate: Number(p.learningRate),
  epochs: Number(p.epochs),
  patience: Number(p.patience),
});

const fromScenario = (s: ToolboxScenario): ParamValues => ({
  dataset: s.dataset,
  trainSize: s.trainSize,
  noise: s.noise,
  hidden: s.hidden,
  depth: s.depth,
  activation: s.activation,
  init: s.init,
  dropout: s.dropout,
  batchNorm: s.batchNorm,
  weightDecay: s.weightDecay,
  optimizer: s.optimizer,
  learningRate: s.learningRate,
  epochs: s.epochs,
  patience: s.patience,
});

function Regions({ run }: { run: ToolboxRun }) {
  const palette = usePalette();
  const { size, labels } = run.grid;
  const w = 420;
  const cell = w / size;
  const xs = run.train.x;
  let x0 = Infinity,
    x1 = -Infinity,
    y0 = Infinity,
    y1 = -Infinity;
  for (let i = 0; i < xs.rows; i++) {
    x0 = Math.min(x0, xs.get(i, 0));
    x1 = Math.max(x1, xs.get(i, 0));
    y0 = Math.min(y0, xs.get(i, 1));
    y1 = Math.max(y1, xs.get(i, 1));
  }
  const pad = 0.5;
  const sx = (v: number) => ((v - (x0 - pad)) / (x1 - x0 + 2 * pad)) * w;
  const sy = (v: number) => w - ((v - (y0 - pad)) / (y1 - y0 + 2 * pad)) * w;
  return (
    <svg
      viewBox={`0 0 ${w} ${w}`}
      className="w-full max-w-[26rem] rounded-xl border border-line"
      role="img"
      aria-label="What the trained network predicts across the plane, with the training points"
    >
      {Array.from(labels, (l, i) => (
        <rect
          key={i}
          x={(i % size) * cell}
          y={Math.floor(i / size) * cell}
          width={cell + 0.5}
          height={cell + 0.5}
          fill={palette.css.classes[l % palette.css.classes.length]}
          opacity={0.22}
        />
      ))}
      {run.train.y.map((c, i) => (
        <circle
          key={i}
          cx={sx(xs.get(i, 0))}
          cy={sy(xs.get(i, 1))}
          r={4}
          fill={palette.css.classes[c % palette.css.classes.length]}
          stroke={palette.css.surface}
          strokeWidth={1.2}
        />
      ))}
    </svg>
  );
}

export default function ToolboxPage() {
  const [params, setParams] = useState<ParamValues>(() => fromScenario(DEFAULT_TOOLBOX_LAB));
  const [activeId, setActiveId] = useState<string | null>(null);
  const [run, setRun] = useState<ToolboxRun | null>(null);
  const scenario = useMemo(() => toScenario(params), [params]);

  useEffect(() => {
    const timer = window.setTimeout(() => setRun(runToolbox(scenario)), 30);
    return () => window.clearTimeout(timer);
  }, [scenario]);

  const setParam = (key: string, value: ParamValue) => setParams((p) => ({ ...p, [key]: value }));
  const apply = (exp: ToolboxExperiment) => {
    setParams(fromScenario({ ...DEFAULT_TOOLBOX_LAB, ...exp.scenario }));
    setActiveId(exp.id);
  };
  const stillActive = (() => {
    const exp = toolboxExperiments.find((e) => e.id === activeId);
    return (
      !!exp &&
      JSON.stringify(fromScenario({ ...DEFAULT_TOOLBOX_LAB, ...exp.scenario })) ===
        JSON.stringify(params)
    );
  })();

  const h = run?.result.history ?? [];
  const metrics: Metric[] = run
    ? [
        { label: 'Accuracy on new data', value: pct(run.testAccuracy), testId: 'test-accuracy' },
        {
          label: 'Loss on new data',
          value: fix(run.testLoss, 3),
          hint: 'cross-entropy on 1,500 fresh points',
        },
        { label: 'Accuracy on training data', value: pct(run.finalTrainAccuracy) },
        {
          label: 'Gap',
          value: `${fix(run.gap * 100, 1)} pts`,
          tone: run.gap > 0.08 ? 'bad' : 'neutral',
          hint: 'training accuracy minus accuracy on new data: the mark of overfitting',
        },
        {
          label: 'Epochs run',
          value: `${h.length}${run.result.stoppedEarly ? ' (stopped early)' : ''}`,
          hint: run.result.stoppedEarly
            ? `best weights were from epoch ${run.result.bestEpoch}`
            : undefined,
        },
      ]
    : [];

  const stages: [string, string][] = [
    ['lab-curves', 'Learning curves'],
    ['lab-regions', 'What it learned'],
    ['lab-finder', 'Learning-rate finder'],
    ['lab-how', 'How it works'],
    ['lab-controls', 'Settings'],
  ];

  return (
    <div className="mx-auto max-w-6xl px-4 pt-8 pb-20 sm:px-6 lg:px-8">
      <LabHeader
        eyebrow="Deep learning"
        title="The training toolbox"
        accent="what makes networks train"
        blurb="Initialisation, dropout, batch normalisation, weight decay, early stopping and the learning rate: every trick that turns a network that can in principle learn into one that does. Switch each on and off and watch what it changes."
        badges={
          <>
            <Badge>
              {scenario.depth} hidden layers × {scenario.hidden}
            </Badge>
            <Badge>{scenario.trainSize} training points</Badge>
          </>
        }
      />
      <StageNav stages={stages} />
      <ExperimentBar
        lessons={toolboxExperiments}
        activeId={stillActive ? activeId : null}
        onApply={apply}
      />

      <div className="flex flex-col gap-5">
        <Panel id="lab-curves" className="scroll-mt-24" title="Learning curves" eyebrow="01">
          {run ? (
            <div className="space-y-6">
              <MetricsStrip metrics={metrics} className="lg:grid-cols-5" />
              <div className="grid gap-8 lg:grid-cols-2">
                <MiniChart
                  title="Loss"
                  x={h.map((e) => e.epoch)}
                  xLabel="epoch"
                  series={[
                    { name: 'training', color: 'var(--brand)', values: h.map((e) => e.trainLoss) },
                    {
                      name: 'held-out',
                      color: 'var(--neg)',
                      values: h.map((e) => e.validationLoss),
                    },
                  ]}
                  height={190}
                />
                <MiniChart
                  title="Accuracy"
                  x={h.map((e) => e.epoch)}
                  xLabel="epoch"
                  yRange={{ lo: 0.4, hi: 1 }}
                  series={[
                    {
                      name: 'training',
                      color: 'var(--brand)',
                      values: h.map((e) => e.trainAccuracy),
                    },
                    {
                      name: 'held-out',
                      color: 'var(--neg)',
                      values: h.map((e) => e.validationAccuracy),
                    },
                  ]}
                  height={190}
                />
              </div>
              <p className="text-sm text-muted" data-testid="curve-note">
                The blue curve is the data the network trains on; the red one is held-out data it
                never learns from. When red turns upward while blue keeps falling, the network has
                started to memorise noise: that is where early stopping would stop.
              </p>
            </div>
          ) : (
            <p className="text-sm text-muted">Training…</p>
          )}
        </Panel>
        <Panel id="lab-regions" className="scroll-mt-24" title="What it learned" eyebrow="02">
          {run && (
            <div className="grid gap-8 lg:grid-cols-[auto_minmax(0,1fr)]">
              <Regions run={run} />
              <p className="text-sm text-muted">
                The coloured regions are what the finished network predicts at every point of the
                plane; the dots are its {scenario.trainSize} training points. An overfit network
                carves out islands around individual noisy points; a well-regularised one draws a
                smooth boundary.
              </p>
            </div>
          )}
        </Panel>
        <Panel id="lab-finder" className="scroll-mt-24" title="Learning-rate finder" eyebrow="03">
          {run && (
            <div className="grid gap-8 lg:grid-cols-[minmax(0,1fr)_20rem]">
              <MiniChart
                title="Smoothed loss while the learning rate rises (log scale)"
                x={run.finder.curve.map((p) => p.learningRate)}
                xLabel="learning rate"
                series={[
                  {
                    name: 'loss',
                    color: 'var(--brand)',
                    values: run.finder.curve.map((p) => Math.max(p.loss, 1e-4)),
                  },
                ]}
                log
                height={200}
              />
              <p className="text-sm text-muted" data-testid="finder-note">
                A fresh network is trained for 150 mini-batches while the rate climbs from 0.0001
                and the loss is recorded. It falls while the rate is useful, bottoms out and then
                explodes. Suggested rate:{' '}
                <strong className="text-fg">{run.finder.suggested.toPrecision(2)}</strong> (a tenth
                of the rate at the lowest loss). You are using {scenario.learningRate}.
              </p>
            </div>
          )}
        </Panel>
        <Panel id="lab-how" className="scroll-mt-24" title="How it works" eyebrow="04">
          <div className="space-y-3 text-sm text-muted">
            <p>
              <strong className="text-fg">Initialisation</strong> sets the starting weights. All
              zeros keeps every unit identical; weights too small make the signal fade layer by
              layer; too large saturate the activations. Xavier (variance 2/(fan-in + fan-out))
              suits tanh, He (2/fan-in) suits ReLU.
            </p>
            <p>
              <strong className="text-fg">Dropout</strong> zeroes each hidden unit with probability
              p during training (scaling the rest by 1/(1−p) so the expected activation is
              unchanged) and does nothing at test time.{' '}
              <strong className="text-fg">Batch normalisation</strong> standardises each unit over
              the mini-batch, then applies a learned scale and shift; at test time it uses running
              averages. <strong className="text-fg">Weight decay</strong> shrinks the weights a
              little every step. <strong className="text-fg">Early stopping</strong> keeps the
              weights from the epoch with the lowest held-out loss.
            </p>
            <p>
              The hand-written backpropagation, including batch normalisation, is checked against
              finite differences in the tests.
            </p>
          </div>
        </Panel>
        <Panel id="lab-controls" className="scroll-mt-24" title="Settings" eyebrow="05">
          <ParamPanel specs={toolboxParams} values={params} onChange={setParam} columns={3} />
        </Panel>
      </div>
    </div>
  );
}
