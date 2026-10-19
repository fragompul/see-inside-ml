import { useMemo, useState } from 'react';
import { Badge, Panel } from '@/design-system';
import { usePalette } from '@/shared/canvas/colors';
import { ExperimentBar } from '@/shared/components/ExperimentBar';
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
import { windowSpan } from '@sim/deep';
import {
  DEFAULT_CONV_ARITH,
  DESIGN_LABEL,
  runConvArith,
  type ConvArithScenario,
} from './convArithCore';
import { convArithExperiments, convArithParams, type ConvArithExperiment } from './convArithSpec';

const toScenario = (p: ParamValues): ConvArithScenario => ({
  design: p.design as ConvArithScenario['design'],
  depth: Number(p.depth),
  kernel: Number(p.kernel),
  input: Number(p.input),
});

const fromScenario = (s: ConvArithScenario): ParamValues => ({ ...s });

const WIDTH = 640;
const ROW = 32;

export default function ConvArithPage() {
  const palette = usePalette();
  const [params, setParams] = useState<ParamValues>(() => fromScenario(DEFAULT_CONV_ARITH));
  const [activeId, setActiveId] = useState<string | null>(null);
  const scenario = useMemo(() => toScenario(params), [params]);
  const run = useMemo(() => runConvArith(scenario), [scenario]);
  const setParam = (key: string, value: ParamValue) => setParams((p) => ({ ...p, [key]: value }));

  const apply = (exp: ConvArithExperiment) => {
    setParams(fromScenario({ ...DEFAULT_CONV_ARITH, ...exp.scenario }));
    setActiveId(exp.id);
  };
  const stillActive = (() => {
    const exp = convArithExperiments.find((e) => e.id === activeId);
    return (
      !!exp &&
      JSON.stringify(fromScenario({ ...DEFAULT_CONV_ARITH, ...exp.scenario })) ===
        JSON.stringify(params)
    );
  })();

  const metrics: Metric[] = [
    {
      label: 'Positions each output sees',
      value: String(run.finalField),
      testId: 'field',
      hint: `${fix(run.coverage * 100, 0)}% of the input`,
    },
    { label: 'Output width', value: String(run.finalSize), hint: `from ${scenario.input}` },
    {
      label: 'Weights (16 channels, image)',
      value: run.parameters.toLocaleString('en-US'),
    },
    {
      label: 'One filter that wide',
      value: run.singleKernelParameters.toLocaleString('en-US'),
      hint: 'same reach, one layer',
    },
  ];

  const cw = WIDTH / scenario.input;
  // how many cells of each layer feed the central output at the top of the stack
  const feeding = (() => {
    const out = [1];
    for (let l = run.layers.length; l >= 1; l--) {
      const w = (run.layers[l - 1] as (typeof run.layers)[number]).window;
      out.unshift(((out[0] as number) - 1) * w.stride + windowSpan(w));
    }
    return out;
  })();
  const rows = run.stats.sizes.length;
  const height = rows * ROW + 8;
  const centre = WIDTH / 2;
  const stages: [string, string][] = [
    ['lab-stack', 'The stack'],
    ['lab-how', 'How it works'],
    ['lab-controls', 'Settings'],
  ];

  return (
    <div className="mx-auto max-w-6xl px-4 pt-8 pb-20 sm:px-6 lg:px-8">
      <LabHeader
        eyebrow="Deep learning"
        title="Convolution arithmetic"
        accent="how far a layer can see"
        blurb="Each output of a convolutional layer reads a small window of the layer before. Stack layers and the window grows, but how fast depends on kernel width, stride, pooling and dilation. Build a stack and watch the size of the output, the part of the input it can see, and the weights it costs."
        badges={
          <>
            <Badge>{DESIGN_LABEL[scenario.design]}</Badge>
            <Badge>{scenario.depth} convolutions</Badge>
          </>
        }
      />
      <StageNav stages={stages} />
      <ExperimentBar
        lessons={convArithExperiments}
        activeId={stillActive ? activeId : null}
        onApply={apply}
      />

      <div className="flex flex-col gap-5">
        <Panel id="lab-stack" className="scroll-mt-24" title="The stack" eyebrow="01">
          <div className="space-y-6">
            <MetricsStrip metrics={metrics} className="lg:grid-cols-4" />
            <svg
              viewBox={`0 0 ${WIDTH + 130} ${height}`}
              className="w-full rounded-xl border border-line"
              role="img"
              aria-label="Each layer as a row of cells, with the part of it that feeds the central output highlighted"
            >
              {[...run.stats.sizes.keys()].reverse().map((l, row) => {
                const size = run.stats.sizes[l] as number;
                const step = cw * (run.stats.jump[l] as number);
                const field = (feeding[l] as number) * step;
                const y = row * ROW + 4;
                const name = l === 0 ? 'input' : (run.layers[l - 1]?.name ?? '');
                return (
                  <g key={l}>
                    <rect
                      x={Math.max(0, centre - field / 2)}
                      y={y}
                      width={Math.min(WIDTH, field)}
                      height={ROW - 4}
                      fill={palette.css.brand}
                      opacity={0.5}
                    />
                    {Array.from({ length: size }, (_, i) => (
                      <rect
                        key={i}
                        x={i * step + 0.5}
                        y={y + 2}
                        width={Math.max(1, step - 1)}
                        height={ROW - 8}
                        fill={palette.css.muted}
                        opacity={0.45}
                      />
                    ))}
                    <text x={WIDTH + 8} y={y + 14} className="fill-muted font-mono text-[10px]">
                      {name} · {size}
                    </text>
                  </g>
                );
              })}
            </svg>
            <p className="text-sm text-muted">
              Each row is a layer, the input at the bottom. A cell is one value; where cells are
              wide the layer has fewer of them. The shaded band shows how much of that layer feeds
              the central output at the top: it narrows as you go up the stack and widens as you go
              down. Only one axis is drawn; an image behaves the same way along both.
            </p>
          </div>
        </Panel>
        <Panel id="lab-how" className="scroll-mt-24" title="How it works" eyebrow="02">
          <div className="space-y-3 text-sm text-muted">
            <p>
              A layer with kernel <em>k</em>, stride <em>s</em>, padding <em>p</em> and dilation{' '}
              <em>d</em> turns <em>n</em> values into <em>⌊(n + 2p − d(k − 1) − 1) / s⌋ + 1</em>.
              Its reach grows by <em>d(k − 1)</em> times the distance between neighbouring outputs
              of the layer before, and that distance is multiplied by the stride, so strides and
              pooling make every later layer see more.
            </p>
            <p>
              The tests check the size formula against counting window positions for hundreds of
              settings, and the reach formula against following one output back through the stack. A
              value near the edge sees less, because part of its window is padding.
            </p>
          </div>
        </Panel>
        <Panel id="lab-controls" className="scroll-mt-24" title="Settings" eyebrow="03">
          <ParamPanel specs={convArithParams} values={params} onChange={setParam} columns={2} />
        </Panel>
      </div>
    </div>
  );
}
