import { Dices } from 'lucide-react';
import { useMemo, useState } from 'react';
import { Badge, Button, Panel, Slider, Switch } from '@/design-system';
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
import {
  DEFAULT_TRACKING,
  runTracking,
  type TrackingRun,
  type TrackingScenario,
} from './trackingCore';
import { trackingExperiments, trackingParams, type TrackingExperiment } from './trackingSpec';

const toScenario = (p: ParamValues, seed: number): TrackingScenario => ({
  ...DEFAULT_TRACKING,
  motion: p.motion as TrackingScenario['motion'],
  sensorNoise: Number(p.sensorNoise),
  truePush: Number(p.truePush),
  dropout: Number(p.dropout),
  outliers: Number(p.outliers),
  assumedPush: Number(p.assumedPush),
  particleLikelihood: p.particleLikelihood as TrackingScenario['particleLikelihood'],
  particles: Number(p.particles),
  seed,
});

const fromScenario = (s: TrackingScenario): ParamValues => ({
  motion: s.motion,
  sensorNoise: s.sensorNoise,
  truePush: s.truePush,
  dropout: s.dropout,
  outliers: s.outliers,
  assumedPush: s.assumedPush,
  particleLikelihood: s.particleLikelihood,
  particles: s.particles,
});

type Layer = 'readings' | 'kalman' | 'smoother' | 'particles';

/** The 2σ ellipse of a 2×2 covariance, as an SVG path in pixel space. */
function ellipsePath(
  cx: number,
  cy: number,
  cov: { get: (r: number, c: number) => number },
  sx: (v: number) => number,
  sy: (v: number) => number,
  scale: number,
): string {
  const a = cov.get(0, 0);
  const b = cov.get(0, 1);
  const d = cov.get(1, 1);
  const tr = a + d;
  const det = a * d - b * b;
  const disc = Math.sqrt(Math.max((tr * tr) / 4 - det, 0));
  const l1 = tr / 2 + disc;
  const l2 = Math.max(tr / 2 - disc, 1e-12);
  const theta = 0.5 * Math.atan2(2 * b, a - d);
  const pts: string[] = [];
  for (let i = 0; i <= 40; i++) {
    const t = (i / 40) * 2 * Math.PI;
    const ex = 2 * Math.sqrt(l1) * Math.cos(t);
    const ey = 2 * Math.sqrt(l2) * Math.sin(t);
    const x = cx + ex * Math.cos(theta) - ey * Math.sin(theta);
    const y = cy + ex * Math.sin(theta) + ey * Math.cos(theta);
    pts.push(`${i === 0 ? 'M' : 'L'}${sx(x).toFixed(1)},${sy(y).toFixed(1)}`);
  }
  void scale;
  return pts.join(' ');
}

function Scene({ run }: { run: TrackingRun }) {
  const palette = usePalette();
  const [t, setT] = useState(40);
  const [layers, setLayers] = useState<Record<Layer, boolean>>({
    readings: true,
    kalman: true,
    smoother: false,
    particles: true,
  });
  const n = run.truth.length;
  const tt = Math.min(t, n - 1);
  const w = 760;
  const h = 460;
  const frame = useMemo(() => {
    const xs: number[] = [];
    const ys: number[] = [];
    run.truth.forEach(([x, y]) => (xs.push(x), ys.push(y)));
    run.readings.forEach((r) => r && (xs.push(r[0]), ys.push(r[1])));
    const x0 = Math.min(...xs);
    const x1 = Math.max(...xs);
    const y0 = Math.min(...ys);
    const y1 = Math.max(...ys);
    const pad = 24;
    const scale = Math.min((w - 2 * pad) / (x1 - x0 || 1), (h - 2 * pad) / (y1 - y0 || 1));
    return {
      sx: (v: number) => pad + (v - x0) * scale,
      sy: (v: number) => h - pad - (v - y0) * scale,
    };
  }, [run]);
  const { sx, sy } = frame;
  const path = (pts: readonly (readonly number[])[], upTo = pts.length) =>
    pts
      .slice(0, upTo)
      .map(
        (p, i) =>
          `${i ? 'L' : 'M'}${sx(p[0] as number).toFixed(1)},${sy(p[1] as number).toFixed(1)}`,
      )
      .join(' ');
  const k = run.kalman[tt]!;
  const pf = run.particles[tt]!;
  const toggle = (l: Layer) => setLayers((s) => ({ ...s, [l]: !s[l] }));
  const maxW = Math.max(...pf.weights);
  const drawn = pf.particles.length > 600 ? Math.ceil(pf.particles.length / 600) : 1;
  return (
    <div className="grid gap-8 lg:grid-cols-[minmax(0,1fr)_16rem]">
      <div className="space-y-4">
        <svg
          viewBox={`0 0 ${w} ${h}`}
          className="w-full rounded-xl border border-line"
          role="img"
          aria-label="Top view: the true path, the sensor readings and the estimates of the filters"
        >
          <path
            d={path(run.truth)}
            fill="none"
            stroke={palette.css.fg}
            strokeWidth={1.4}
            strokeDasharray="5 4"
            opacity={0.7}
          />
          {layers.readings &&
            run.readings.map((r, i) =>
              r ? (
                <circle
                  key={i}
                  cx={sx(r[0])}
                  cy={sy(r[1])}
                  r={i === tt ? 5 : 2.6}
                  fill={palette.css.muted}
                  opacity={i <= tt ? 0.75 : 0.2}
                />
              ) : null,
            )}
          {layers.particles &&
            pf.particles.map((p, i) =>
              i % drawn === 0 ? (
                <circle
                  key={i}
                  cx={sx(p[0] as number)}
                  cy={sy(p[1] as number)}
                  r={1.6}
                  fill={palette.css.neg}
                  opacity={0.15 + 0.6 * ((pf.weights[i] as number) / maxW)}
                />
              ) : null,
            )}
          {layers.smoother && (
            <path
              d={path(run.smoothed.map((s) => s.mean))}
              fill="none"
              stroke={palette.css.pos}
              strokeWidth={2}
            />
          )}
          {layers.kalman && (
            <>
              <path
                d={path(
                  run.kalman.map((s) => s.mean),
                  tt + 1,
                )}
                fill="none"
                stroke={palette.css.brand}
                strokeWidth={2.2}
              />
              <path
                d={ellipsePath(k.mean[0] as number, k.mean[1] as number, k.cov, sx, sy, 1)}
                fill={palette.css.brand}
                fillOpacity={0.14}
                stroke={palette.css.brand}
                strokeWidth={1}
              />
            </>
          )}
          <circle
            cx={sx(run.truth[tt]![0])}
            cy={sy(run.truth[tt]![1])}
            r={5}
            fill="none"
            stroke={palette.css.fg}
            strokeWidth={2}
          />
        </svg>
        <Slider label="Time step" min={0} max={n - 1} value={tt} onValueChange={setT} />
      </div>
      <div className="space-y-4">
        <p className="eyebrow">Show</p>
        <Switch
          label="Sensor readings"
          checked={layers.readings}
          onCheckedChange={() => toggle('readings')}
        />
        <Switch
          label="Kalman filter + 2σ ellipse"
          checked={layers.kalman}
          onCheckedChange={() => toggle('kalman')}
        />
        <Switch
          label="Particle cloud"
          checked={layers.particles}
          onCheckedChange={() => toggle('particles')}
        />
        <Switch
          label="Smoothed track"
          checked={layers.smoother}
          onCheckedChange={() => toggle('smoother')}
        />
        <p className="text-[0.78rem] text-muted" data-testid="scene-note">
          Dashed line: the true path (ring = where the target is at this step). Grey dots: readings
          so far. The ellipse is the Kalman filter&apos;s uncertainty about the position: it shrinks
          as readings arrive and swells when they are lost. The red cloud is the particle
          filter&apos;s belief at this step (brighter = heavier). Effective particles:{' '}
          <strong className="text-fg">{fix(pf.ess, 0)}</strong> of {pf.particles.length}
          {pf.resampled ? ', resampled' : ''}.
        </p>
      </div>
    </div>
  );
}

export default function FiltersPage() {
  const [params, setParams] = useState<ParamValues>(() => fromScenario(DEFAULT_TRACKING));
  const [seed, setSeed] = useState(DEFAULT_TRACKING.seed);
  const [activeId, setActiveId] = useState<string | null>(null);
  const scenario = useMemo(() => toScenario(params, seed), [params, seed]);
  const run = useMemo(() => runTracking(scenario), [scenario]);
  const setParam = (key: string, value: ParamValue) => setParams((p) => ({ ...p, [key]: value }));

  const apply = (exp: TrackingExperiment) => {
    setParams(fromScenario({ ...DEFAULT_TRACKING, ...exp.scenario }));
    setSeed(DEFAULT_TRACKING.seed);
    setActiveId(exp.id);
  };
  const stillActive = (() => {
    const exp = trackingExperiments.find((e) => e.id === activeId);
    return (
      !!exp &&
      seed === DEFAULT_TRACKING.seed &&
      JSON.stringify(fromScenario({ ...DEFAULT_TRACKING, ...exp.scenario })) ===
        JSON.stringify(params)
    );
  })();

  const metrics: Metric[] = [
    {
      label: 'Raw readings: error',
      value: fix(run.rmse.raw, 2),
      testId: 'rmse-raw',
      hint: 'root-mean-square distance to the true position, over the readings that arrived',
    },
    { label: 'Kalman filter: error', value: fix(run.rmse.kalman, 2), testId: 'rmse-kalman' },
    { label: 'Smoother: error', value: fix(run.rmse.smoother, 2) },
    { label: 'Particle filter: error', value: fix(run.rmse.particle, 2) },
  ];

  const stages: [string, string][] = [
    ['lab-scene', 'The track'],
    ['lab-how', 'How it works'],
    ['lab-controls', 'Settings'],
  ];

  return (
    <div className="mx-auto max-w-6xl px-4 pt-8 pb-20 sm:px-6 lg:px-8">
      <LabHeader
        eyebrow="Classic · probabilistic"
        title="Kalman & particle filters"
        accent="tracking through the noise"
        blurb="A target moves where you cannot see it and a jittery sensor reports its position. Blend what the physics predicts with what the sensor says, and watch the uncertainty shrink, swell and shrink again."
        badges={
          <>
            <Badge>{scenario.motion === 'cruise' ? 'cruising target' : 'turning target'}</Badge>
            <Badge>{scenario.particles} particles</Badge>
          </>
        }
      />
      <StageNav stages={stages} />
      <ExperimentBar
        lessons={trackingExperiments}
        activeId={stillActive ? activeId : null}
        onApply={apply}
      />

      <div className="flex flex-col gap-5">
        <Panel id="lab-scene" className="scroll-mt-24" title="Following the target" eyebrow="01">
          <div className="space-y-6">
            <MetricsStrip metrics={metrics} />
            <Scene key={`${scenario.steps}-${seed}`} run={run} />
          </div>
        </Panel>
        <Panel id="lab-how" className="scroll-mt-24" title="How it works" eyebrow="02">
          <div className="space-y-3 text-sm text-muted">
            <p>
              The <strong className="text-fg">Kalman filter</strong> keeps a Gaussian belief (a mean
              and a covariance) about the state: position and velocity. Each step has two moves:{' '}
              <em>predict</em> (advance the belief with the motion model, adding the uncertainty of
              random pushes) and <em>update</em> (compare the prediction with the reading, and move
              towards the reading by the <em>Kalman gain</em>, the share of trust given to the
              sensor rather than the model). When it is exact: linear motion, linear measurement,
              Gaussian noise.
            </p>
            <p>
              The <strong className="text-fg">smoother</strong> (Rauch-Tung-Striebel) runs backwards
              over the filtered track and revises each estimate with the later readings.
            </p>
            <p>
              The <strong className="text-fg">particle filter</strong> represents the belief by a
              cloud of samples: move every particle with the motion model, weight it by how well it
              explains the reading, and resample when only a few carry the weight (the effective
              sample size drops). It needs no formulas for the posterior, so it copes with
              heavy-tailed noise, but it needs enough particles.
            </p>
            <p>
              Tests check the Kalman gain against the closed-form Riccati solution, the filter
              against the plain running mean in the no-motion case, and the particle filter against
              the Kalman filter on a linear Gaussian problem.
            </p>
          </div>
        </Panel>
        <Panel id="lab-controls" className="scroll-mt-24" title="Settings" eyebrow="03">
          <div className="space-y-6">
            <ParamPanel specs={trackingParams} values={params} onChange={setParam} columns={3} />
            <Button
              size="sm"
              variant="secondary"
              onClick={() => setSeed(Math.floor(Math.random() * 1e6))}
            >
              <Dices className="size-4" /> New track
            </Button>
          </div>
        </Panel>
      </div>
    </div>
  );
}
