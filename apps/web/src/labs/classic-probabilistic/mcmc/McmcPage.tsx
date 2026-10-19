import { autocorrelation, chainColumn } from '@sim/classic';
import { Dices } from 'lucide-react';
import { useMemo, useState } from 'react';
import { Badge, Button, Panel, Slider } from '@/design-system';
import { usePalette } from '@/shared/canvas/colors';
import { ExperimentBar } from '@/shared/components/ExperimentBar';
import { MiniChart } from '@/shared/components/MiniChart';
import {
  LabHeader,
  MetricsStrip,
  ParamPanel,
  StageNav,
  contourSegments,
  fix,
  type Metric,
  type ParamValue,
  type ParamValues,
} from '@/shared/lab-kit';
import { DEFAULT_MCMC, runMcmc, type McmcRun, type McmcScenario } from './mcmcCore';
import { mcmcExperiments, mcmcParams, type McmcExperiment } from './mcmcSpec';

const toScenario = (p: ParamValues, seed: number): McmcScenario => ({
  target: p.target as McmcScenario['target'],
  sampler: p.sampler as McmcScenario['sampler'],
  steps: Number(p.steps),
  burnIn: Number(p.burnIn),
  proposalSd: Number(p.proposalSd),
  stepSize: Number(p.stepSize),
  leapfrog: Number(p.leapfrog),
  seed,
});

const fromScenario = (s: McmcScenario): ParamValues => ({
  target: s.target,
  sampler: s.sampler,
  steps: s.steps,
  burnIn: s.burnIn,
  proposalSd: s.proposalSd,
  stepSize: s.stepSize,
  leapfrog: s.leapfrog,
});

const N = 64;

function Walk({ run }: { run: McmcRun }) {
  const palette = usePalette();
  const steps = run.scenario.steps;
  const [upTo, setUpTo] = useState(Math.min(steps, 400));
  const k = Math.min(upTo, steps);
  const w = 640;
  const h = 460;
  const { x0, x1, y0, y1 } = run.target.bounds;
  const sx = (v: number) => ((v - x0) / (x1 - x0)) * w;
  const sy = (v: number) => h - ((v - y0) / (y1 - y0)) * h;
  const contours = useMemo(() => {
    const vals = new Float64Array(N * N);
    let max = -Infinity;
    for (let r = 0; r < N; r++)
      for (let c = 0; c < N; c++) {
        const v = run.target.logDensity([
          x0 + ((c + 0.5) / N) * (x1 - x0),
          y1 - ((r + 0.5) / N) * (y1 - y0),
        ]);
        vals[r * N + c] = v;
        max = Math.max(max, v);
      }
    return [0.6, 0.1, 0.01].map((rel) => contourSegments(vals, N, max + Math.log(rel)));
  }, [run.target, x0, x1, y0, y1]);
  const colors = palette.css.classes;
  const first = run.chains[0]!;
  const from = Math.max(0, k - 120);
  const path = first.samples
    .slice(from, k)
    .map(
      (p, i) => `${i ? 'L' : 'M'}${sx(p[0] as number).toFixed(1)},${sy(p[1] as number).toFixed(1)}`,
    )
    .join(' ');
  const stride = Math.max(1, Math.floor((k * run.chains.length) / 1500));
  return (
    <div className="grid gap-8 lg:grid-cols-[minmax(0,1fr)_17rem]">
      <div className="space-y-4">
        <svg
          viewBox={`0 0 ${w} ${h}`}
          className="w-full rounded-xl border border-line"
          role="img"
          aria-label="Samples from four chains drawn over the contours of the target distribution"
        >
          {contours.map((segs, li) =>
            segs.map(([ax, ay, bx, by], i) => (
              <line
                key={`${li}-${i}`}
                x1={(ax / N) * w}
                y1={(ay / N) * h}
                x2={(bx / N) * w}
                y2={(by / N) * h}
                stroke={palette.css.fg}
                strokeWidth={1}
                opacity={0.55 - li * 0.15}
              />
            )),
          )}
          {run.chains.map((c, ci) =>
            c.samples
              .slice(0, k)
              .map((p, i) =>
                i % stride === 0 ? (
                  <circle
                    key={`${ci}-${i}`}
                    cx={sx(p[0] as number)}
                    cy={sy(p[1] as number)}
                    r={2}
                    fill={colors[ci % colors.length]}
                    opacity={i < run.scenario.burnIn ? 0.15 : 0.5}
                  />
                ) : null,
              ),
          )}
          <path d={path} fill="none" stroke={palette.css.fg} strokeWidth={1.2} opacity={0.8} />
          {first.proposals.slice(Math.max(0, k - 40), k).map((p, i) => {
            const idx = Math.max(0, k - 40) + i;
            return first.accepted[idx] ? null : (
              <g key={i} stroke={palette.css.neg} strokeWidth={1.4} opacity={0.8}>
                <line
                  x1={sx(p[0] as number) - 3}
                  y1={sy(p[1] as number) - 3}
                  x2={sx(p[0] as number) + 3}
                  y2={sy(p[1] as number) + 3}
                />
                <line
                  x1={sx(p[0] as number) - 3}
                  y1={sy(p[1] as number) + 3}
                  x2={sx(p[0] as number) + 3}
                  y2={sy(p[1] as number) - 3}
                />
              </g>
            );
          })}
          {run.target.starts.map((p, i) => (
            <circle
              key={i}
              cx={sx(p[0])}
              cy={sy(p[1])}
              r={5}
              fill="none"
              stroke={colors[i % colors.length]}
              strokeWidth={2}
            />
          ))}
        </svg>
        <Slider
          label="Replay the first steps"
          min={1}
          max={steps}
          value={k}
          onValueChange={setUpTo}
        />
      </div>
      <div className="space-y-3 text-sm text-muted" data-testid="walk-note">
        <p>
          Contours: the distribution being explored. Four chains (four colours) start from the
          ringed points. Faint dots are the burn-in you will discard; the black trail is the last
          120 steps of the first chain, and red crosses mark proposals it rejected.
        </p>
        <p>{run.target.note}</p>
      </div>
    </div>
  );
}

function Diagnostics({ run }: { run: McmcRun }) {
  const palette = usePalette();
  const colors = palette.css.classes;
  const shown = Math.min(run.scenario.steps, 400);
  const trace = run.chains.map((c) => chainColumn(c, 0).slice(0, shown));
  const rho = useMemo(() => {
    const kept = chainColumn(
      { ...run.chains[0]!, samples: run.chains[0]!.samples.slice(run.scenario.burnIn) },
      0,
    );
    return autocorrelation(kept, 60);
  }, [run]);
  return (
    <div className="grid gap-8 lg:grid-cols-2">
      <MiniChart
        title="Trace of x for the four chains (first 400 steps)"
        x={trace[0]!.map((_, i) => i)}
        series={trace.map((t, i) => ({
          name: `chain ${i + 1}`,
          color: colors[i % colors.length]!,
          values: t,
        }))}
        xLabel="step"
        height={190}
      />
      <MiniChart
        title="Autocorrelation of x (after burn-in)"
        x={rho.map((_, i) => i)}
        series={[{ name: 'autocorrelation', color: palette.css.brand, values: rho }]}
        xLabel="lag"
        yRange={{ lo: -0.2, hi: 1 }}
        guide={0}
        height={190}
      />
      <p className="text-sm text-muted lg:col-span-2" data-testid="diag-note">
        A healthy trace looks like fuzzy noise with all four chains overlapping. Chains that stay
        apart, or long slow waves, mean poor mixing. The autocorrelation says how long a chain
        remembers: if it takes{' '}
        {rho.findIndex((v) => v < 0.2) < 0 ? 'more than 60' : rho.findIndex((v) => v < 0.2)} steps
        to fall below 0.2, then neighbouring samples carry almost the same information, and the
        effective sample size is small.
      </p>
    </div>
  );
}

export default function McmcPage() {
  const [params, setParams] = useState<ParamValues>(() => fromScenario(DEFAULT_MCMC));
  const [seed, setSeed] = useState(DEFAULT_MCMC.seed);
  const [activeId, setActiveId] = useState<string | null>(null);
  const scenario = useMemo(() => toScenario(params, seed), [params, seed]);
  const outcome = useMemo(() => {
    try {
      return { run: runMcmc(scenario), error: null as string | null };
    } catch (e) {
      return { run: null, error: e instanceof Error ? e.message : String(e) };
    }
  }, [scenario]);
  const run = outcome.run;
  const setParam = (key: string, value: ParamValue) => setParams((p) => ({ ...p, [key]: value }));

  const apply = (exp: McmcExperiment) => {
    setParams(fromScenario({ ...DEFAULT_MCMC, ...exp.scenario }));
    setSeed(DEFAULT_MCMC.seed);
    setActiveId(exp.id);
  };
  const stillActive = (() => {
    const exp = mcmcExperiments.find((e) => e.id === activeId);
    return (
      !!exp &&
      seed === DEFAULT_MCMC.seed &&
      JSON.stringify(fromScenario({ ...DEFAULT_MCMC, ...exp.scenario })) === JSON.stringify(params)
    );
  })();

  const metrics: Metric[] = run
    ? [
        {
          label: 'Accepted moves',
          value: `${Math.round(run.acceptance * 100)}%`,
          testId: 'acceptance',
          hint: 'share of proposals accepted',
        },
        {
          label: 'Effective samples',
          value: `${fix(run.ess, 0)} of ${run.kept}`,
          testId: 'ess',
          hint: 'how many independent draws the correlated chains are worth',
        },
        {
          label: 'R-hat',
          value: fix(run.rHat, 2),
          testId: 'rhat',
          hint: 'agreement between the four chains: near 1 is good, above 1.1 is not converged',
          tone: run.rHat < 1.05 ? 'good' : run.rHat > 1.1 ? 'bad' : 'neutral',
        },
        {
          label: 'Error of the mean',
          value: fix(run.meanError, 2),
          hint: 'distance between the estimated and the true mean',
        },
      ]
    : [];

  const stages: [string, string][] = [
    ['lab-walk', 'The walk'],
    ['lab-diag', 'Diagnostics'],
    ['lab-how', 'How it works'],
    ['lab-controls', 'Settings'],
  ];

  return (
    <div className="mx-auto max-w-6xl px-4 pt-8 pb-20 sm:px-6 lg:px-8">
      <LabHeader
        eyebrow="Classic · probabilistic"
        title="MCMC sampling"
        accent="exploring a distribution by walking"
        blurb="Some distributions can be evaluated but not sampled directly. Markov chain Monte Carlo explores them with a walker. Tune the jump, give it momentum, and learn to tell when the walk can be trusted."
        badges={
          <>
            <Badge>{run?.target.label ?? 'distribution'}</Badge>
            <Badge>
              {scenario.sampler === 'metropolis'
                ? 'Metropolis'
                : scenario.sampler === 'gibbs'
                  ? 'Gibbs'
                  : 'Hamiltonian'}
            </Badge>
          </>
        }
      />
      <StageNav stages={stages} />
      <ExperimentBar
        lessons={mcmcExperiments}
        activeId={stillActive ? activeId : null}
        onApply={apply}
      />

      <div className="flex flex-col gap-5">
        {outcome.error && (
          <p
            role="alert"
            className="rounded-xl border border-danger/40 bg-danger/10 px-4 py-2 text-sm"
          >
            {outcome.error}
          </p>
        )}
        {run && (
          <>
            <Panel id="lab-walk" className="scroll-mt-24" title="The walk" eyebrow="01">
              <div className="space-y-6">
                <MetricsStrip metrics={metrics} />
                <Walk
                  key={`${scenario.target}-${scenario.sampler}-${scenario.steps}-${seed}`}
                  run={run}
                />
              </div>
            </Panel>
            <Panel id="lab-diag" className="scroll-mt-24" title="Diagnostics" eyebrow="02">
              <Diagnostics run={run} />
            </Panel>
          </>
        )}
        <Panel id="lab-how" className="scroll-mt-24" title="How it works" eyebrow="03">
          <div className="space-y-3 text-sm text-muted">
            <p>
              <strong className="text-fg">Metropolis</strong>: from the current point propose a
              random jump; accept it with probability min(1, p(new)/p(old)), otherwise stay. Only
              ratios of the density are used, so the normalising constant (usually the hard part of
              Bayesian inference) never has to be computed.
            </p>
            <p>
              <strong className="text-fg">Gibbs</strong>: update one coordinate at a time, drawing
              it from its exact conditional distribution. No rejections, but moves are axis aligned.
            </p>
            <p>
              <strong className="text-fg">Hamiltonian Monte Carlo</strong>: add a random momentum
              and simulate a ball rolling on the surface of −log p with the leapfrog integrator,
              then accept or reject with a Metropolis step that corrects the simulation error. The
              gradient steers the ball along ridges.
            </p>
            <p>
              <strong className="text-fg">Diagnostics</strong>: acceptance rate (about 20–40% is
              healthy for a random walk), the autocorrelation and the effective sample size, and
              R-hat, which compares the variation between chains with the variation inside them.
              Tests check the samplers against exact distributions with Kolmogorov-Smirnov tests and
              the ESS and R-hat against AR(1) theory.
            </p>
          </div>
        </Panel>
        <Panel id="lab-controls" className="scroll-mt-24" title="Settings" eyebrow="04">
          <div className="space-y-6">
            <ParamPanel specs={mcmcParams} values={params} onChange={setParam} columns={3} />
            <Button size="sm" variant="secondary" onClick={() => setSeed((s) => s + 1)}>
              <Dices className="size-4" /> New chains
            </Button>
          </div>
        </Panel>
      </div>
    </div>
  );
}
