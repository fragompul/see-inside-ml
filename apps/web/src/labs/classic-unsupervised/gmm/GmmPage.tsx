import { GaussianMixture, bicCurve, type CovarianceType, type MixtureState } from '@sim/classic';
import type { Matrix } from '@sim/core';
import { plotBounds, type Table } from '@sim/core';
import { useMemo, useState } from 'react';
import { usePalette } from '@/shared/canvas/colors';
import { MiniChart } from '@/shared/components/MiniChart';
import {
  ClusterLab,
  DataPlane,
  StepPlayer,
  drawEllipse,
  fix,
  num,
  str,
  type ClusterContext,
  type Painter,
} from '@/shared/lab-kit';
import { gmmOf, gmmSpec } from './gmmSpec';

const paintComponents =
  (weights: ArrayLike<number>, means: Matrix, covs: Matrix[]) => (p: Painter) => {
    for (let c = 0; c < means.rows; c++) {
      const color = p.palette.css.classes[c % p.palette.css.classes.length] as string;
      const mean = [means.get(c, 0), means.get(c, 1)];
      drawEllipse(p, mean, covs[c] as Matrix, 1, { color, width: 2.2, fill: true });
      drawEllipse(p, mean, covs[c] as Matrix, 2, { color, width: 1.2, alpha: 0.7 });
      p.ctx.beginPath();
      p.ctx.arc(
        p.px(mean[0] as number),
        p.py(mean[1] as number),
        3 + 9 * (weights[c] as number),
        0,
        Math.PI * 2,
      );
      p.ctx.fillStyle = color;
      p.ctx.fill();
      p.ctx.lineWidth = 2;
      p.ctx.strokeStyle = p.palette.css.surface;
      p.ctx.stroke();
    }
  };

const unsurePoints = (m: GaussianMixture): Set<number> => {
  const out = new Set<number>();
  for (let i = 0; i < m.responsibilities.rows; i++) {
    let best = 0;
    for (let c = 0; c < m.responsibilities.cols; c++)
      best = Math.max(best, m.responsibilities.get(i, c));
    if (best < 0.9) out.add(i);
  }
  return out;
};

function Em({ ctx }: { ctx: ClusterContext }) {
  const palette = usePalette();
  const k = num(ctx.params, 'k');
  const covariance = str(ctx.params, 'covariance') as CovarianceType;
  const seed = num(ctx.params, 'seed');
  const { data } = ctx.run;
  const [step, setStep] = useState(0);
  const states = useMemo<MixtureState[]>(
    () =>
      Array.from(
        new GaussianMixture({ k, covariance, seed, tolerance: 1e-4, maxIterations: 80 }).states(
          data.x,
        ),
      ),
    [data, k, covariance, seed],
  );
  const s = states[Math.min(step, states.length - 1)] as MixtureState;
  const bounds = useMemo(() => plotBounds(data.x, 0.2), [data]);
  const painted = useMemo<Table>(() => {
    const labels = Float64Array.from({ length: data.x.rows }, (_, i) => {
      let best = 0;
      for (let c = 1; c < s.responsibilities.cols; c++)
        if (s.responsibilities.get(i, c) > s.responsibilities.get(i, best)) best = c;
      return best;
    });
    return { ...data, y: labels };
  }, [data, s]);
  const noTest = useMemo(() => new Uint8Array(data.x.rows), [data]);
  const overlay = useMemo(() => paintComponents(s.weights, s.means, s.covariances), [s]);
  return (
    <div className="grid gap-8 lg:grid-cols-[auto_minmax(0,1fr)]">
      <DataPlane
        all={painted}
        testMask={noTest}
        bounds={bounds}
        width={420}
        height={320}
        overlay={overlay}
        ariaLabel="One run of expectation-maximisation, step by step: the ellipses are the Gaussian components, the points take the colour of their most responsible component."
      />
      <div className="min-w-0 space-y-5">
        <StepPlayer
          label="Iteration"
          count={states.length}
          step={step}
          onStep={setStep}
          intervalMs={500}
          describe={(i) => (i === 0 ? 'start' : `after step ${i}`)}
        />
        <MiniChart
          title="Mean log-likelihood of the data"
          x={states.map((st) => st.iteration)}
          xLabel="iteration"
          series={[
            {
              name: 'log-likelihood',
              color: palette.css.brand,
              values: states.map((st) => st.logLikelihood),
            },
          ]}
          note="Every EM step can only raise it. The run ends when the gain drops below the tolerance."
          height={150}
        />
        <p className="font-mono text-[0.8rem] tabular-nums" data-testid="em-readout">
          iteration {s.iteration}: log-likelihood {fix(s.logLikelihood, 3)}
        </p>
      </div>
    </div>
  );
}

function Selection({ ctx }: { ctx: ClusterContext }) {
  const palette = usePalette();
  const covariance = str(ctx.params, 'covariance') as CovarianceType;
  const k = num(ctx.params, 'k');
  const { data } = ctx.run;
  const bic = useMemo(() => bicCurve(data.x, 7, { covariance, seed: 3 }), [data, covariance]);
  const best = bic.indexOf(Math.min(...bic)) + 1;
  return (
    <div className="grid gap-8 lg:grid-cols-[minmax(0,1fr)_20rem]">
      <MiniChart
        title="BIC against the number of components"
        x={bic.map((_, i) => i + 1)}
        xLabel="k"
        series={[{ name: 'BIC', color: palette.css.pos, values: bic }]}
        height={190}
      />
      <div className="space-y-3 text-sm text-muted">
        <p>
          More components always fit the data better, so the likelihood alone would say “as many as
          possible”. The BIC adds a penalty for every extra parameter; the best k is where it is
          lowest.
        </p>
        <p data-testid="bic-best">
          Lowest BIC at <strong className="text-fg">k = {best}</strong>; you are using k = {k}.
        </p>
      </div>
    </div>
  );
}

export default function GmmPage() {
  return (
    <ClusterLab
      spec={gmmSpec}
      text={{
        eyebrow: 'Classic · unsupervised',
        title: 'Gaussian mixtures & EM',
        accent: 'soft clusters with a shape',
        blurb:
          'Each cluster is a Gaussian bell with its own centre, spread and weight. EM alternates between guessing who belongs where and refitting the bells.',
      }}
      overlay={(ctx) => {
        const m = gmmOf(ctx.run.model);
        return m ? paintComponents(m.weights, m.means, m.covariances) : undefined;
      }}
      highlight={(ctx) => {
        const m = gmmOf(ctx.run.model);
        return m ? unsurePoints(m) : undefined;
      }}
      extraMetrics={(ctx) => {
        const m = gmmOf(ctx.run.model);
        return m
          ? [
              { label: 'Log-likelihood', value: fix(m.score(ctx.run.data.x)), testId: 'loglik' },
              { label: 'BIC', value: fix(m.bic(ctx.run.data.x), 1), testId: 'bic' },
            ]
          : [];
      }}
      extras={[
        {
          id: 'lab-em',
          title: 'Expectation–maximisation, step by step',
          render: (ctx) => <Em ctx={ctx} />,
        },
        {
          id: 'lab-bic',
          title: 'Choosing the number of components',
          render: (ctx) => <Selection ctx={ctx} />,
        },
      ]}
      howItWorks={
        <div className="space-y-3 text-sm text-muted">
          <p>
            A Gaussian mixture says each point was drawn by first picking a component (with
            probability equal to its weight) and then sampling from that component’s Gaussian. EM
            fits it without knowing the picks: the <strong className="text-fg">E-step</strong>{' '}
            computes each point’s responsibility for each component; the{' '}
            <strong className="text-fg">M-step</strong> re-estimates every weight, mean and
            covariance from the points, weighted by those responsibilities. The likelihood rises
            every round.
          </p>
          <p>
            Compared with k-means, a mixture gives soft assignments, clusters of different sizes and
            orientations, and a likelihood to compare models with (BIC). k-means is the special case
            of equal spherical components with hard assignments. Ringed points are those split
            between components.
          </p>
        </div>
      }
    />
  );
}
