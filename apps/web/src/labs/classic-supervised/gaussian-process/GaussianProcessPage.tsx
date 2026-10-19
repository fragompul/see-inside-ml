import { kernelValue, type GpKernel } from '@sim/classic';
import { Matrix, cholesky, createRng } from '@sim/core';
import { useMemo } from 'react';
import { usePalette } from '@/shared/canvas/colors';
import { MiniChart } from '@/shared/components/MiniChart';
import {
  RegressorLab,
  fix,
  str,
  type Band,
  type Curve,
  type RegressorContext,
} from '@/shared/lab-kit';
import { gpOf, gpSpec } from './gpSpec';

const posterior = (ctx: RegressorContext) => {
  const m = gpOf(ctx.run.model);
  if (!m) return null;
  const x = new Matrix(ctx.curve.xs.length, 1, Float64Array.from(ctx.curve.xs));
  return { m, x, post: m.gp.posterior(x) };
};

function Hyper({ ctx }: { ctx: RegressorContext }) {
  const m = gpOf(ctx.run.model);
  if (!m) return null;
  const h = m.gp.hyper;
  return (
    <div className="space-y-3 text-sm text-muted" data-testid="gp-hyper">
      <p>
        Length scale <strong className="text-fg">{fix(h.lengthScale)}</strong>, signal variance{' '}
        <strong className="text-fg">{fix(h.variance)}</strong>, noise variance{' '}
        <strong className="text-fg">{fix(h.noise)}</strong>. Log marginal likelihood{' '}
        <strong className="text-fg">{fix(m.likelihood, 2)}</strong>.
      </p>
      <p>
        The marginal likelihood is the probability the model assigns to the data after averaging
        over every function it could draw. It rewards fitting the points and penalises needless
        complexity, so it can pick hyperparameters without a validation set: a too-short length
        scale fits perfectly but gets a low score, a too-long one cannot fit at all.
      </p>
    </div>
  );
}

function Prior({ ctx }: { ctx: RegressorContext }) {
  const palette = usePalette();
  const m = gpOf(ctx.run.model);
  const h = m?.gp.hyper;
  const kernel = str(ctx.params, 'kernel') as GpKernel;
  const series = useMemo(() => {
    if (!h) return null;
    const xs = Array.from({ length: 80 }, (_, i) => -3 + (6 * i) / 79);
    const k = new Matrix(80, 80);
    for (let a = 0; a < 80; a++)
      for (let b = 0; b < 80; b++)
        k.set(
          a,
          b,
          kernelValue([xs[a] as number], [xs[b] as number], { ...h, kernel }) +
            (a === b ? 1e-6 : 0),
        );
    const l = cholesky(k);
    const rng = createRng(11);
    return {
      xs,
      draws: Array.from({ length: 4 }, () => {
        const z = Float64Array.from({ length: 80 }, () => rng.normal());
        return Float64Array.from({ length: 80 }, (_, i) => {
          let s = 0;
          for (let j = 0; j <= i; j++) s += l.get(i, j) * (z[j] as number);
          return s;
        });
      }),
    };
  }, [h, kernel]);
  if (!series) return null;
  return (
    <div className="grid gap-8 lg:grid-cols-[minmax(0,1fr)_18rem]">
      <MiniChart
        title="Functions the kernel considers plausible (before seeing data)"
        x={series.xs}
        xLabel="x"
        series={series.draws.map((d, i) => ({
          name: `draw ${i + 1}`,
          color: palette.css.classes[i % palette.css.classes.length] as string,
          values: d,
        }))}
        height={200}
      />
      <p className="text-sm text-muted">
        A Gaussian process is a distribution over functions, and the kernel says which are likely.
        RBF draws are infinitely smooth; Matérn 3/2 draws are rougher; the periodic kernel repeats.
        Conditioning on the data keeps only the draws that pass through the points.
      </p>
    </div>
  );
}

export default function GaussianProcessPage() {
  const palette = usePalette();
  return (
    <RegressorLab
      spec={gpSpec}
      text={{
        eyebrow: 'Classic · supervised',
        title: 'Gaussian processes',
        accent: 'regression that knows what it does not know',
        blurb:
          'Instead of one curve, keep a whole distribution of curves consistent with the data. The band is the model’s uncertainty: tight near points, wide far away.',
      }}
      curves={(ctx): Curve[] => {
        const p = posterior(ctx);
        if (!p) return [];
        return p.m.gp.sample(p.x, 3, createRng(5)).map((ys) => ({
          xs: ctx.curve.xs,
          ys,
          color: palette.css.muted,
          dashed: true,
          width: 1,
        }));
      }}
      bands={(ctx): Band[] => {
        const p = posterior(ctx);
        if (!p) return [];
        const noise = Math.sqrt(p.m.gp.hyper.noise);
        const lo = Float64Array.from(
          p.post.mean,
          (v, i) => v - 2 * Math.hypot(p.post.std[i] as number, noise),
        );
        const hi = Float64Array.from(
          p.post.mean,
          (v, i) => v + 2 * Math.hypot(p.post.std[i] as number, noise),
        );
        return [{ xs: ctx.curve.xs, lo, hi, color: palette.css.brand, alpha: 0.16 }];
      }}
      datasets={['sine', 'curve', 'line', 'step', 'heteroscedastic']}
      extras={[
        {
          id: 'lab-evidence',
          title: 'Hyperparameters and evidence',
          render: (ctx) => <Hyper ctx={ctx} />,
        },
        {
          id: 'lab-prior',
          title: 'What the kernel believes',
          render: (ctx) => <Prior ctx={ctx} />,
        },
      ]}
      howItWorks={
        <div className="space-y-3 text-sm text-muted">
          <p>
            A GP puts a Gaussian prior on functions with covariance k(x, x′). Given noisy
            observations, the posterior at a new point is also Gaussian: its mean is k*ᵀ(K + σ²I)⁻¹y
            and its variance k(x, x) − k*ᵀ(K + σ²I)⁻¹k*. The shaded band is ±2 standard deviations
            (including observation noise); dashed lines are three random functions drawn from the
            posterior.
          </p>
          <p>
            Cost grows as n³ because of the matrix inverse, so GPs shine on small datasets where
            calibrated uncertainty matters (and they are the engine behind Bayesian optimisation).
            The kernel mean alone is kernel ridge regression.
          </p>
        </div>
      }
    />
  );
}
