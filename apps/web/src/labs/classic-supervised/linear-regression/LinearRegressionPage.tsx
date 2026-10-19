import { lineGradientDescent } from '@sim/classic';
import { anscombe } from '@sim/core';
import { useMemo, useState } from 'react';
import { Button, Slider } from '@/design-system';
import { usePalette } from '@/shared/canvas/colors';
import { MiniChart } from '@/shared/components/MiniChart';
import { RegressorLab, StepPlayer, fix, type RegressorContext } from '@/shared/lab-kit';
import { linearModel, linearRegressionSpec } from './linearRegressionSpec';

const STEPS = 80;

function GradientDescent({ ctx }: { ctx: RegressorContext }) {
  const palette = usePalette();
  const [lrExp, setLrExp] = useState(-1.3);
  const [step, setStep] = useState(0);
  const lr = 10 ** lrExp;
  const model = linearModel(ctx.run.model);
  const { train } = ctx.run.data;

  const path = useMemo(() => {
    const x = Array.from({ length: train.x.rows }, (_, i) => train.x.get(i, 0));
    return Array.from(lineGradientDescent(x, train.y, { lr, steps: STEPS }));
  }, [train, lr]);
  const current = path[Math.min(step, path.length - 1)];
  if (!current || !model) return null;
  const best = path[path.length - 1];
  const optimum = model.summary().rss / train.x.rows;

  return (
    <div className="grid gap-8 lg:grid-cols-2">
      <div className="space-y-5">
        <p className="text-sm text-muted">
          Instead of solving the normal equations, start from a flat line and walk downhill on the
          squared error. Each step moves the slope and intercept against the gradient. The result
          converges to the same line the closed form gives.
        </p>
        <Slider
          label="Learning rate"
          min={-3}
          max={-0.5}
          step={0.05}
          value={lrExp}
          onValueChange={(v) => {
            setLrExp(v);
            setStep(0);
          }}
          format={(v) => (10 ** v).toFixed(3)}
        />
        <StepPlayer
          count={path.length}
          step={step}
          onStep={setStep}
          intervalMs={90}
          describe={(s) => `step ${s}`}
        />
        <dl className="grid grid-cols-2 gap-3 font-mono text-[0.8rem] tabular-nums">
          <div>
            <dt className="eyebrow">Gradient descent</dt>
            <dd data-testid="gd-line">
              y = {fix(current.slope)}·x {current.intercept < 0 ? '−' : '+'}{' '}
              {fix(Math.abs(current.intercept))}
            </dd>
          </div>
          <div>
            <dt className="eyebrow">Closed form</dt>
            <dd>
              y = {fix(model.coef[0] ?? 0)}·x {model.intercept < 0 ? '−' : '+'}{' '}
              {fix(Math.abs(model.intercept))}
            </dd>
          </div>
        </dl>
      </div>
      <MiniChart
        title="Mean squared error per step"
        x={path.map((p) => p.step)}
        series={[{ name: 'loss', color: palette.css.brand, values: path.map((p) => p.loss) }]}
        guide={optimum}
        log
        xLabel="step"
        note={`Final loss ${best ? fix(best.loss) : ''} vs the least-squares optimum ${fix(optimum)} (dashed). Too large a rate overshoots; too small crawls.`}
      />
    </div>
  );
}

function Diagnostics({ ctx }: { ctx: RegressorContext }) {
  const model = linearModel(ctx.run.model);
  if (!model) return null;
  const s = model.summary(['x']);
  const n = ctx.run.data.train.x.rows;
  const cooks = model.cooksDistance();
  const worst = Array.from(cooks, (d, i) => ({ d, i }))
    .sort((a, b) => b.d - a.d)
    .slice(0, 3);
  return (
    <div className="space-y-6">
      <div className="overflow-x-auto">
        <table className="w-full min-w-[30rem] text-left font-mono text-[0.8rem] tabular-nums">
          <thead className="text-muted">
            <tr>
              {['term', 'estimate', 'std. error', 't', 'p', '95% interval'].map((h) => (
                <th key={h} className="pr-4 pb-2 font-medium">
                  {h}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {s.coefficients.map((c) => (
              <tr key={c.name} className="border-t border-line">
                <td className="py-1.5 pr-4">{c.name}</td>
                <td className="pr-4">{fix(c.estimate)}</td>
                <td className="pr-4">{fix(c.stdError)}</td>
                <td className="pr-4">{fix(c.t)}</td>
                <td className="pr-4">{c.p < 0.0001 ? '<0.0001' : fix(c.p)}</td>
                <td>
                  [{fix(c.ci[0])}, {fix(c.ci[1])}]
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="text-sm text-muted" data-testid="lr-summary">
        n = {n}, R² = {fix(s.r2)}, adjusted R² = {fix(s.adjustedR2)}, F = {fix(s.fStatistic)}, AIC ={' '}
        {fix(s.aic)}. Points ringed on the plot have Cook’s distance above 4/n: removing one would
        move the line noticeably. The most influential here:{' '}
        {worst.map((w) => `#${w.i} (${fix(w.d)})`).join(', ')}.
      </p>
    </div>
  );
}

function Anscombe({ ctx }: { ctx: RegressorContext }) {
  return (
    <div className="space-y-4">
      <p className="text-sm text-muted">
        Four datasets with the same mean, variance, correlation and regression line, yet nothing
        alike. Load each one and compare the statistics above with the picture.
      </p>
      <div className="flex flex-wrap gap-2">
        {([1, 2, 3, 4] as const).map((k) => (
          <Button key={k} size="sm" variant="secondary" onClick={() => ctx.setTable(anscombe(k))}>
            Anscombe {['I', 'II', 'III', 'IV'][k - 1]}
          </Button>
        ))}
        <Button size="sm" variant="ghost" onClick={() => ctx.setTable(null)}>
          Back to generated data
        </Button>
      </div>
    </div>
  );
}

const influential = (ctx: RegressorContext): ReadonlySet<number> | undefined => {
  const model = linearModel(ctx.run.model);
  if (!model) return undefined;
  const cooks = model.cooksDistance();
  const { testMask } = ctx.run.data;
  const threshold = 4 / cooks.length;
  const marked = new Set<number>();
  let k = 0;
  for (let i = 0; i < testMask.length; i++) {
    if (testMask[i] === 1) continue;
    if ((cooks[k] ?? 0) > threshold) marked.add(i);
    k++;
  }
  return marked;
};

export default function LinearRegressionPage() {
  return (
    <RegressorLab
      spec={linearRegressionSpec}
      text={{
        eyebrow: 'Classic · supervised',
        title: 'Linear regression',
        accent: 'the best straight line',
        blurb:
          'Fit y = a·x + b by least squares, see the squared errors the line minimises, then break it with outliers, curves and Anscombe’s quartet.',
      }}
      highlight={influential}
      extras={[
        {
          id: 'lab-gd',
          title: 'Descending to the line',
          render: (ctx) => <GradientDescent ctx={ctx} />,
        },
        {
          id: 'lab-diagnostics',
          title: 'Diagnostics: is the fit trustworthy?',
          render: (ctx) => <Diagnostics ctx={ctx} />,
        },
        {
          id: 'lab-anscombe',
          title: 'Anscombe’s quartet',
          render: (ctx) => <Anscombe ctx={ctx} />,
        },
      ]}
      howItWorks={
        <div className="space-y-3 text-sm text-muted">
          <p>
            Least squares picks the slope and intercept that minimise the sum of squared vertical
            distances to the line. For one feature that has a closed form: slope = cov(x, y) /
            var(x) and intercept = ȳ − slope·x̄. With several features it solves the normal equations
            (XᵀX)β = Xᵀy.
          </p>
          <p>
            R² is the share of the variance of y the line explains. A high R² does not mean the
            model is right (Anscombe) and a low one does not mean it is useless: always look at the
            residuals and the influential points.
          </p>
        </div>
      }
    />
  );
}
