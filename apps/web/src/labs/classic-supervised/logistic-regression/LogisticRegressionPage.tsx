import { binaryLogLoss, logisticGradientDescent } from '@sim/classic';
import { useMemo, useState } from 'react';
import { Slider } from '@/design-system';
import { usePalette } from '@/shared/canvas/colors';
import { MiniChart } from '@/shared/components/MiniChart';
import { ClassifierLab, StepPlayer, fix, num, type ClassifierContext } from '@/shared/lab-kit';
import { logisticOf, logisticSpec } from './logisticSpec';

const STEPS = 200;

function Optimisation({ ctx }: { ctx: ClassifierContext }) {
  const palette = usePalette();
  const [lrExp, setLrExp] = useState(-0.5);
  const [step, setStep] = useState(0);
  const model = logisticOf(ctx.run.model);
  const binary = model?.classes === 2 && num(ctx.params, 'degree') === 1;
  const { train } = ctx.run.data;
  const lr = 10 ** lrExp;
  const l2 = 1 / (num(ctx.params, 'C') * train.x.rows);

  const path = useMemo(
    () =>
      binary ? Array.from(logisticGradientDescent(train.x, train.y, { lr, steps: STEPS, l2 })) : [],
    [binary, train, lr, l2],
  );
  if (!model || !binary || path.length === 0) {
    return (
      <p className="text-sm text-muted">
        This view compares gradient descent with Newton’s method for two classes and raw features:
        pick a two-class dataset (not Blobs) and set the feature degree to 1.
      </p>
    );
  }
  const current = path[Math.min(step, path.length - 1)]!;
  const optimum = binaryLogLoss(
    train.x,
    train.y,
    Array.from(model.weights.data),
    model.intercepts[0] as number,
    l2,
  );
  return (
    <div className="grid gap-8 lg:grid-cols-2">
      <div className="space-y-5">
        <p className="text-sm text-muted">
          The fit above comes from Newton’s method (IRLS): it needed{' '}
          <strong data-testid="newton-steps">{model.iterations}</strong> iterations. Plain gradient
          descent walks the same loss downhill with a fixed step size and needs far more.
        </p>
        <Slider
          label="Learning rate"
          min={-2.5}
          max={0.7}
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
          intervalMs={40}
          describe={(s) => `step ${s}`}
        />
        <p className="font-mono text-[0.8rem] tabular-nums" data-testid="gd-state">
          w = ({current.weights.map((w) => fix(w)).join(', ')}), b = {fix(current.intercept)}, loss
          = {fix(current.loss)}
        </p>
      </div>
      <MiniChart
        title="Log-loss per step"
        x={path.map((p) => p.step)}
        series={[
          { name: 'gradient descent', color: palette.css.brand, values: path.map((p) => p.loss) },
        ]}
        guide={optimum}
        xLabel="step"
        note={`Dashed: the minimum Newton’s method reaches (${fix(optimum)}).`}
      />
    </div>
  );
}

function Coefficients({ ctx }: { ctx: ClassifierContext }) {
  const model = logisticOf(ctx.run.model);
  const degree = num(ctx.params, 'degree');
  if (!model || degree !== 1) {
    return (
      <p className="text-sm text-muted">
        With polynomial features there is one weight per power and product; the table is shown for
        the two raw inputs (degree 1).
      </p>
    );
  }
  const names = ctx.run.data.all.names;
  const classes = ctx.run.data.all.classes ?? [];
  const binary = model.classes === 2;
  return (
    <div className="space-y-4">
      <div className="overflow-x-auto">
        <table className="w-full min-w-[26rem] text-left font-mono text-[0.8rem] tabular-nums">
          <thead className="text-muted">
            <tr>
              <th className="pr-4 pb-2 font-medium">{binary ? '' : 'class'}</th>
              <th className="pr-4 pb-2 font-medium">intercept</th>
              {names.map((n) => (
                <th key={n} className="pr-4 pb-2 font-medium">
                  {n} {binary ? '(odds ratio)' : ''}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {Array.from({ length: model.weights.rows }, (_, c) => (
              <tr key={c} className="border-t border-line">
                <td className="py-1.5 pr-4">{binary ? 'class 1' : (classes[c] ?? `class ${c}`)}</td>
                <td className="pr-4">{fix(model.intercepts[c] as number)}</td>
                {names.map((n, j) => {
                  const w = model.weights.get(c, j);
                  return (
                    <td key={n} className="pr-4">
                      {fix(w)}
                      {binary ? ` (×${fix(Math.exp(w))})` : ''}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="text-sm text-muted">
        {binary
          ? 'Each weight is the change in the log-odds of class 1 per unit of its feature; exp(weight) is the odds ratio: a factor of 2 doubles the odds.'
          : 'Each class has its own weights; the class with the highest score wins and the softmax converts the scores into probabilities.'}
      </p>
    </div>
  );
}

export default function LogisticRegressionPage() {
  return (
    <ClassifierLab
      spec={logisticSpec}
      text={{
        eyebrow: 'Classic · supervised',
        title: 'Logistic regression',
        accent: 'a line that gives probabilities',
        blurb:
          'Squash a linear score through the sigmoid and read it as a probability. See where the 50% boundary falls, how regularisation tames it and how extra features bend it.',
      }}
      extras={[
        {
          id: 'lab-optimisation',
          title: 'Optimisation: Newton vs gradient descent',
          render: (ctx) => <Optimisation ctx={ctx} />,
        },
        {
          id: 'lab-coefficients',
          title: 'Reading the weights',
          render: (ctx) => <Coefficients ctx={ctx} />,
        },
      ]}
      howItWorks={
        <div className="space-y-3 text-sm text-muted">
          <p>
            For two classes the model says P(class 1 | x) = σ(w·x + b) with σ(z) = 1 / (1 + e⁻ᶻ).
            The boundary is where the score is zero, a straight line; the probability fades smoothly
            on either side. The weights minimise the log-loss, a convex function, so there is
            exactly one best answer. Newton’s method (iteratively reweighted least squares) finds it
            in a handful of steps.
          </p>
          <p>
            A penalty ‖w‖²/(2C) keeps the weights small. With three or more classes the sigmoid
            becomes a softmax over one score per class. Because the model is linear in its features,
            adding powers and products of the inputs is all it takes to bend the boundary.
          </p>
        </div>
      }
    />
  );
}
