import { HuberRegression, LinearRegression } from '@sim/classic';
import { Matrix, mae } from '@sim/core';
import { useMemo } from 'react';
import { usePalette } from '@/shared/canvas/colors';
import { RegressorLab, fix, str, type RegressorContext } from '@/shared/lab-kit';
import { MODELS, makeModel, robustSpec, type RobustModelId } from './robustSpec';

const LINEAR: RobustModelId[] = ['ols', 'huber', 'median', 'theilsen'];

function Comparison({ ctx }: { ctx: RegressorContext }) {
  const { train, test } = ctx.run.data;
  const rows = useMemo(
    () =>
      LINEAR.map((id) => {
        const m = makeModel(id);
        m.fit(train.x, train.y);
        const at0 = m.predict(Matrix.from([[0]]))[0] as number;
        const at1 = m.predict(Matrix.from([[1]]))[0] as number;
        return {
          id,
          intercept: at0,
          slope: at1 - at0,
          trainMae: mae(train.y, m.predict(train.x)),
          testMae: test.x.rows ? mae(test.y, m.predict(test.x)) : null,
        };
      }),
    [train, test],
  );
  return (
    <div className="space-y-4">
      <div className="overflow-x-auto">
        <table className="w-full min-w-[28rem] text-left font-mono text-[0.8rem] tabular-nums">
          <thead className="text-muted">
            <tr>
              {['method', 'intercept', 'slope', 'train MAE', 'test MAE'].map((h) => (
                <th key={h} className="pr-4 pb-2 font-medium">
                  {h}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.id} className="border-t border-line" data-testid={`row-${r.id}`}>
                <td className="py-1.5 pr-4">{MODELS.find((m) => m.value === r.id)?.label}</td>
                <td className="pr-4">{fix(r.intercept)}</td>
                <td className="pr-4">{fix(r.slope)}</td>
                <td className="pr-4">{fix(r.trainMae)}</td>
                <td>{fix(r.testMae)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="text-sm text-muted">
        All four fit the same training rows. On clean data they agree; with outliers least squares
        moves while the robust ones stay near the true line (intercept 0.5, slope 0.9 for the
        Outliers dataset). Mean absolute error is used because squared error would again let
        outliers dominate the comparison.
      </p>
    </div>
  );
}

function Weights({ ctx }: { ctx: RegressorContext }) {
  const model = ctx.run.model instanceof HuberRegression ? ctx.run.model : null;
  if (!model) {
    return (
      <p className="text-sm text-muted">
        Pick the Huber model to see the weight it gives each point (ringed on the plot when below
        0.5).
      </p>
    );
  }
  const low = Array.from(model.weights).filter((w) => w < 0.5).length;
  return (
    <p className="text-sm text-muted" data-testid="huber-summary">
      Huber’s robust scale is <strong className="text-fg">{fix(model.scale)}</strong>; it needed{' '}
      {model.iterations} reweighting rounds and down-weighted{' '}
      <strong className="text-fg">{low}</strong> of {model.weights.length} training points below 0.5
      (ringed on the plot). Each point’s weight is 1 while its residual is within the threshold,
      then t / |residual/σ|, so the farther it is the less it counts.
    </p>
  );
}

export default function RobustGlmPage() {
  const palette = usePalette();
  return (
    <RegressorLab
      spec={robustSpec}
      text={{
        eyebrow: 'Classic · supervised',
        title: 'Robust regression & GLMs',
        accent: 'when least squares is not enough',
        blurb:
          'Outliers, skewed noise and counts break the assumptions behind least squares. Compare Huber, median, Theil–Sen and quantile lines, and a Poisson model for counts.',
      }}
      curves={(ctx) => {
        if (str(ctx.params, 'model') === 'ols' || str(ctx.params, 'model') === 'poisson') return [];
        const { train } = ctx.run.data;
        const ols = new LinearRegression().fit(train.x, train.y);
        const xs = ctx.curve.xs;
        const ys = ols.predict(Matrix.from(Array.from(xs, (x) => [x])));
        return [{ xs, ys, color: palette.css.muted, dashed: true }];
      }}
      highlight={(ctx) => {
        const m = ctx.run.model instanceof HuberRegression ? ctx.run.model : null;
        if (!m) return undefined;
        const marked = new Set<number>();
        let k = 0;
        for (let i = 0; i < ctx.run.data.testMask.length; i++) {
          if (ctx.run.data.testMask[i] === 1) continue;
          if ((m.weights[k] ?? 1) < 0.5) marked.add(i);
          k++;
        }
        return marked;
      }}
      datasets={['outliers', 'line', 'heteroscedastic', 'counts', 'curve']}
      extras={[
        {
          id: 'lab-compare',
          title: 'Four lines, one dataset',
          render: (ctx) => <Comparison ctx={ctx} />,
        },
        {
          id: 'lab-weights',
          title: 'Who counts as an outlier?',
          render: (ctx) => <Weights ctx={ctx} />,
        },
      ]}
      howItWorks={
        <div className="space-y-3 text-sm text-muted">
          <p>
            Least squares minimises Σ(error)². Squaring makes a point ten times off a hundred times
            costlier, so one outlier can pull the whole line. Huber’s loss is quadratic for small
            errors and linear for large ones; the median line minimises Σ|error|; Theil–Sen takes
            the median of all pairwise slopes. Quantile regression minimises a tilted absolute loss
            to fit any percentile, not just the middle.
          </p>
          <p>
            A generalised linear model changes the noise model and the link: for counts, y ~ Poisson
            with mean exp(β₀ + β₁x), so predictions are positive and the variance grows with the
            mean. It is fitted by iteratively reweighted least squares.
          </p>
        </div>
      }
    />
  );
}
