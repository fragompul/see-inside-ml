import {
  PolynomialRegression,
  polynomialFeatures,
  regularizationPath,
  ridgePath,
  type PolynomialPenalty,
} from '@sim/classic';
import { Standardizer, rmse } from '@sim/core';
import { useMemo } from 'react';
import { usePalette } from '@/shared/canvas/colors';
import { MiniChart } from '@/shared/components/MiniChart';
import { RegressorLab, num, str, type RegressorContext } from '@/shared/lab-kit';
import { polynomialSpec } from './polynomialSpec';

const MAX_DEGREE = 15;

function Complexity({ ctx }: { ctx: RegressorContext }) {
  const palette = usePalette();
  const penalty = str(ctx.params, 'penalty') as PolynomialPenalty;
  const alpha = num(ctx.params, 'alpha');
  const degree = num(ctx.params, 'degree');
  const { train, test } = ctx.run.data;
  const curves = useMemo(() => {
    const degrees = Array.from({ length: MAX_DEGREE }, (_, i) => i + 1);
    const errs = degrees.map((d) => {
      const m = new PolynomialRegression({ degree: d, penalty, alpha }).fit(train.x, train.y);
      return {
        train: rmse(train.y, m.predict(train.x)),
        test: test.x.rows ? rmse(test.y, m.predict(test.x)) : NaN,
      };
    });
    return { degrees, errs };
  }, [train, test, penalty, alpha]);
  // test errors explode at high degree without a penalty: cap the axis at a readable level
  const cap = Math.max(1.5, 3 * Math.min(...curves.errs.map((e) => e.train)) + 1);
  const clip = (v: number) => Math.min(v, cap);
  const bestDegree = curves.errs.reduce(
    (best, e, i) => (e.test < (curves.errs[best]?.test ?? Infinity) ? i : best),
    0,
  );
  return (
    <div className="grid gap-8 lg:grid-cols-[minmax(0,1fr)_18rem]">
      <MiniChart
        title="Error as the model gets more flexible"
        x={curves.degrees}
        xLabel="degree"
        yRange={{ lo: 0, hi: cap }}
        series={[
          {
            name: 'training RMSE',
            color: palette.css.brand,
            values: curves.errs.map((e) => clip(e.train)),
          },
          {
            name: 'test RMSE',
            color: palette.css.pos,
            values: curves.errs.map((e) => clip(e.test)),
          },
        ]}
        height={200}
      />
      <div className="space-y-3 text-sm text-muted">
        <p>
          Training error only ever falls as the degree grows; the test error falls, then climbs
          again. That U-shape is the bias–variance trade-off.
        </p>
        <p data-testid="best-degree">
          Lowest test error at degree <strong className="text-fg">{bestDegree + 1}</strong>; you are
          at {degree}. (Values above {cap.toFixed(1)} are clipped.)
        </p>
      </div>
    </div>
  );
}

function Paths({ ctx }: { ctx: RegressorContext }) {
  const palette = usePalette();
  const degree = num(ctx.params, 'degree');
  const kind = str(ctx.params, 'penalty') === 'ridge' ? 'ridge' : 'lasso';
  const { train } = ctx.run.data;
  const result = useMemo(() => {
    let scale = 0;
    for (let i = 0; i < train.x.rows; i++) scale = Math.max(scale, Math.abs(train.x.get(i, 0)));
    const raw = polynomialFeatures(
      Float64Array.from({ length: train.x.rows }, (_, i) => train.x.get(i, 0) / (scale || 1)),
      degree,
    );
    const z = new Standardizer().fit(raw).transform(raw);
    const p =
      kind === 'ridge'
        ? ridgePath(
            z,
            train.y,
            Array.from({ length: 40 }, (_, k) => 10 ** (3 - (k * 7) / 39)),
          )
        : regularizationPath(z, train.y, { count: 40, ratio: 1e-4 });
    return { x: p.alphas.map((a) => Math.log10(a)), coefs: p.coefs };
  }, [train, degree, kind]);
  const series = Array.from({ length: degree }, (_, j) => ({
    name: `x^${j + 1}`,
    color: palette.css.classes[j % palette.css.classes.length] as string,
    values: result.coefs.map((c) => c[j] as number),
  }));
  return (
    <div className="grid gap-8 lg:grid-cols-[minmax(0,1fr)_18rem]">
      <MiniChart
        title={`${kind === 'ridge' ? 'Ridge' : 'Lasso'} coefficient paths`}
        x={result.x}
        xLabel="log10 of the penalty α"
        series={series}
        height={220}
      />
      <p className="text-sm text-muted">
        Each line is one standardised power of x. With a heavy penalty every coefficient is pushed
        towards zero. Ridge shrinks them all smoothly and never reaches zero; the lasso drives them
        to exactly zero one by one, which is variable selection. The path follows the penalty chosen
        in Settings (lasso unless ridge is selected).
      </p>
    </div>
  );
}

export default function PolynomialRegularizationPage() {
  return (
    <RegressorLab
      spec={polynomialSpec}
      text={{
        eyebrow: 'Classic · supervised',
        title: 'Polynomials & regularisation',
        accent: 'flexibility under control',
        blurb:
          'Fit a polynomial of any degree to noisy data, watch it overfit, then rein it in with ridge and lasso penalties and see their coefficient paths.',
      }}
      extras={[
        {
          id: 'lab-complexity',
          title: 'Complexity: the U-shaped test error',
          render: (ctx) => <Complexity ctx={ctx} />,
        },
        { id: 'lab-paths', title: 'Coefficient paths', render: (ctx) => <Paths ctx={ctx} /> },
      ]}
      howItWorks={
        <div className="space-y-3 text-sm text-muted">
          <p>
            Polynomial regression is linear regression on the features x, x², …, x^d. More degrees
            mean more flexibility: the training error always falls, but past a point the curve
            starts to chase noise and the test error rises.
          </p>
          <p>
            A penalty on the coefficients limits that. Ridge adds α·Σβ² (smooth shrinkage); the
            lasso adds α·Σ|β|, whose corner at zero produces exact zeros. Powers are standardised
            first, so the penalty treats each of them alike.
          </p>
        </div>
      }
    />
  );
}
