import { Matrix, rmse } from '@sim/core';
import { useMemo } from 'react';
import { usePalette } from '@/shared/canvas/colors';
import { MiniChart } from '@/shared/components/MiniChart';
import {
  RegressorLab,
  StepPlayer,
  fix,
  num,
  type Curve,
  type RegressorContext,
} from '@/shared/lab-kit';
import { boostingSpec, type StagedRegressor } from './boostingSpec';

const staged = (ctx: RegressorContext) => ctx.run.model as StagedRegressor;

function Stages({ ctx }: { ctx: RegressorContext }) {
  const palette = usePalette();
  const rounds = num(ctx.params, 'rounds');
  const { train, test } = ctx.run.data;
  const curves = useMemo(() => {
    const model = staged(ctx);
    const stride = Math.max(1, Math.floor(rounds / 80));
    const ks: number[] = [];
    for (let k = 0; k <= rounds; k += stride) ks.push(k);
    if (ks[ks.length - 1] !== rounds) ks.push(rounds);
    const err = (t: typeof train, k: number) => rmse(t.y, model.predict(t.x, k));
    return {
      ks,
      train: ks.map((k) => err(train, k)),
      test: test.x.rows ? ks.map((k) => err(test, k)) : null,
    };
  }, [ctx, rounds, train, test]);
  const best = curves.test ? curves.ks[curves.test.indexOf(Math.min(...curves.test))] : undefined;
  return (
    <div className="space-y-6">
      <StepPlayer
        label="Round"
        count={300}
        step={Math.max(0, rounds - 1)}
        onStep={(s) => ctx.setParam('rounds', s + 1)}
        intervalMs={90}
        describe={(s) => `${s + 1} trees`}
      />
      <div className="grid gap-8 lg:grid-cols-[minmax(0,1fr)_18rem]">
        <MiniChart
          title="Error after each round"
          x={curves.ks}
          xLabel="trees"
          series={[
            { name: 'training RMSE', color: palette.css.brand, values: curves.train },
            ...(curves.test
              ? [{ name: 'test RMSE', color: palette.css.pos, values: curves.test }]
              : []),
          ]}
          height={200}
        />
        <div className="space-y-3 text-sm text-muted">
          <p>
            Each tree is fitted to what the ensemble still gets wrong and added with a small step.
            The training error keeps falling; the test error falls and then flattens or rises: the
            sweet spot is where to stop (early stopping).
          </p>
          <p data-testid="best-round">
            {best !== undefined ? (
              <>
                Lowest test error near round <strong className="text-fg">{best}</strong> (
                {fix(Math.min(...(curves.test ?? [NaN])))}).
              </>
            ) : null}
          </p>
        </div>
      </div>
    </div>
  );
}

export default function GradientBoostingPage() {
  const palette = usePalette();
  return (
    <RegressorLab
      spec={boostingSpec}
      text={{
        eyebrow: 'Classic · supervised',
        title: 'Gradient boosting & XGBoost',
        accent: 'trees that fix residuals',
        blurb:
          'Build a model as a sum of small trees, each fitted to the errors of the sum so far. Compare the classic version with an XGBoost-style one that adds curvature and penalties.',
      }}
      curves={(ctx): Curve[] => {
        const rounds = num(ctx.params, 'rounds');
        const model = staged(ctx);
        const xs = ctx.curve.xs;
        const m = new Matrix(xs.length, 1, Float64Array.from(xs));
        return [1, 5, 20]
          .filter((k) => k < rounds)
          .map((k) => ({
            xs,
            ys: model.predict(m, k),
            color: palette.css.muted,
            dashed: true,
            width: 1.2,
          }));
      }}
      datasets={['sine', 'curve', 'step', 'heteroscedastic', 'outliers']}
      extras={[
        { id: 'lab-stages', title: 'Growing the ensemble', render: (ctx) => <Stages ctx={ctx} /> },
      ]}
      howItWorks={
        <div className="space-y-3 text-sm text-muted">
          <p>
            Start from a constant (the mean). Repeat: compute the residuals (the negative gradient
            of the squared loss), fit a small tree to them, and add it to the model scaled by the
            learning rate. Each step is gradient descent, but in the space of functions.
          </p>
          <p>
            The XGBoost variant uses a second-order Taylor expansion of the loss: a split’s gain is
            G_L²/(H_L+λ) + G_R²/(H_R+λ) − G²/(H+λ), a leaf is −G/(H+λ), λ shrinks leaves, and a
            split must beat γ to survive. Dashed grey curves show the model after 1, 5 and 20 trees.
          </p>
        </div>
      }
    />
  );
}
