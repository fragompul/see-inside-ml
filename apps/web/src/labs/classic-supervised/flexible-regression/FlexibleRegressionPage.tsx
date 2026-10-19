import { IsotonicRegression, Mars, PSpline } from '@sim/classic';
import { useMemo } from 'react';
import { usePalette } from '@/shared/canvas/colors';
import { MiniChart } from '@/shared/components/MiniChart';
import { RegressorLab, fix, str, type Metric, type RegressorContext } from '@/shared/lab-kit';
import { flexSpec } from './flexSpec';
import { TRUTH_LABEL, twoInputs, type Truth } from './twoInputs';

function TwoInputs() {
  const palette = usePalette();
  const results = useMemo(() => (['additive', 'interaction'] as Truth[]).map(twoInputs), []);
  const additive = results[0];
  return (
    <div className="space-y-6">
      <div className="overflow-x-auto">
        <table className="w-full min-w-[30rem] text-left font-mono text-[0.8rem] tabular-nums">
          <thead className="text-muted">
            <tr>
              {[
                'the output is',
                'straight line',
                'additive spline model',
                'MARS, no products',
                'MARS, with products',
              ].map((h) => (
                <th key={h} className="pr-4 pb-2 font-medium">
                  {h}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {results.map((r) => (
              <tr key={r.truth} className="border-t border-line" data-testid={`two-${r.truth}`}>
                <td className="py-1.5 pr-4">{TRUTH_LABEL[r.truth]}</td>
                <td className="pr-4">{fix(r.r2.linear, 2)}</td>
                <td className="pr-4">{fix(r.r2.additive, 2)}</td>
                <td className="pr-4">{fix(r.r2.marsAdditive, 2)}</td>
                <td>{fix(r.r2.marsInteracting, 2)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="text-sm text-muted">
        Variation explained on new points (R²) for two inputs and 400 training points. An additive
        model sums one smooth function per input, so it fits the first truth and cannot fit the
        second, where the effect of one input depends on the other. MARS can multiply hinges of
        different inputs when it is allowed to.
      </p>
      {additive && (
        <div className="grid gap-6 lg:grid-cols-2">
          {[0, 1].map((j) => (
            <MiniChart
              key={j}
              title={`Shape found for input x${j === 0 ? '₁' : '₂'} (additive truth)`}
              x={additive.grid}
              xLabel={`x${j === 0 ? '₁' : '₂'}`}
              series={[
                {
                  name: 'true',
                  color: palette.css.muted,
                  values: additive.shapes.truth[j] as number[],
                  dashed: true,
                },
                {
                  name: 'found',
                  color: palette.css.brand,
                  values: additive.shapes.recovered[j] as number[],
                },
              ]}
              height={190}
            />
          ))}
        </div>
      )}
    </div>
  );
}

export default function FlexibleRegressionPage() {
  const extraMetrics = (ctx: RegressorContext): Metric[] => {
    const id = str(ctx.params, 'model');
    if (id === 'isotonic' && ctx.run.model instanceof IsotonicRegression)
      return [{ label: 'Steps in the fit', value: String(ctx.run.model.pieces), testId: 'pieces' }];
    if (id === 'spline' && ctx.run.model instanceof PSpline)
      return [
        { label: 'Effective degrees of freedom', value: fix(ctx.run.model.df, 1), testId: 'df' },
      ];
    if (id === 'mars' && ctx.run.model instanceof Mars)
      return [
        {
          label: 'Terms kept',
          value: `${ctx.run.model.terms.length} of ${ctx.run.model.forwardPath[ctx.run.model.forwardPath.length - 1]?.terms ?? 1}`,
          testId: 'terms',
        },
      ];
    return [];
  };
  return (
    <RegressorLab
      spec={flexSpec}
      text={{
        eyebrow: 'Classic · supervised',
        title: 'Flexible regression: isotonic, splines and MARS',
        accent: 'shapes without a formula',
        blurb:
          'When the relationship is not a straight line and you do not want to pick a polynomial, there are other ways to let the data draw the curve: assume only that it goes one way, penalise how much it bends, or build it from hinges.',
      }}
      datasets={['step', 'sine', 'curve', 'line', 'outliers']}
      extraMetrics={extraMetrics}
      extras={[
        {
          id: 'lab-two',
          title: 'Two inputs: additive models and interactions',
          render: () => <TwoInputs />,
        },
      ]}
      howItWorks={
        <div className="space-y-3 text-sm text-muted">
          <p>
            <strong className="text-fg">Isotonic regression</strong> finds the closest
            non-decreasing function to the data. Wherever neighbouring points disagree with the
            order they are pooled into a block and take their average, repeatedly, until the order
            holds (the pool-adjacent-violators algorithm). A{' '}
            <strong className="text-fg">penalised spline</strong> writes the curve as a sum of
            smooth bumps and penalises the second differences of their weights: a small penalty lets
            it follow the data, a huge one leaves a straight line, and the effective degrees of
            freedom measure what is in between.
          </p>
          <p>
            <strong className="text-fg">MARS</strong> builds the curve from hinge functions max(0, x
            − t). The forward pass adds the pair of hinges that most reduces the error, the backward
            pass prunes terms by cross-validation. A sum of one smooth function per input is a{' '}
            <strong className="text-fg">generalised additive model</strong>; the bottom panel shows
            what it can and cannot do. The tests check the isotonic fit against scikit-learn, the
            spline against exact limits, and the experiments above.
          </p>
        </div>
      }
    />
  );
}
