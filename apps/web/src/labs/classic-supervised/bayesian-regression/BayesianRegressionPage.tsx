import { Matrix, createRng, rmse } from '@sim/core';
import {
  RegressorLab,
  fix,
  num,
  type Band,
  type Curve,
  type RegressorContext,
} from '@/shared/lab-kit';
import { usePalette } from '@/shared/canvas/colors';
import { bayesOf, bayesSpec, plainPolynomial } from './bayesSpec';

const grid = (ctx: RegressorContext) =>
  new Matrix(ctx.curve.xs.length, 1, Float64Array.from(ctx.curve.xs));

function Evidence({ ctx }: { ctx: RegressorContext }) {
  const m = bayesOf(ctx.run.model);
  if (!m) return null;
  const { train, test } = ctx.run.data;
  const plain = plainPolynomial(num(ctx.params, 'degree')).fit(train.x, train.y);
  const err = (t: typeof train, model: { predict: (x: Matrix) => Float64Array }) =>
    t.x.rows ? rmse(t.y, model.predict(t.x)) : null;
  return (
    <div className="space-y-4">
      <p className="text-sm text-muted" data-testid="bayes-evidence">
        From the data alone the model estimates the noise level σ ={' '}
        <strong className="text-fg">{fix(1 / Math.sqrt(m.model.alpha))}</strong> (precision α ={' '}
        {fix(m.model.alpha, 2)}) and the weight precision λ ={' '}
        <strong className="text-fg">{fix(m.model.lambda, 3)}</strong>. Their ratio λ/α ={' '}
        {fix(m.model.lambda / m.model.alpha, 3)} is exactly the ridge penalty it chose for itself,
        found in {m.model.iterations} iterations of evidence maximisation.
      </p>
      <div className="overflow-x-auto">
        <table className="w-full min-w-[22rem] text-left font-mono text-[0.8rem] tabular-nums">
          <thead className="text-muted">
            <tr>
              {['model', 'train RMSE', 'test RMSE'].map((h) => (
                <th key={h} className="pr-4 pb-2 font-medium">
                  {h}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            <tr className="border-t border-line">
              <td className="py-1.5 pr-4">Bayesian (self-tuned shrinkage)</td>
              <td className="pr-4">{fix(err(train, m))}</td>
              <td>{fix(err(test, m))}</td>
            </tr>
            <tr className="border-t border-line">
              <td className="py-1.5 pr-4">Plain least squares, same degree</td>
              <td className="pr-4">{fix(err(train, plain))}</td>
              <td>{fix(err(test, plain))}</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  );
}

export default function BayesianRegressionPage() {
  const palette = usePalette();
  return (
    <RegressorLab
      spec={bayesSpec}
      text={{
        eyebrow: 'Classic · supervised',
        title: 'Bayesian regression',
        accent: 'a posterior over curves',
        blurb:
          'Rather than one best polynomial, keep every polynomial the data allow, weighted by plausibility. See uncertainty shrink as data arrives.',
      }}
      curves={(ctx): Curve[] => {
        const m = bayesOf(ctx.run.model);
        if (!m) return [];
        return m.sampleCurves(grid(ctx), 6, createRng(3)).map((ys) => ({
          xs: ctx.curve.xs,
          ys,
          color: palette.css.muted,
          dashed: true,
          width: 1,
        }));
      }}
      bands={(ctx): Band[] => {
        const m = bayesOf(ctx.run.model);
        if (!m) return [];
        const mean = m.predict(grid(ctx));
        const std = m.predictStd(grid(ctx));
        return [
          {
            xs: ctx.curve.xs,
            lo: Float64Array.from(mean, (v, i) => v - 2 * (std[i] as number)),
            hi: Float64Array.from(mean, (v, i) => v + 2 * (std[i] as number)),
            color: palette.css.brand,
            alpha: 0.16,
          },
        ];
      }}
      datasets={['sine', 'curve', 'line', 'heteroscedastic']}
      extras={[
        {
          id: 'lab-evidence',
          title: 'What the model learned about itself',
          render: (ctx) => <Evidence ctx={ctx} />,
        },
      ]}
      howItWorks={
        <div className="space-y-3 text-sm text-muted">
          <p>
            Put a Gaussian prior on the weights (precision λ) and assume Gaussian noise (precision
            α). By Bayes’ rule the posterior over the weights is again Gaussian, with mean (αΦᵀΦ +
            λI)⁻¹αΦᵀy and covariance (αΦᵀΦ + λI)⁻¹. A prediction is the average over all those
            weight vectors: its mean is the usual one, its variance adds the noise to the
            uncertainty about the weights.
          </p>
          <p>
            α and λ are not hand-tuned: the model maximises the marginal likelihood of the data,
            which automatically balances fit and complexity. The dashed curves are random
            polynomials drawn from the posterior; the band is ±2 predictive standard deviations.
          </p>
        </div>
      }
    />
  );
}
