import { SYNTHETIC_TRUTH, TabularLab, fix, type TabularContext } from '@/shared/lab-kit';
import { asLinear, multipleRegressionSpec } from './multipleRegressionSpec';

function Diagnostics({ ctx }: { ctx: TabularContext }) {
  const model = asLinear(ctx.run.model);
  if (!model) return null;
  const vif = model.vif();
  const summary = (model.options.l2 ?? 0) === 0 ? model.summary(ctx.names) : null;
  return (
    <div className="space-y-6">
      <div className="overflow-x-auto">
        <table className="w-full min-w-[34rem] text-left font-mono text-[0.8rem] tabular-nums">
          <thead className="text-muted">
            <tr>
              {['term', 'estimate', 'std. error', 't', 'p', 'VIF'].map((h) => (
                <th key={h} className="pr-4 pb-2 font-medium">
                  {h}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {summary
              ? summary.coefficients.map((c, i) => (
                  <tr key={c.name} className="border-t border-line">
                    <td className="py-1.5 pr-4">{c.name}</td>
                    <td className="pr-4">{fix(c.estimate)}</td>
                    <td className="pr-4">{fix(c.stdError)}</td>
                    <td className="pr-4">{fix(c.t)}</td>
                    <td className="pr-4">{c.p < 0.0001 ? '<0.0001' : fix(c.p)}</td>
                    <td className={(vif[i - 1] ?? 0) > 10 ? 'text-neg' : ''}>
                      {i === 0 ? '' : fix(vif[i - 1] ?? 0)}
                    </td>
                  </tr>
                ))
              : ctx.names.map((n, i) => (
                  <tr key={n} className="border-t border-line">
                    <td className="py-1.5 pr-4">{n}</td>
                    <td className="pr-4">{fix(model.coef[i] ?? 0)}</td>
                    <td colSpan={3} />
                    <td className={(vif[i] ?? 0) > 10 ? 'text-neg' : ''}>{fix(vif[i] ?? 0)}</td>
                  </tr>
                ))}
          </tbody>
        </table>
      </div>
      <p className="text-sm text-muted" data-testid="mr-summary">
        {summary
          ? `R² = ${fix(summary.r2)}, adjusted R² = ${fix(summary.adjustedR2)}, F = ${fix(summary.fStatistic)}. `
          : 'Standard errors are not shown for a penalised fit. '}
        The VIF of a feature says how much its coefficient’s variance is inflated by correlation
        with the others: above 5 to 10 is a warning.
      </p>
    </div>
  );
}

export default function MultipleRegressionPage() {
  return (
    <TabularLab
      spec={multipleRegressionSpec}
      text={{
        eyebrow: 'Classic · supervised',
        title: 'Multiple regression',
        accent: 'many features, one plane',
        blurb:
          'Fit y from several features at once. See which coefficients are trustworthy, what correlated features do to them, and how a ridge penalty steadies them.',
      }}
      coefficients={(ctx) => {
        const m = asLinear(ctx.run.model);
        if (!m) return [];
        const synthetic = ctx.setup.source === 'synthetic';
        const s = (m.options.l2 ?? 0) === 0 ? m.summary(ctx.names) : null;
        return ctx.names.map((name, j) => {
          const row = s?.coefficients[j + 1];
          return {
            name,
            value: m.coef[j] ?? 0,
            ...(row ? { lo: row.ci[0], hi: row.ci[1] } : {}),
            ...(synthetic && j < 3 ? { truth: SYNTHETIC_TRUTH.coef[j] as number } : {}),
          };
        });
      }}
      extras={[
        {
          id: 'lab-diagnostics',
          title: 'Diagnostics: which coefficients can be trusted?',
          render: (ctx) => <Diagnostics ctx={ctx} />,
        },
      ]}
      howItWorks={
        <div className="space-y-3 text-sm text-muted">
          <p>
            With features x₁ … x_d the model is y ≈ β₀ + β₁x₁ + … + β_d x_d. Least squares chooses β
            to minimise the squared errors, which has the closed form β = (XᵀX)⁻¹Xᵀy. Each
            coefficient is the effect of its feature <em>holding the others fixed</em>.
          </p>
          <p>
            When two features carry the same information XᵀX is nearly singular: many different β
            fit equally well, so each coefficient is uncertain (large standard error, large VIF)
            while the predictions stay stable. A ridge penalty λ‖β‖² adds λ to the diagonal and
            shrinks the coefficients, trading a little bias for a lot of stability.
          </p>
        </div>
      }
    />
  );
}
