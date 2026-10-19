import { StackingClassifier } from '@sim/classic';
import { ClassifierLab, fix, flag, pct, type ClassifierContext } from '@/shared/lab-kit';
import { MEMBERS, ensemblesSpec, ensembleOf, memberAccuracies } from './ensemblesSpec';

function Members({ ctx }: { ctx: ClassifierContext }) {
  const e = ensembleOf(ctx.run.model);
  if (!e) return null;
  const acc = memberAccuracies(ctx.run);
  const active = MEMBERS.filter((m) => flag(ctx.params, m.key));
  const trust = e instanceof StackingClassifier ? e.trust() : null;
  const total = trust ? trust.reduce((a, b) => a + b, 0) || 1 : 1;
  return (
    <div className="space-y-4">
      <div className="overflow-x-auto">
        <table className="w-full min-w-[26rem] text-left font-mono text-[0.8rem] tabular-nums">
          <thead className="text-muted">
            <tr>
              {[
                'model',
                'held-out accuracy',
                ...(trust ? ['trust placed by the meta-model'] : []),
              ].map((h) => (
                <th key={h} className="pr-4 pb-2 font-medium">
                  {h}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {active.map((m, i) => (
              <tr key={m.key} className="border-t border-line">
                <td className="py-1.5 pr-4">{m.label}</td>
                <td className="pr-4">{pct(acc.members[i])}</td>
                {trust && <td>{fix((trust[i] as number) / total, 2)}</td>}
              </tr>
            ))}
            <tr className="border-t-2 border-line-strong text-brand" data-testid="ensemble-row">
              <td className="py-1.5 pr-4">The ensemble</td>
              <td className="pr-4">{pct(acc.ensemble)}</td>
              {trust && <td />}
            </tr>
          </tbody>
        </table>
      </div>
      <p className="text-sm text-muted">
        Ensembles help when the members make <em>different</em> mistakes: a point one model gets
        wrong is often rescued by the others. Combining near-identical models gains nothing;
        combining weak models gains little. Switch members on and off in Settings to see the effect.
      </p>
    </div>
  );
}

export default function EnsemblesPage() {
  return (
    <ClassifierLab
      spec={ensemblesSpec}
      text={{
        eyebrow: 'Classic · supervised',
        title: 'Voting & stacking',
        accent: 'many models, one answer',
        blurb:
          'Combine different classifiers by majority vote, by averaging their probabilities, or by training a referee on their predictions.',
      }}
      extras={[
        {
          id: 'lab-members',
          title: 'Members versus the ensemble',
          render: (ctx) => <Members ctx={ctx} />,
        },
      ]}
      howItWorks={
        <div className="space-y-3 text-sm text-muted">
          <p>
            <strong className="text-fg">Majority vote</strong> counts each model’s predicted class.{' '}
            <strong className="text-fg">Soft voting</strong> averages their probabilities, so
            confident models count for more. <strong className="text-fg">Stacking</strong> trains a
            logistic regression on the first-level models’ probabilities, learned from predictions
            on rows each model did <em>not</em> train on (cross-validation), so the referee is not
            fooled by models that merely memorised.
          </p>
          <p>
            Bagging (random forests) and boosting (AdaBoost, gradient boosting) are the same idea
            with many copies of one kind of model; this lab mixes different families.
          </p>
        </div>
      }
    />
  );
}
