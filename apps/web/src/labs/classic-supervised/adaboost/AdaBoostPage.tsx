import { usePalette } from '@/shared/canvas/colors';
import { MiniChart } from '@/shared/components/MiniChart';
import {
  ClassifierLab,
  StepPlayer,
  fix,
  num,
  pct,
  type ClassifierContext,
  type Painter,
} from '@/shared/lab-kit';
import { adaOf, adaboostSpec } from './adaboostSpec';

/** Rings sized by how much the *last* round cared about each training point. */
const weightRings = (ctx: ClassifierContext) => {
  const model = adaOf(ctx.run.model);
  const last = model?.rounds[model.rounds.length - 1];
  if (!model || !last) return undefined;
  const { testMask, all } = ctx.run.data;
  const n = last.weights.length;
  return (p: Painter) => {
    let k = 0;
    for (let i = 0; i < testMask.length; i++) {
      if (testMask[i] === 1) continue;
      const w = (last.weights[k++] as number) * n; // 1 = average attention
      const r = Math.min(26, 3 + 5.5 * Math.sqrt(w));
      p.ctx.beginPath();
      p.ctx.arc(p.px(all.x.get(i, 0)), p.py(all.x.get(i, 1)), r, 0, Math.PI * 2);
      p.ctx.lineWidth = 1.2;
      p.ctx.strokeStyle = p.palette.css.fg;
      p.ctx.globalAlpha = Math.min(0.9, 0.18 + 0.12 * w);
      p.ctx.stroke();
      p.ctx.globalAlpha = 1;
    }
  };
};

function Rounds({ ctx }: { ctx: ClassifierContext }) {
  const palette = usePalette();
  const model = adaOf(ctx.run.model);
  if (!model) return null;
  const rounds = num(ctx.params, 'rounds');
  return (
    <div className="space-y-6">
      <StepPlayer
        label="Round"
        count={100}
        step={Math.max(0, rounds - 1)}
        onStep={(s) => ctx.setParam('rounds', s + 1)}
        intervalMs={260}
        describe={(s) => `${s + 1} rounds`}
      />
      <p className="text-sm text-muted">
        Rings around the training points show how much attention the latest weak learner paid to
        each one: a point that earlier learners got wrong grows a big ring. After{' '}
        {model.rounds.length} round
        {model.rounds.length === 1 ? '' : 's'}: train {pct(ctx.run.scores.trainAccuracy)}, test{' '}
        <span data-testid="ada-test">{pct(ctx.run.scores.testAccuracy)}</span>.
      </p>
      <div className="grid gap-8 lg:grid-cols-2">
        <MiniChart
          title="Weight (α) each learner gets in the final vote"
          x={model.rounds.map((_, i) => i + 1)}
          xLabel="round"
          series={[
            { name: 'α', color: palette.css.brand, values: model.rounds.map((r) => r.alpha) },
          ]}
          height={170}
        />
        <MiniChart
          title="Weighted error of each learner"
          x={model.rounds.map((_, i) => i + 1)}
          xLabel="round"
          yRange={{ lo: 0, hi: 0.75 }}
          guide={0.5}
          series={[
            { name: 'error', color: palette.css.pos, values: model.rounds.map((r) => r.error) },
          ]}
          note={`Later learners face the points earlier ones failed on, so their error on the reweighted data creeps towards 0.5 (a coin flip); the last is ${fix(model.rounds[model.rounds.length - 1]?.error)}.`}
          height={170}
        />
      </div>
    </div>
  );
}

export default function AdaBoostPage() {
  return (
    <ClassifierLab
      spec={adaboostSpec}
      text={{
        eyebrow: 'Classic · supervised',
        title: 'AdaBoost',
        accent: 'weak learners that learn from mistakes',
        blurb:
          'Add one tiny model at a time, each trained to fix the points the others get wrong. Watch the attention rings migrate to the hard cases.',
      }}
      overlay={weightRings}
      extras={[
        { id: 'lab-rounds', title: 'Round by round', render: (ctx) => <Rounds ctx={ctx} /> },
      ]}
      howItWorks={
        <div className="space-y-3 text-sm text-muted">
          <p>
            Start with equal weights on all points. Train a weak learner (a stump) on the weighted
            data, compute its weighted error ε, and give it a vote α = ln((1 − ε)/ε) (+ ln(K − 1)
            for K classes). Then multiply the weight of every misclassified point by e^α and
            renormalise: the next learner is forced to concentrate on them.
          </p>
          <p>
            The final prediction is the α-weighted vote of all learners. Because it only ever adds
            axis-aligned stumps it is additive in the features, which is why XOR defeats it until
            the learners may ask two questions in a row.
          </p>
        </div>
      }
    />
  );
}
