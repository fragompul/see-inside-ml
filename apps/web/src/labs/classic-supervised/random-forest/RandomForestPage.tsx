import { useMemo } from 'react';
import { usePalette } from '@/shared/canvas/colors';
import { MiniChart } from '@/shared/components/MiniChart';
import {
  ClassifierLab,
  RegionThumb,
  fix,
  pct,
  predictRegions,
  type ClassifierContext,
} from '@/shared/lab-kit';
import { forestOf, forestSpec } from './forestSpec';

function Disagreement({ ctx }: { ctx: ClassifierContext }) {
  const forest = forestOf(ctx.run.model);
  const thumbs = useMemo(() => {
    if (!forest) return [];
    return forest.trees.slice(0, 7).map((t) => predictRegions(t, ctx.run.data, 48));
  }, [forest, ctx.run.data]);
  if (!forest) return null;
  return (
    <div className="space-y-4">
      <div className="flex flex-wrap gap-4">
        {thumbs.map((r, i) => (
          <RegionThumb key={i} regions={r} label={`tree ${i + 1}`} />
        ))}
        <RegionThumb regions={ctx.regions} label="the forest" size={168} />
      </div>
      <p className="text-sm text-muted">
        Each tree saw a different bootstrap sample and a random feature at every split, so each one
        draws a different, jagged boundary. The forest (right) averages all {forest.trees.length}{' '}
        votes: the errors that differ from tree to tree cancel, the signal they share remains.
      </p>
    </div>
  );
}

function Oob({ ctx }: { ctx: ClassifierContext }) {
  const palette = usePalette();
  const forest = forestOf(ctx.run.model);
  const curve = useMemo(
    () =>
      forest && forest.oobAccuracy !== null
        ? forest.oobCurve(ctx.run.data.train.x, ctx.run.data.train.y)
        : [],
    [forest, ctx.run.data],
  );
  if (!forest || curve.length === 0) {
    return (
      <p className="text-sm text-muted">
        Out-of-bag estimates need the bootstrap switched on: without it every tree sees every row.
      </p>
    );
  }
  return (
    <div className="grid gap-8 lg:grid-cols-[minmax(0,1fr)_18rem]">
      <MiniChart
        title="Out-of-bag accuracy as trees are added"
        x={curve.map((_, i) => i + 1)}
        xLabel="trees"
        yRange={{ lo: 0.5, hi: 1 }}
        series={[{ name: 'out-of-bag accuracy', color: palette.css.brand, values: curve }]}
        height={200}
      />
      <div className="space-y-3 text-sm text-muted">
        <p>
          Each tree is trained on about 63% of the rows. A row is judged only by the trees that left
          it out, which makes this a fair estimate of generalisation, for free.
        </p>
        <p data-testid="oob-summary">
          Out-of-bag accuracy <strong className="text-fg">{pct(forest.oobAccuracy)}</strong>; the
          held-out test points give {pct(ctx.run.scores.testAccuracy)}.
        </p>
      </div>
    </div>
  );
}

function Importance({ ctx }: { ctx: ClassifierContext }) {
  const forest = forestOf(ctx.run.model);
  if (!forest) return null;
  return (
    <ul className="space-y-2">
      {ctx.run.data.all.names.map((n, j) => (
        <li key={n} className="flex items-center gap-3 font-mono text-[0.8rem]">
          <span className="w-8">{n}</span>
          <span className="h-2 flex-1 overflow-hidden rounded-full bg-raised">
            <span
              className="block h-full rounded-full bg-brand"
              style={{ width: `${(forest.importances[j] ?? 0) * 100}%` }}
            />
          </span>
          <span className="w-12 text-right tabular-nums">{fix(forest.importances[j])}</span>
        </li>
      ))}
    </ul>
  );
}

export default function RandomForestPage() {
  return (
    <ClassifierLab
      spec={forestSpec}
      text={{
        eyebrow: 'Classic · supervised',
        title: 'Random forests',
        accent: 'a crowd of overfit trees',
        blurb:
          'Grow many deep trees on random resamples, let them vote, and watch the noise average out. The out-of-bag estimate comes for free.',
      }}
      extras={[
        {
          id: 'lab-trees',
          title: 'Trees disagree, the forest averages',
          render: (ctx) => <Disagreement ctx={ctx} />,
        },
        { id: 'lab-oob', title: 'Out-of-bag error', render: (ctx) => <Oob ctx={ctx} /> },
        {
          id: 'lab-importance',
          title: 'Which features matter?',
          render: (ctx) => <Importance ctx={ctx} />,
        },
      ]}
      howItWorks={
        <div className="space-y-3 text-sm text-muted">
          <p>
            A random forest is bagging plus randomness. Each tree is trained on a bootstrap sample
            (n rows drawn with replacement) and, at every split, may only look at a random subset of
            the features. Individually the trees overfit; but their mistakes are largely
            independent, so the average of many of them has much lower variance and about the same
            bias.
          </p>
          <p>
            More trees never overfit: they only reduce the variance until it stops improving. The
            rows a tree did not see (out-of-bag) provide a validation set for free.
          </p>
        </div>
      }
    />
  );
}
