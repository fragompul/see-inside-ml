import {
  KnnClassifier,
  distanceConcentration,
  looAccuracyByK,
  type Metric,
  type NeighborWeights,
} from '@sim/classic';
import { useMemo } from 'react';
import { usePalette } from '@/shared/canvas/colors';
import { MiniChart } from '@/shared/components/MiniChart';
import { ClassifierLab, num, pct, str, type ClassifierContext } from '@/shared/lab-kit';
import { knnSpec } from './knnSpec';

const MAX_K = 40;

function ChoosingK({ ctx }: { ctx: ClassifierContext }) {
  const palette = usePalette();
  const { train, test } = ctx.run.data;
  const weights = str(ctx.params, 'weights') as NeighborWeights;
  const metric = str(ctx.params, 'metric') as Metric;
  const k = num(ctx.params, 'k');
  const curves = useMemo(() => {
    const maxK = Math.min(MAX_K, train.x.rows - 1);
    const loo = looAccuracyByK(train.x, train.y, maxK, { weights, metric });
    const heldOut =
      test.x.rows > 0
        ? Array.from({ length: maxK }, (_, i) => {
            const m = new KnnClassifier({ k: i + 1, weights, metric }).fit(train.x, train.y);
            const pred = m.predict(test.x);
            return pred.reduce((s, p, r) => s + (p === test.y[r] ? 1 : 0), 0) / test.x.rows;
          })
        : null;
    return { loo, heldOut, ks: Array.from({ length: maxK }, (_, i) => i + 1) };
  }, [train, test, weights, metric]);
  const best = curves.loo.indexOf(Math.max(...curves.loo)) + 1;
  return (
    <div className="grid gap-8 lg:grid-cols-[minmax(0,1fr)_18rem]">
      <MiniChart
        title="Accuracy as k grows"
        x={curves.ks}
        xLabel="k"
        yRange={{ lo: 0.4, hi: 1 }}
        series={[
          { name: 'leave-one-out (training data)', color: palette.css.brand, values: curves.loo },
          ...(curves.heldOut
            ? [
                {
                  name: 'held-out test points',
                  color: palette.css.classes[1] as string,
                  values: curves.heldOut,
                },
              ]
            : []),
        ]}
        height={190}
      />
      <div className="space-y-3 text-sm text-muted">
        <p>
          Leave-one-out classifies each training point from its neighbours <em>without</em> itself,
          so unlike plain training accuracy (always 100% at k = 1) it is an honest guide to the best
          k.
        </p>
        <p data-testid="best-k">
          Best k by leave-one-out: <strong className="text-fg">{best}</strong> (
          {pct(curves.loo[best - 1])}). You are using k = {k}.
        </p>
      </div>
    </div>
  );
}

function Curse() {
  const palette = usePalette();
  const dims = useMemo(() => [1, 2, 3, 5, 8, 12, 20, 35, 60, 100], []);
  const ratio = useMemo(
    () => distanceConcentration(dims, { n: 200, queries: 30, seed: 3 }),
    [dims],
  );
  return (
    <div className="grid gap-8 lg:grid-cols-[minmax(0,1fr)_18rem]">
      <MiniChart
        title="Nearest ÷ farthest distance"
        x={dims}
        xLabel="dimensions"
        yRange={{ lo: 0, hi: 1 }}
        series={[{ name: 'nearest / farthest', color: palette.css.pos, values: ratio }]}
        height={190}
      />
      <p className="text-sm text-muted">
        200 random points in the unit cube. In 1-D the nearest neighbour is tiny compared with the
        farthest point; in 100-D every point is about equally far, so “nearest” stops meaning
        anything. That is why k-NN needs few, relevant features or lots of data.
      </p>
    </div>
  );
}

export default function KnnPage() {
  return (
    <ClassifierLab
      spec={knnSpec}
      text={{
        eyebrow: 'Classic · supervised',
        title: 'k-nearest neighbours',
        accent: 'the data is the model',
        blurb:
          'Classify a point by a vote among its k closest training points. Nothing is learned: watch the boundary crumple at k = 1 and blur when k is huge.',
      }}
      extras={[
        { id: 'lab-k', title: 'Choosing k', render: (ctx) => <ChoosingK ctx={ctx} /> },
        { id: 'lab-curse', title: 'The curse of dimensionality', render: () => <Curse /> },
      ]}
      howItWorks={
        <div className="space-y-3 text-sm text-muted">
          <p>
            To predict, k-NN measures the distance from the query to every training point, takes the
            k nearest and lets them vote (optionally weighted by 1/distance). Training is free; all
            the work happens at prediction time. A small k gives a jagged, low-bias, high-variance
            boundary; a large k a smooth, high-bias, low-variance one.
          </p>
          <p>
            Because it relies on distances, features should be on comparable scales, and it degrades
            when many irrelevant features dilute them.
          </p>
        </div>
      }
    />
  );
}
