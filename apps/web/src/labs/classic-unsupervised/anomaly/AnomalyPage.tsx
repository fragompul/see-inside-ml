import { rocAuc } from '@sim/classic';
import { useMemo } from 'react';
import { usePalette } from '@/shared/canvas/colors';
import { MiniChart } from '@/shared/components/MiniChart';
import { ClusterLab, fix, pct, type ClusterContext, type Metric } from '@/shared/lab-kit';
import { anomalyReport, anomalySpec, detectorOf, strayMask } from './anomalySpec';

function Roc({ ctx }: { ctx: ClusterContext }) {
  const palette = usePalette();
  const { scores } = detectorOf(ctx.run.model);
  const stray = useMemo(() => strayMask(ctx.run.data.y), [ctx.run.data]);
  const curve = useMemo(() => {
    const idx = Array.from(scores.keys()).sort(
      (a, b) => (scores[b] as number) - (scores[a] as number),
    );
    const pos = stray.reduce((a, b) => a + b, 0);
    const neg = stray.length - pos;
    const fpr: number[] = [0];
    const tpr: number[] = [0];
    let tp = 0;
    let fp = 0;
    for (const i of idx) {
      if (stray[i]) tp++;
      else fp++;
      fpr.push(neg ? fp / neg : 0);
      tpr.push(pos ? tp / pos : 0);
    }
    // resample on a regular false-positive grid so the chart has one x value per sample
    const grid = Array.from({ length: 51 }, (_, i) => i / 50);
    const at = grid.map((g) => {
      let best = 0;
      fpr.forEach((f, j) => {
        if (f <= g) best = Math.max(best, tpr[j] as number);
      });
      return best;
    });
    return { grid, at, pos };
  }, [scores, stray]);
  if (curve.pos === 0) {
    return (
      <p className="text-sm text-muted">
        Add some injected outliers (Settings, The data) to see how well the scores separate them
        from the real points.
      </p>
    );
  }
  const auc = rocAuc(scores, stray);
  return (
    <div className="grid gap-8 lg:grid-cols-[minmax(0,1fr)_20rem]">
      <MiniChart
        title="ROC curve: strays caught vs real points wrongly flagged"
        x={curve.grid}
        xLabel="share of real points flagged"
        yRange={{ lo: 0, hi: 1 }}
        series={[
          { name: 'detector', color: palette.css.brand, values: curve.at },
          { name: 'guessing', color: palette.css.muted, values: curve.grid, dashed: true },
        ]}
        height={220}
      />
      <div className="space-y-3 text-sm text-muted" data-testid="roc-note">
        <p>
          Sweep the flagging threshold from strict to lenient: the curve shows how many strays are
          caught for each share of real points flagged by mistake. A perfect ranking hugs the top
          left corner; guessing follows the dashed diagonal.
        </p>
        <p>
          The area under the curve is <strong className="text-fg">{fix(auc, 3)}</strong>: the chance
          that a random stray gets a higher score than a random real point. It judges the ranking
          and ignores where you cut.
        </p>
      </div>
    </div>
  );
}

function ScoreStrip({ ctx }: { ctx: ClusterContext }) {
  const palette = usePalette();
  const d = detectorOf(ctx.run.model);
  const stray = strayMask(ctx.run.data.y);
  const lo = Math.min(...d.scores);
  const hi = Math.max(...d.scores);
  const w = 640;
  const h = 70;
  const px = (v: number) => 8 + ((v - lo) / (hi - lo || 1)) * (w - 16);
  return (
    <div className="space-y-3">
      <svg
        viewBox={`0 0 ${w} ${h}`}
        className="w-full"
        role="img"
        aria-label="Every point scored on one line, strays above real points"
      >
        <line x1={8} x2={w - 8} y1={h - 18} y2={h - 18} stroke={palette.css.muted} />
        {Array.from(d.scores, (s, i) => (
          <circle
            key={i}
            cx={px(s)}
            cy={stray[i] ? 24 : 40}
            r={3}
            fill={stray[i] ? palette.css.neg : palette.css.brand}
            opacity={0.55}
          />
        ))}
        <line
          x1={px(d.threshold)}
          x2={px(d.threshold)}
          y1={6}
          y2={h - 14}
          stroke={palette.css.fg}
          strokeDasharray="4 3"
        />
        <text
          x={Math.min(px(d.threshold) + 5, w - 70)}
          y={12}
          className="fill-muted font-mono text-[10px]"
        >
          threshold
        </text>
      </svg>
      <p className="text-sm text-muted">
        One dot per point, placed by its score (upper row: the injected strays, lower row: the real
        points). Everything right of the dashed line is flagged. Real points sit in a tight bunch
        and strays trail off to the right, but the two overlap, and that overlap is the errors.
      </p>
    </div>
  );
}

export default function AnomalyPage() {
  return (
    <ClusterLab
      spec={anomalySpec}
      anomaly
      showRegions={false}
      datasets={['blobs', 'moons', 'circles', 'spirals']}
      text={{
        eyebrow: 'Classic · unsupervised',
        title: 'Anomaly detection',
        accent: 'spot the odd one out',
        blurb:
          'Score every point by how unusual it is, by density, by isolation or by distance, then flag the top few. Strays are scattered over the plane so you can see which detectors find them.',
      }}
      highlight={(ctx) => {
        const d = detectorOf(ctx.run.model);
        return new Set(Array.from(d.labels.keys()).filter((i) => d.labels[i] === 1));
      }}
      metricsOnly={(ctx): Metric[] => {
        const r = anomalyReport(ctx.run.model, ctx.run.data.y);
        return [
          { label: 'Flagged', value: String(r.flagged), testId: 'flagged' },
          { label: 'Strays in the data', value: String(r.strays), testId: 'strays' },
          {
            label: 'Ranking quality (AUC)',
            value: r.strays ? fix(r.auc, 3) : '–',
            testId: 'auc',
            hint: 'chance that a random stray outscores a random real point (1 = perfect, 0.5 = guessing)',
          },
          {
            label: 'Precision / recall',
            value: r.strays ? `${pct(r.precision)} / ${pct(r.recall)}` : '–',
            testId: 'precision',
            hint: 'of the flagged points, how many are strays / of the strays, how many were flagged',
          },
        ];
      }}
      extras={[
        { id: 'lab-roc', title: 'How good is the ranking?', render: (ctx) => <Roc ctx={ctx} /> },
        { id: 'lab-scores', title: 'The scores', render: (ctx) => <ScoreStrip ctx={ctx} /> },
      ]}
      howItWorks={
        <div className="space-y-3 text-sm text-muted">
          <p>
            <strong className="text-fg">Local outlier factor</strong> compares each point&apos;s
            density with that of its k nearest neighbours. A score near 1 means &ldquo;as crowded as
            my neighbours&rdquo;; well above 1 means &ldquo;much lonelier than the points around
            me&rdquo;, which is why it copes with clusters of different density.
          </p>
          <p>
            <strong className="text-fg">Isolation forest</strong> builds random trees that cut a
            random feature at a random value. Points in a crowd need many cuts to be fenced in,
            strays only a few: the average depth, normalised by what a random search tree would
            give, becomes the score.
          </p>
          <p>
            <strong className="text-fg">Mahalanobis distance</strong> measures how many standard
            deviations a point is from the mean after accounting for the correlations. It is exact
            for one elliptical cloud and wrong for anything curved or in several groups.
          </p>
        </div>
      }
    />
  );
}
