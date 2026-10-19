import { KMeans, initialCenters, kmeansSteps, type KMeansInit } from '@sim/classic';
import type { Table } from '@sim/core';
import { plotBounds } from '@sim/core';
import { useMemo, useState } from 'react';
import { usePalette } from '@/shared/canvas/colors';
import { MiniChart } from '@/shared/components/MiniChart';
import {
  ClusterLab,
  DataPlane,
  StepPlayer,
  fix,
  num,
  scoreClusters,
  str,
  type ClusterContext,
  type Painter,
} from '@/shared/lab-kit';
import { kmeansOf, kmeansSpec } from './kmeansSpec';

/** Draws the centres as big rings (and where they came from). */
const centresOverlay = (ctx: ClusterContext) => {
  const m = kmeansOf(ctx.run.model);
  if (!m) return undefined;
  return (p: Painter) => {
    for (let c = 0; c < m.centers.rows; c++) {
      const color = p.palette.css.classes[c % p.palette.css.classes.length] as string;
      p.ctx.beginPath();
      p.ctx.arc(p.px(m.centers.get(c, 0)), p.py(m.centers.get(c, 1)), 9, 0, Math.PI * 2);
      p.ctx.fillStyle = color;
      p.ctx.fill();
      p.ctx.lineWidth = 2.5;
      p.ctx.strokeStyle = p.palette.css.surface;
      p.ctx.stroke();
      p.ctx.lineWidth = 1.5;
      p.ctx.strokeStyle = p.palette.css.fg;
      p.ctx.beginPath();
      p.ctx.arc(p.px(m.centers.get(c, 0)), p.py(m.centers.get(c, 1)), 11.5, 0, Math.PI * 2);
      p.ctx.stroke();
    }
  };
};

function Converge({ ctx }: { ctx: ClusterContext }) {
  const k = num(ctx.params, 'k');
  const init = str(ctx.params, 'init') as KMeansInit;
  const seed = num(ctx.params, 'seed');
  const { data } = ctx.run;
  const [step, setStep] = useState(0);
  const states = useMemo(
    () => Array.from(kmeansSteps(data.x, initialCenters(data.x, k, init, seed))),
    [data, k, init, seed],
  );
  const s = states[Math.min(step, states.length - 1)];
  const bounds = useMemo(() => plotBounds(data.x, 0.2), [data]);
  const painted = useMemo<Table | null>(
    () => (s ? { ...data, y: Float64Array.from(s.labels) } : null),
    [s, data],
  );
  const noTest = useMemo(() => new Uint8Array(data.x.rows), [data]);
  const overlay = useMemo(() => {
    if (!s) return undefined;
    const upTo = states.slice(0, Math.min(step, states.length - 1) + 1);
    return (p: Painter) => {
      for (let c = 0; c < k; c++) {
        const color = p.palette.css.classes[c % p.palette.css.classes.length] as string;
        p.ctx.beginPath();
        upTo.forEach((st, i) => {
          const x = p.px(st.centers.get(c, 0));
          const y = p.py(st.centers.get(c, 1));
          if (i === 0) p.ctx.moveTo(x, y);
          else p.ctx.lineTo(x, y);
        });
        p.ctx.strokeStyle = color;
        p.ctx.lineWidth = 2;
        p.ctx.stroke();
        p.ctx.beginPath();
        p.ctx.arc(p.px(s.centers.get(c, 0)), p.py(s.centers.get(c, 1)), 8, 0, Math.PI * 2);
        p.ctx.fillStyle = color;
        p.ctx.fill();
        p.ctx.lineWidth = 2;
        p.ctx.strokeStyle = p.palette.css.fg;
        p.ctx.stroke();
      }
    };
  }, [s, states, step, k]);
  if (!s || !painted) return null;
  return (
    <div className="grid gap-8 lg:grid-cols-[auto_minmax(0,1fr)]">
      <DataPlane
        all={painted}
        testMask={noTest}
        bounds={bounds}
        width={420}
        height={320}
        {...(overlay ? { overlay } : {})}
        ariaLabel="One run of k-means, step by step: points coloured by their nearest centre, with the trail of every centre."
      />
      <div className="min-w-0 space-y-5">
        <StepPlayer
          label="Iteration"
          count={states.length}
          step={step}
          onStep={setStep}
          intervalMs={700}
          describe={(i) => (i === 0 ? 'start' : `after update ${i}`)}
        />
        <p className="font-mono text-[0.8rem] tabular-nums" data-testid="lloyd-readout">
          iteration {s.iteration}: inertia {fix(s.inertia, 2)}
          {s.iteration > 0 ? `, centres moved ${fix(Math.sqrt(s.shift), 3)}` : ''}
        </p>
        <p className="text-sm text-muted">
          Each round does two things: <strong className="text-fg">assign</strong> every point to its
          nearest centre (the colours), then <strong className="text-fg">update</strong> every
          centre to the mean of its points (the dots jump). Inertia can only go down, and the run
          stops when nothing changes. The trails show where each centre has been: change the seed or
          the start type and watch a different path unfold.
        </p>
      </div>
    </div>
  );
}

function Choosing({ ctx }: { ctx: ClusterContext }) {
  const palette = usePalette();
  const { data } = ctx.run;
  const k = num(ctx.params, 'k');
  const curve = useMemo(() => {
    const ks = Array.from({ length: 8 }, (_, i) => i + 1);
    const fits = ks.map((kk) => {
      const m = new KMeans({ k: kk, restarts: 4, seed: 2 }).fit(data.x);
      return { k: kk, inertia: m.inertia, labels: m.labels };
    });
    return { ks, fits };
  }, [data]);
  const sil = useMemo(
    () =>
      curve.fits.map((f) => {
        if (f.k < 2) return NaN;
        // silhouette through the same scoring helper the metrics strip uses
        return silhouetteOf(data, f.labels);
      }),
    [curve, data],
  );
  const bestSil = curve.ks[sil.indexOf(Math.max(...sil.filter((v) => !Number.isNaN(v))))];
  return (
    <div className="grid gap-8 lg:grid-cols-2">
      <MiniChart
        title="Inertia (the elbow)"
        x={curve.ks}
        xLabel="k"
        series={[
          { name: 'inertia', color: palette.css.brand, values: curve.fits.map((f) => f.inertia) },
        ]}
        note="Always falls as k grows. Look for the elbow where adding a cluster stops paying off."
        height={180}
      />
      <MiniChart
        title="Silhouette"
        x={curve.ks.slice(1)}
        xLabel="k"
        yRange={{ lo: 0, hi: 1 }}
        series={[{ name: 'silhouette', color: palette.css.pos, values: sil.slice(1) }]}
        note={`How well separated the clusters are. Highest at k = ${bestSil}; you are using k = ${k}.`}
        height={180}
      />
    </div>
  );
}

const silhouetteOf = (data: Table, labels: Int32Array) =>
  scoreClusters(data, labels).silhouette ?? NaN;

function Lottery({ ctx }: { ctx: ClusterContext }) {
  const { data } = ctx.run;
  const k = num(ctx.params, 'k');
  const init = str(ctx.params, 'init') as KMeansInit;
  const inertias = useMemo(
    () =>
      Array.from(
        { length: 20 },
        (_, s) => new KMeans({ k, init, restarts: 1, seed: s + 1 }).fit(data.x).inertia,
      ),
    [data, k, init],
  );
  const best = Math.min(...inertias);
  const worst = Math.max(...inertias);
  const good = inertias.filter((v) => v <= best * 1.01).length;
  const sorted = [...inertias].sort((a, b) => a - b);
  return (
    <div className="space-y-4">
      <svg
        viewBox="0 0 400 90"
        role="img"
        aria-label="Final inertia of twenty single runs from twenty different starts, sorted"
        className="block w-full max-w-[28rem]"
      >
        {sorted.map((v, i) => {
          const h = 8 + (worst > best ? ((v - best) / (worst - best)) * 62 : 0);
          return (
            <rect
              key={i}
              x={i * 20 + 2}
              y={84 - h}
              width={16}
              height={h}
              rx={2}
              className={v <= best * 1.01 ? 'fill-ok/80' : 'fill-neg/70'}
            />
          );
        })}
        <line x1={0} x2={400} y1={84.5} y2={84.5} className="stroke-line-strong" />
      </svg>
      <p className="text-sm text-muted" data-testid="lottery-summary">
        Twenty single runs of k-means from twenty different starting seeds (
        {init === 'random' ? 'random points' : 'k-means++'}):{' '}
        <strong className="text-fg">{good}</strong> of 20 reached the best inertia found (
        {fix(best, 1)}); the worst stopped at {fix(worst, 1)}. Because the outcome depends on the
        start, practical k-means runs from several starts and keeps the best.
      </p>
    </div>
  );
}

export default function KMeansPage() {
  return (
    <ClusterLab
      spec={kmeansSpec}
      text={{
        eyebrow: 'Classic · unsupervised',
        title: 'k-means',
        accent: 'assign, then move',
        blurb:
          'Drop k centres, give each point to its nearest one, move every centre to the middle of its points, repeat. Simple, fast, and full of instructive failures.',
      }}
      overlay={centresOverlay}
      extras={[
        { id: 'lab-converge', title: 'Watch it converge', render: (ctx) => <Converge ctx={ctx} /> },
        { id: 'lab-choose', title: 'Choosing k', render: (ctx) => <Choosing ctx={ctx} /> },
        {
          id: 'lab-lottery',
          title: 'The initialisation lottery',
          render: (ctx) => <Lottery ctx={ctx} />,
        },
      ]}
      howItWorks={
        <div className="space-y-3 text-sm text-muted">
          <p>
            k-means minimises the inertia: the sum of squared distances from each point to its
            cluster’s centre. Lloyd’s algorithm alternates two steps, each of which can only lower
            it, so it always stops, but only at a <em>local</em> minimum that depends on the
            starting centres. k-means++ picks them far apart (each new one is drawn with probability
            proportional to its squared distance to the nearest chosen); restarts add robustness.
          </p>
          <p>
            Because every cluster is the set of points closest to its centre, clusters are convex
            cells of a Voronoi diagram: round blobs of similar size are fine, crescents, rings and
            very unequal densities are not. That is when to reach for spectral clustering, DBSCAN or
            a Gaussian mixture.
          </p>
        </div>
      }
    />
  );
}
