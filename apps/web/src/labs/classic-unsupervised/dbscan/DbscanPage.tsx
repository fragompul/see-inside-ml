import { kDistances, optics, opticsClusters } from '@sim/classic';
import { useMemo } from 'react';
import { usePalette } from '@/shared/canvas/colors';
import { MiniChart } from '@/shared/components/MiniChart';
import { ClusterLab, fix, num, type ClusterContext } from '@/shared/lab-kit';
import { dbscanOf, dbscanSpec } from './dbscanSpec';

function KDistance({ ctx }: { ctx: ClusterContext }) {
  const palette = usePalette();
  const k = num(ctx.params, 'minSamples');
  const eps = num(ctx.params, 'eps');
  const d = useMemo(() => kDistances(ctx.run.data.x, k), [ctx.run.data, k]);
  return (
    <div className="grid gap-8 lg:grid-cols-[minmax(0,1fr)_20rem]">
      <MiniChart
        title={`Distance to the ${k}th nearest neighbour, sorted`}
        x={d.map((_, i) => i)}
        xLabel="points, from densest to sparsest"
        guide={eps}
        series={[{ name: `${k}-distance`, color: palette.css.brand, values: d }]}
        height={190}
      />
      <p className="text-sm text-muted">
        A good ε sits at the “knee” of this curve: below it most points have {k} neighbours that
        close (they are in dense regions), above it the distance shoots up (the sparse points, i.e.
        noise). The dashed line is your ε = {eps}.
      </p>
    </div>
  );
}

function Reachability({ ctx }: { ctx: ClusterContext }) {
  const palette = usePalette();
  const minSamples = num(ctx.params, 'minSamples');
  const eps = num(ctx.params, 'eps');
  const result = useMemo(() => optics(ctx.run.data.x, minSamples), [ctx.run.data, minSamples]);
  const labels = useMemo(() => opticsClusters(result, eps), [result, eps]);
  const finite = Array.from(result.reachability).filter(Number.isFinite);
  const top = Math.max(...finite, eps) * 1.1;
  const n = result.ordering.length;
  const W = Math.max(480, n * 3);
  const H = 160;
  const y = (v: number) => H - 6 - (Math.min(v, top) / top) * (H - 14);
  const found = new Set(Array.from(labels).filter((l) => l >= 0)).size;
  return (
    <div className="space-y-4">
      <div className="max-w-full overflow-x-auto rounded-xl border border-line bg-surface/40 p-2">
        <svg
          width={W}
          height={H}
          role="img"
          aria-label="OPTICS reachability plot: valleys are clusters"
          className="block"
        >
          {result.ordering.map((p, i) => {
            const r = result.reachability[p] as number;
            const label = labels[p] as number;
            const color =
              label < 0
                ? palette.css.muted
                : (palette.css.classes[label % palette.css.classes.length] as string);
            const h = Number.isFinite(r) ? H - 6 - y(r) : H - 8;
            return (
              <rect
                key={p}
                x={(i * (W - 8)) / n + 4}
                y={H - 6 - h}
                width={Math.max(1.5, (W - 8) / n - 0.5)}
                height={h}
                fill={color}
                opacity={Number.isFinite(r) ? 0.9 : 0.35}
              />
            );
          })}
          <line
            x1={0}
            x2={W}
            y1={y(eps)}
            y2={y(eps)}
            stroke={palette.css.fg}
            strokeDasharray="5 4"
          />
          <text x={6} y={y(eps) - 4} className="fill-muted font-mono text-[10px]">
            ε = {eps}
          </text>
        </svg>
      </div>
      <p className="text-sm text-muted" data-testid="optics-summary">
        OPTICS visits the points in an order that keeps neighbours together and records how far each
        one is “reachable” from the previous dense region.{' '}
        <strong className="text-fg">Valleys are clusters</strong>, tall spikes are the jumps between
        them. Cutting the plot at ε gives the DBSCAN clustering for that ε ({fix(found, 0)} cluster
        {found === 1 ? '' : 's'} here); dragging ε up or down shows every density at once, which is
        how OPTICS copes with clusters of different densities.
      </p>
    </div>
  );
}

export default function DbscanPage() {
  return (
    <ClusterLab
      spec={dbscanSpec}
      text={{
        eyebrow: 'Classic · unsupervised',
        title: 'DBSCAN & OPTICS',
        accent: 'clusters are dense regions',
        blurb:
          'Grow clusters through crowded neighbourhoods and leave the lonely points as noise. No k to choose, and the shapes can be anything.',
      }}
      showRegions={false}
      highlight={(ctx) => {
        const m = dbscanOf(ctx.run.model);
        return m ? new Set(m.coreIndices) : undefined;
      }}
      extras={[
        {
          id: 'lab-kdist',
          title: 'Choosing ε: the k-distance plot',
          render: (ctx) => <KDistance ctx={ctx} />,
        },
        {
          id: 'lab-optics',
          title: 'OPTICS: every ε at once',
          render: (ctx) => <Reachability ctx={ctx} />,
        },
      ]}
      howItWorks={
        <div className="space-y-3 text-sm text-muted">
          <p>
            A point is a <strong className="text-fg">core point</strong> if at least{' '}
            <em>min samples</em> points (itself included) lie within ε of it (ringed on the plot). A
            cluster is a set of core points that are connected through overlapping neighbourhoods,
            together with the non-core points on its edge (
            <strong className="text-fg">border points</strong>). Anything else is{' '}
            <strong className="text-fg">noise</strong>.
          </p>
          <p>
            DBSCAN needs no k, finds arbitrary shapes and rejects outliers, but a single ε and
            density threshold must suit the whole dataset. OPTICS sidesteps that by ordering the
            points once and drawing the reachability plot, from which clusterings at all ε can be
            read off.
          </p>
        </div>
      }
    />
  );
}
