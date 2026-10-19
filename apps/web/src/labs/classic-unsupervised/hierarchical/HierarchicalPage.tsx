import { Agglomerative, copheneticCorrelation, type Linkage } from '@sim/classic';
import { useMemo } from 'react';
import {
  ClusterLab,
  Dendrogram,
  fix,
  num,
  scoreClusters,
  str,
  type ClusterContext,
} from '@/shared/lab-kit';
import { agglomerativeOf, hierarchicalSpec } from './hierarchicalSpec';

const LINKAGES: Linkage[] = ['single', 'complete', 'average', 'ward'];
const NAMES: Record<Linkage, string> = {
  single: 'Single (nearest pair)',
  complete: 'Complete (farthest pair)',
  average: 'Average (mean pair)',
  ward: 'Ward (least variance added)',
};

function Tree({ ctx }: { ctx: ClusterContext }) {
  const m = agglomerativeOf(ctx.run.model);
  const k = num(ctx.params, 'k');
  const n = ctx.run.data.x.rows;
  const cut = useMemo(() => {
    if (!m) return 0;
    const heights = m.merges.map((x) => x.height);
    const top = heights[heights.length - 1] as number;
    if (k <= 1) return top * 1.05;
    // between the last merge kept and the first one undone
    return ((heights[n - k - 1] ?? 0) + (heights[n - k] ?? top)) / 2;
  }, [m, k, n]);
  if (!m) return null;
  const coph = copheneticCorrelation(m.distances, n, m.merges);
  return (
    <div className="space-y-4">
      <Dendrogram merges={m.merges} n={n} cutHeight={cut} labels={ctx.run.labels} />
      <p className="text-sm text-muted" data-testid="tree-summary">
        Cut into <strong className="text-fg">{k}</strong> clusters at height {fix(cut, 2)}. The
        tree’s distances match the real ones with correlation{' '}
        <strong className="text-fg">{fix(coph, 3)}</strong> (the cophenetic correlation: closer to 1
        means the tree distorts the geometry less). Change k to move the cut line.
      </p>
    </div>
  );
}

function Compare({ ctx }: { ctx: ClusterContext }) {
  const { data } = ctx.run;
  const k = num(ctx.params, 'k');
  const current = str(ctx.params, 'linkage');
  const rows = useMemo(
    () =>
      LINKAGES.map((linkage) => {
        const model = new Agglomerative({ k, linkage }).fit(data.x);
        const s = scoreClusters(data, model.labels);
        const sizes = Array.from(
          { length: k },
          (_, c) => model.labels.filter((l) => l === c).length,
        ).sort((a, b) => b - a);
        return { linkage, ...s, sizes };
      }),
    [data, k],
  );
  return (
    <div className="space-y-4">
      <div className="overflow-x-auto">
        <table className="w-full min-w-[30rem] text-left font-mono text-[0.8rem] tabular-nums">
          <thead className="text-muted">
            <tr>
              {['linkage', 'match with truth', 'silhouette', 'cluster sizes'].map((h) => (
                <th key={h} className="pr-4 pb-2 font-medium">
                  {h}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr
                key={r.linkage}
                className={`border-t border-line ${r.linkage === current ? 'text-brand' : ''}`}
              >
                <td className="py-1.5 pr-4">{NAMES[r.linkage]}</td>
                <td className="pr-4">{fix(r.ari)}</td>
                <td className="pr-4">{r.silhouette === null ? '–' : fix(r.silhouette)}</td>
                <td>{r.sizes.slice(0, 6).join(' / ')}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="text-sm text-muted">
        The same data, four ways of measuring the distance between two clusters. Single linkage can
        follow elongated shapes but chains through noise and peels off single points; complete and
        Ward favour compact groups. There is no best linkage: it encodes what you mean by a cluster.
      </p>
    </div>
  );
}

export default function HierarchicalPage() {
  return (
    <ClusterLab
      spec={hierarchicalSpec}
      text={{
        eyebrow: 'Classic · unsupervised',
        title: 'Hierarchical clustering',
        accent: 'a tree of merges',
        blurb:
          'Start with every point alone and keep merging the two closest groups. The result is a whole tree, and you choose where to cut it.',
      }}
      showRegions={false}
      extras={[
        { id: 'lab-tree', title: 'The dendrogram', render: (ctx) => <Tree ctx={ctx} /> },
        {
          id: 'lab-linkage',
          title: 'Four linkages on the same data',
          render: (ctx) => <Compare ctx={ctx} />,
        },
      ]}
      howItWorks={
        <div className="space-y-3 text-sm text-muted">
          <p>
            Agglomerative clustering begins with n clusters of one point and repeatedly merges the
            two closest, recording the distance at which each merge happens. How far apart two
            clusters are is the <em>linkage</em>: the closest pair of points (single), the farthest
            (complete), the average over all pairs (average), or the increase in total variance the
            merge would cause (Ward).
          </p>
          <p>
            The merge heights form a dendrogram. A long vertical stretch with no merges means the
            groups below it are clearly separate, so cutting there is a natural choice of k. Unlike
            k-means there is no random start and no need to fix k in advance.
          </p>
        </div>
      }
    />
  );
}
