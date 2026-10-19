import { dendrogramOrder, type Merge } from '@sim/classic';
import { useMemo } from 'react';
import { usePalette } from '@/shared/canvas/colors';

/**
 * A dendrogram of a linkage matrix: leaves along the bottom in tree order, every merge a "U" whose
 * height is the distance at which the two groups joined. A dashed line shows the cut, and the
 * branches below it are coloured by the cluster they form.
 */
export function Dendrogram({
  merges,
  n,
  cutHeight,
  labels,
  width = 640,
  height = 280,
  ariaLabel = 'Dendrogram of the hierarchical clustering',
}: {
  merges: readonly Merge[];
  n: number;
  /** Height of the cut (drawn dashed); branches fully below it take their cluster's colour. */
  cutHeight: number;
  /** Cluster of every point (the flat clustering at the cut), for colouring. */
  labels: ArrayLike<number>;
  width?: number;
  height?: number;
  ariaLabel?: string;
}) {
  const palette = usePalette();
  const geo = useMemo(() => {
    const order = dendrogramOrder(merges, n);
    const xOf = new Map<number, number>();
    order.forEach((leaf, i) => xOf.set(leaf, i));
    const padL = 36;
    const padB = 8;
    const padT = 10;
    const maxH = Math.max(...merges.map((m) => m.height), 1e-9);
    const sx = (width - padL - 8) / Math.max(1, n - 1);
    const px = (i: number) => padL + i * sx;
    const py = (h: number) => height - padB - (h / maxH) * (height - padB - padT);
    const node = new Map<number, { x: number; h: number }>();
    for (let i = 0; i < n; i++) node.set(i, { x: xOf.get(i) as number, h: 0 });
    const links = merges.map((m, i) => {
      const a = node.get(m.a) as { x: number; h: number };
      const b = node.get(m.b) as { x: number; h: number };
      node.set(n + i, { x: (a.x + b.x) / 2, h: m.height });
      return { a, b, h: m.height, id: n + i, ids: [m.a, m.b] as const };
    });
    // the cluster a merged group belongs to, when everything inside it shares one label
    const members = new Map<number, number[]>();
    for (let i = 0; i < n; i++) members.set(i, [i]);
    merges.forEach((m, i) =>
      members.set(n + i, [...(members.get(m.a) as number[]), ...(members.get(m.b) as number[])]),
    );
    const colour = (id: number): number | null => {
      const set = new Set((members.get(id) as number[]).map((p) => labels[p] as number));
      return set.size === 1 ? ([...set][0] as number) : null;
    };
    return { links, px, py, maxH, colour };
  }, [merges, n, labels, width, height]);

  const classes = palette.css.classes;
  return (
    <figure className="space-y-2">
      <div className="max-w-full overflow-x-auto rounded-xl border border-line bg-surface/40 p-2">
        <svg width={width} height={height} role="img" aria-label={ariaLabel} className="block">
          {geo.links.map((l) => {
            const x1 = geo.px(l.a.x);
            const x2 = geo.px(l.b.x);
            const belowCut = l.h <= cutHeight;
            const c = belowCut ? geo.colour(l.id) : null;
            const stroke = c !== null ? (classes[c % classes.length] as string) : palette.css.muted;
            return (
              <path
                key={l.id}
                d={`M${x1} ${geo.py(l.a.h)} V${geo.py(l.h)} H${x2} V${geo.py(l.b.h)}`}
                fill="none"
                stroke={stroke}
                strokeWidth={belowCut ? 1.8 : 1.2}
                strokeOpacity={belowCut ? 1 : 0.7}
              />
            );
          })}
          <line
            x1={30}
            x2={width}
            y1={geo.py(cutHeight)}
            y2={geo.py(cutHeight)}
            stroke={palette.css.fg}
            strokeDasharray="5 4"
            strokeWidth={1.2}
          />
          <text x={4} y={geo.py(cutHeight) - 4} className="fill-muted font-mono text-[10px]">
            cut
          </text>
        </svg>
      </div>
      <figcaption className="text-[0.74rem] text-faint">
        Height = the distance at which two groups merged. Cutting the tree at the dashed line gives
        the clusters; a tall gap between merges means the groups below it are well separated.
      </figcaption>
    </figure>
  );
}
