import type { TreeNode } from '@sim/classic';
import { useMemo } from 'react';

const LEAF_W = 76;
const LEVEL_H = 84;
const NODE_W = 68;
const NODE_H = 46;

interface Laid {
  node: TreeNode;
  x: number;
  y: number;
  left?: Laid;
  right?: Laid;
}

function layout(root: TreeNode): { laid: Laid; width: number; height: number } {
  let leaf = 0;
  let maxDepth = 0;
  const place = (n: TreeNode): Laid => {
    maxDepth = Math.max(maxDepth, n.depth);
    if (!n.left || !n.right)
      return { node: n, x: leaf++ * LEAF_W + LEAF_W / 2, y: n.depth * LEVEL_H + NODE_H / 2 + 6 };
    const left = place(n.left);
    const right = place(n.right);
    return {
      node: n,
      x: (left.x + right.x) / 2,
      y: n.depth * LEVEL_H + NODE_H / 2 + 6,
      left,
      right,
    };
  };
  const laid = place(root);
  return { laid, width: Math.max(leaf, 1) * LEAF_W, height: (maxDepth + 1) * LEVEL_H + 6 };
}

/**
 * A node-link drawing of a decision tree: each internal node shows its question, each node a bar of
 * its class mix (or its mean for regression). Wide trees scroll sideways instead of shrinking.
 */
export function TreeDiagram({
  root,
  featureNames,
  classColors,
  highlight,
  ariaLabel = 'The decision tree',
}: {
  root: TreeNode;
  featureNames: readonly string[];
  /** Colours of the classes; omit for a regression tree. */
  classColors?: readonly string[];
  /** Nodes on this path are drawn emphasised (the route one example takes). */
  highlight?: ReadonlySet<TreeNode>;
  ariaLabel?: string;
}) {
  const { laid, width, height } = useMemo(() => layout(root), [root]);
  const nodes: Laid[] = [];
  const walk = (l: Laid) => {
    nodes.push(l);
    if (l.left && l.right) {
      walk(l.left);
      walk(l.right);
    }
  };
  walk(laid);
  return (
    <div className="max-w-full overflow-x-auto rounded-xl border border-line bg-surface/40 p-2">
      <svg width={width} height={height} role="img" aria-label={ariaLabel} className="block">
        {nodes.map((l) =>
          l.left && l.right
            ? [l.left, l.right].map((c, k) => (
                <path
                  key={`${l.node.order}-${k}`}
                  d={`M${l.x} ${l.y + NODE_H / 2} C${l.x} ${l.y + NODE_H / 2 + 22} ${c.x} ${c.y - NODE_H / 2 - 22} ${c.x} ${c.y - NODE_H / 2}`}
                  fill="none"
                  strokeWidth={highlight?.has(l.node) && highlight.has(c.node) ? 2.4 : 1.2}
                  className={
                    highlight?.has(l.node) && highlight.has(c.node)
                      ? 'stroke-brand'
                      : 'stroke-line-strong'
                  }
                />
              ))
            : null,
        )}
        {nodes.map((l) => {
          const n = l.node;
          const internal = !!n.left;
          const total = n.value.reduce((a, b) => a + b, 0) || 1;
          const on = highlight?.has(n);
          return (
            <g key={n.order} transform={`translate(${l.x - NODE_W / 2} ${l.y - NODE_H / 2})`}>
              <rect
                width={NODE_W}
                height={NODE_H}
                rx={9}
                strokeWidth={on ? 2 : 1}
                className={`fill-raised ${on ? 'stroke-brand' : 'stroke-line-strong'}`}
              />
              <text
                x={NODE_W / 2}
                y={16}
                textAnchor="middle"
                className="fill-fg font-mono text-[9.5px]"
              >
                {internal
                  ? `${featureNames[n.feature] ?? `x${n.feature + 1}`} ≤ ${n.threshold.toFixed(2)}`
                  : classColors
                    ? `class ${n.value.indexOf(Math.max(...n.value))}`
                    : `≈ ${(n.value[0] ?? 0).toFixed(2)}`}
              </text>
              <text
                x={NODE_W / 2}
                y={28}
                textAnchor="middle"
                className="fill-faint font-mono text-[8.5px]"
              >
                n = {n.samples}
              </text>
              {classColors && (
                <g transform="translate(6 34)">
                  {
                    n.value.reduce<{ x: number; els: React.ReactNode[] }>(
                      (acc, v, c) => {
                        const w = ((NODE_W - 12) * v) / total;
                        acc.els.push(
                          <rect
                            key={c}
                            x={acc.x}
                            width={Math.max(w, 0)}
                            height={6}
                            fill={classColors[c % classColors.length]}
                          />,
                        );
                        acc.x += w;
                        return acc;
                      },
                      { x: 0, els: [] },
                    ).els
                  }
                </g>
              )}
            </g>
          );
        })}
      </svg>
    </div>
  );
}
