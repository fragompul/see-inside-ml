import {
  DecisionTree,
  leafCount,
  nodeCount,
  pruningPath,
  treeDepth,
  type Criterion,
  type TreeNode,
} from '@sim/classic';
import { plotBounds } from '@sim/core';
import { useMemo } from 'react';
import { usePalette } from '@/shared/canvas/colors';
import { MiniChart } from '@/shared/components/MiniChart';
import {
  ClassifierLab,
  StepPlayer,
  TreeDiagram,
  fix,
  num,
  pct,
  str,
  type ClassifierContext,
  type Painter,
} from '@/shared/lab-kit';
import { treeOf, treeSpec } from './treeSpec';

/** Draws every split as a line clipped to the region of its node. */
const splits = (ctx: ClassifierContext) => {
  const tree = treeOf(ctx.run.model);
  if (!tree) return undefined;
  const b = plotBounds(ctx.run.data.all.x, 0.2);
  return (p: Painter) => {
    p.ctx.lineWidth = 1.6;
    p.ctx.strokeStyle = p.palette.css.fg;
    p.ctx.setLineDash([]);
    const draw = (n: TreeNode, x0: number, x1: number, y0: number, y1: number) => {
      if (!n.left || !n.right) return;
      p.ctx.beginPath();
      if (n.feature === 0) {
        p.ctx.moveTo(p.px(n.threshold), p.py(y0));
        p.ctx.lineTo(p.px(n.threshold), p.py(y1));
        p.ctx.stroke();
        draw(n.left, x0, n.threshold, y0, y1);
        draw(n.right, n.threshold, x1, y0, y1);
      } else {
        p.ctx.moveTo(p.px(x0), p.py(n.threshold));
        p.ctx.lineTo(p.px(x1), p.py(n.threshold));
        p.ctx.stroke();
        draw(n.left, x0, x1, y0, n.threshold);
        draw(n.right, x0, x1, n.threshold, y1);
      }
    };
    draw(tree.root, b.x0, b.x1, b.y0, b.y1);
  };
};

function TreeView({ ctx }: { ctx: ClassifierContext }) {
  const palette = usePalette();
  const tree = treeOf(ctx.run.model);
  if (!tree) return null;
  return (
    <div className="space-y-3">
      <TreeDiagram
        root={tree.root}
        featureNames={ctx.run.data.all.names}
        classColors={palette.css.classes}
      />
      <p className="text-sm text-muted" data-testid="tree-stats">
        {nodeCount(tree.root)} nodes, {leafCount(tree.root)} leaves, depth {treeDepth(tree.root)}.
        Each box asks a yes/no question about one feature (yes goes left); the coloured bar is the
        mix of classes that reach it, so a pure leaf is a single colour.
      </p>
    </div>
  );
}

function Growth({ ctx }: { ctx: ClassifierContext }) {
  const depth = num(ctx.params, 'maxDepth');
  return (
    <div className="space-y-4">
      <p className="text-sm text-muted">
        Step through the maximum depth to watch the tree grow: each level adds a round of splits and
        the plane is cut into smaller and smaller rectangles.
      </p>
      <StepPlayer
        label="Depth"
        count={12}
        step={Math.max(0, depth - 1)}
        onStep={(s) => {
          ctx.setParam('maxDepth', s + 1);
        }}
        intervalMs={650}
        describe={(s) => `depth ${s + 1}`}
      />
      <p className="font-mono text-[0.8rem] tabular-nums" data-testid="growth-readout">
        depth {depth}: train {pct(ctx.run.scores.trainAccuracy)}, test{' '}
        {pct(ctx.run.scores.testAccuracy)}
      </p>
    </div>
  );
}

function Pruning({ ctx }: { ctx: ClassifierContext }) {
  const palette = usePalette();
  const criterion = str(ctx.params, 'criterion') as Criterion;
  const { train, test } = ctx.run.data;
  const result = useMemo(() => {
    const full = new DecisionTree({ criterion }).fit(train.x, train.y);
    const path = pruningPath(full.root);
    const alphas = path.alphas.filter((a, i) => i > 0 && a > 1e-9);
    const unique = alphas.filter((a, i) => i === 0 || a - (alphas[i - 1] as number) > 1e-9);
    const rows = unique.map((a) => {
      const m = new DecisionTree({ criterion, ccpAlpha: a }).fit(train.x, train.y);
      const acc = (t: typeof train) =>
        t.x.rows
          ? Array.from(m.predict(t.x)).filter((p, i) => p === t.y[i]).length / t.x.rows
          : NaN;
      return { alpha: a, leaves: leafCount(m.root), train: acc(train), test: acc(test) };
    });
    return { rows, fullLeaves: leafCount(full.root) };
  }, [train, test, criterion]);
  if (result.rows.length < 2) {
    return (
      <p className="text-sm text-muted">The tree is already tiny; there is nothing to prune.</p>
    );
  }
  const best = result.rows.reduce((b, r) => (r.test > b.test ? r : b), result.rows[0]!);
  return (
    <div className="grid gap-8 lg:grid-cols-[minmax(0,1fr)_18rem]">
      <MiniChart
        title="Accuracy as the tree is pruned harder"
        x={result.rows.map((r) => Math.log10(r.alpha))}
        xLabel="log10 α (right = smaller tree)"
        yRange={{ lo: 0.5, hi: 1 }}
        series={[
          { name: 'training', color: palette.css.brand, values: result.rows.map((r) => r.train) },
          { name: 'test', color: palette.css.pos, values: result.rows.map((r) => r.test) },
        ]}
        height={200}
      />
      <div className="space-y-3 text-sm text-muted">
        <p>
          The full tree has {result.fullLeaves} leaves. Pruning removes the branch whose accuracy
          gain per leaf is smallest, one at a time; training accuracy can only fall, while test
          accuracy often rises first.
        </p>
        <p data-testid="best-alpha">
          Best test accuracy {pct(best.test)} at α = {fix(best.alpha, 4)} ({best.leaves} leaves).
        </p>
      </div>
    </div>
  );
}

function Importance({ ctx }: { ctx: ClassifierContext }) {
  const tree = treeOf(ctx.run.model);
  if (!tree) return null;
  return (
    <ul className="space-y-2">
      {ctx.run.data.all.names.map((n, j) => (
        <li key={n} className="flex items-center gap-3 font-mono text-[0.8rem]">
          <span className="w-8">{n}</span>
          <span className="h-2 flex-1 overflow-hidden rounded-full bg-raised">
            <span
              className="block h-full rounded-full bg-brand"
              style={{ width: `${(tree.importances[j] ?? 0) * 100}%` }}
            />
          </span>
          <span className="w-12 text-right tabular-nums">{fix(tree.importances[j])}</span>
        </li>
      ))}
    </ul>
  );
}

export default function DecisionTreePage() {
  return (
    <ClassifierLab
      spec={treeSpec}
      text={{
        eyebrow: 'Classic · supervised',
        title: 'Decision trees',
        accent: 'twenty questions on your data',
        blurb:
          'A tree asks one yes/no question at a time and cuts the plane into rectangles. Watch it grow, overfit, and get pruned back.',
      }}
      overlay={splits}
      extras={[
        { id: 'lab-tree', title: 'The tree', render: (ctx) => <TreeView ctx={ctx} /> },
        {
          id: 'lab-growth',
          title: 'Growing it level by level',
          render: (ctx) => <Growth ctx={ctx} />,
        },
        { id: 'lab-pruning', title: 'Pruning', render: (ctx) => <Pruning ctx={ctx} /> },
        {
          id: 'lab-importance',
          title: 'Which features matter?',
          render: (ctx) => <Importance ctx={ctx} />,
        },
      ]}
      howItWorks={
        <div className="space-y-3 text-sm text-muted">
          <p>
            At each node the tree tries every feature and every threshold between neighbouring
            values and keeps the split that makes the two halves purest: the largest drop in Gini
            impurity (1 − Σp²) or entropy (−Σp log p). It then repeats on each half until a stopping
            rule (maximum depth, minimum leaf size) says to stop.
          </p>
          <p>
            Left alone a tree fits the training data perfectly, which is overfitting. Limiting
            growth or pruning back with cost-complexity pruning (penalising each leaf by α) trades a
            little training accuracy for a much simpler, more stable tree.
          </p>
        </div>
      }
    />
  );
}
