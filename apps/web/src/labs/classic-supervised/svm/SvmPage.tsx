import { SVC, type KernelName } from '@sim/classic';
import { gridPoints } from '@sim/core';
import { useMemo } from 'react';
import { usePalette } from '@/shared/canvas/colors';
import { MiniChart } from '@/shared/components/MiniChart';
import {
  ClassifierLab,
  drawContour,
  num,
  pct,
  str,
  type ClassifierContext,
  type Painter,
} from '@/shared/lab-kit';
import { C_VALUES, GAMMA_VALUES, svcOf, svmSpec } from './svmSpec';

const GRID = 60;

/** The decision boundary f = 0 and the margins f = ±1 (two-class models only). */
const margins = (ctx: ClassifierContext) => {
  const svc = svcOf(ctx.run.model);
  if (!svc || svc.classes !== 2) return undefined;
  const b = ctx.regions.bounds;
  const f = svc.decision(gridPoints(b.x0, b.x1, b.y0, b.y1, GRID));
  return (p: Painter) => {
    drawContour(p, f, GRID, b, 1, { color: p.palette.css.fg, width: 1.2, dash: [5, 4] });
    drawContour(p, f, GRID, b, -1, { color: p.palette.css.fg, width: 1.2, dash: [5, 4] });
    drawContour(p, f, GRID, b, 0, { color: p.palette.css.fg, width: 2.2 });
  };
};

/** Ring the support vectors: the only points the model actually remembers. */
const supportVectors = (ctx: ClassifierContext): ReadonlySet<number> | undefined => {
  const svc = svcOf(ctx.run.model);
  if (!svc) return undefined;
  const trainToAll: number[] = [];
  ctx.run.data.testMask.forEach((t, i) => {
    if (t === 0) trainToAll.push(i);
  });
  return new Set(svc.supportIndices.map((i) => trainToAll[i] as number));
};

function Sweep({ ctx, over }: { ctx: ClassifierContext; over: 'C' | 'gamma' }) {
  const palette = usePalette();
  const { train, test } = ctx.run.data;
  const kernel = str(ctx.params, 'kernel') as KernelName;
  const values: readonly number[] = over === 'C' ? C_VALUES : GAMMA_VALUES;
  const rows = useMemo(
    () =>
      values.map((v) => {
        const m = new SVC({
          kernel,
          C: over === 'C' ? v : num(ctx.params, 'C'),
          gamma: over === 'gamma' ? v : num(ctx.params, 'gamma'),
          degree: num(ctx.params, 'degree'),
          coef0: 1,
          tol: 1e-3,
        }).fit(train.x, train.y);
        const acc = (t: typeof train) =>
          t.x.rows
            ? Array.from(m.predict(t.x)).filter((p, i) => p === t.y[i]).length / t.x.rows
            : NaN;
        return { v, train: acc(train), test: acc(test), sv: m.supportIndices.length };
      }),
    [values, over, kernel, ctx.params, train, test],
  );
  const best = rows.reduce((b, r) => (r.test > b.test ? r : b), rows[0]!);
  return (
    <div className="grid gap-8 lg:grid-cols-[minmax(0,1fr)_18rem]">
      <MiniChart
        title={`Accuracy as ${over === 'C' ? 'C' : 'γ'} grows`}
        x={rows.map((r) => Math.log10(r.v))}
        xLabel={`log10 ${over === 'C' ? 'C' : 'γ'}`}
        yRange={{ lo: 0.4, hi: 1 }}
        series={[
          { name: 'training', color: palette.css.brand, values: rows.map((r) => r.train) },
          { name: 'test', color: palette.css.pos, values: rows.map((r) => r.test) },
        ]}
        height={190}
      />
      <div className="space-y-3 text-sm text-muted">
        <p>
          {over === 'C'
            ? 'Small C tolerates margin violations (smooth, high bias); large C punishes them (tight, high variance).'
            : 'Small γ makes each kernel wide and the boundary smooth; large γ makes it spiky and memorises the points.'}
        </p>
        <p data-testid={`best-${over}`}>
          Best test accuracy {pct(best.test)} at {over === 'C' ? 'C' : 'γ'} = {best.v} ({best.sv}{' '}
          support vectors).
        </p>
      </div>
    </div>
  );
}

function Margin({ ctx }: { ctx: ClassifierContext }) {
  const svc = svcOf(ctx.run.model);
  if (!svc) return null;
  const n = ctx.run.data.train.x.rows;
  const kernel = str(ctx.params, 'kernel');
  const bin = svc.classes === 2 ? svc.binaryModel() : null;
  let width: string | null = null;
  if (bin && kernel === 'linear') {
    // ‖w‖² = Σ αᵢαⱼ yᵢyⱼ xᵢ·xⱼ for the linear kernel: the margin is 2/‖w‖
    const x = ctx.run.data.train.x;
    let w2 = 0;
    for (let a = 0; a < bin.support.length; a++)
      for (let b = 0; b < bin.support.length; b++) {
        let dot = 0;
        for (let j = 0; j < x.cols; j++)
          dot += x.get(bin.support[a] as number, j) * x.get(bin.support[b] as number, j);
        w2 += (bin.coef[a] as number) * (bin.coef[b] as number) * dot;
      }
    width = (2 / Math.sqrt(w2)).toFixed(3);
  }
  return (
    <div className="space-y-2 text-sm text-muted" data-testid="svm-summary">
      <p>
        <strong className="text-fg">{svc.supportIndices.length}</strong> of {n} training points are
        support vectors (ringed on the plot): the points on or inside the margin. Delete any other
        point and the boundary does not move.
      </p>
      {width && (
        <p>
          For this linear model the margin (between the dashed lines) is{' '}
          <strong className="text-fg">{width}</strong> wide, 2/‖w‖.
        </p>
      )}
      <p>The solid line is the decision boundary f(x) = 0; the dashed lines are f(x) = ±1.</p>
    </div>
  );
}

export default function SvmPage() {
  return (
    <ClassifierLab
      spec={svmSpec}
      text={{
        eyebrow: 'Classic · supervised',
        title: 'Support vector machines',
        accent: 'the widest street between classes',
        blurb:
          'Find the boundary with the widest margin, and let a kernel bend it. Only the support vectors on the margin matter.',
      }}
      overlay={margins}
      highlight={supportVectors}
      extras={[
        {
          id: 'lab-margin',
          title: 'Margin and support vectors',
          render: (ctx) => <Margin ctx={ctx} />,
        },
        { id: 'lab-c', title: 'The penalty C', render: (ctx) => <Sweep ctx={ctx} over="C" /> },
        {
          id: 'lab-gamma',
          title: 'The kernel width γ',
          render: (ctx) => <Sweep ctx={ctx} over="gamma" />,
        },
      ]}
      howItWorks={
        <div className="space-y-3 text-sm text-muted">
          <p>
            A linear SVM chooses the hyperplane that separates the classes with the largest margin.
            When the classes overlap, the penalty C sets the price of a point on the wrong side of
            its margin. The solution depends only on the support vectors, the points on or inside
            the margin; it is found by sequential minimal optimisation (SMO) on the dual problem.
          </p>
          <p>
            The kernel trick replaces every dot product xᵢ·xⱼ by k(xᵢ, xⱼ), which is a dot product
            in a much larger space the model never builds. The RBF kernel exp(−γ‖x − z‖²) gives
            curved boundaries; γ controls how local they are.
          </p>
        </div>
      }
    />
  );
}
