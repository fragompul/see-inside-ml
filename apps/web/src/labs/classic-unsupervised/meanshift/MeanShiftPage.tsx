import { MeanShift, estimateBandwidth } from '@sim/classic';
import { plotBounds } from '@sim/core';
import { useMemo } from 'react';
import { ClusterLab, fix, num, type ClusterContext, type Painter } from '@/shared/lab-kit';
import { meanShiftOf, meanshiftSpec } from './meanshiftSpec';

const GRID = 34;

/** A faint map of the density estimate: how many points each cell's window contains. */
const densityUnderlay = (ctx: ClusterContext) => {
  const bw = num(ctx.params, 'bandwidth');
  const { x } = ctx.run.data;
  const b = plotBounds(x, 0.2);
  const counts = new Float64Array(GRID * GRID);
  let max = 0;
  for (let r = 0; r < GRID; r++)
    for (let c = 0; c < GRID; c++) {
      const cx = b.x0 + ((c + 0.5) / GRID) * (b.x1 - b.x0);
      const cy = b.y0 + ((r + 0.5) / GRID) * (b.y1 - b.y0);
      let n = 0;
      for (let i = 0; i < x.rows; i++)
        if (Math.hypot(x.get(i, 0) - cx, x.get(i, 1) - cy) <= bw) n++;
      counts[r * GRID + c] = n;
      max = Math.max(max, n);
    }
  return (p: Painter) => {
    if (max === 0) return;
    const w = (b.x1 - b.x0) / GRID;
    const h = (b.y1 - b.y0) / GRID;
    p.ctx.save();
    for (let r = 0; r < GRID; r++)
      for (let c = 0; c < GRID; c++) {
        const v = (counts[r * GRID + c] as number) / max;
        if (v < 0.02) continue;
        p.ctx.globalAlpha = 0.5 * v;
        p.ctx.fillStyle = p.palette.css.brand;
        const x0 = p.px(b.x0 + c * w);
        const y0 = p.py(b.y0 + (r + 1) * h);
        p.ctx.fillRect(x0, y0, p.px(b.x0 + (c + 1) * w) - x0 + 0.6, p.py(b.y0 + r * h) - y0 + 0.6);
      }
    p.ctx.restore();
  };
};

const climbs = (ctx: ClusterContext) => {
  const m = meanShiftOf(ctx.run.model);
  if (!m) return undefined;
  const n = m.paths.length;
  const stride = Math.max(1, Math.floor(n / 14));
  return (p: Painter) => {
    p.ctx.save();
    p.ctx.lineWidth = 1.2;
    p.ctx.strokeStyle = p.palette.css.fg;
    p.ctx.globalAlpha = 0.4;
    for (let i = 0; i < n; i += stride) {
      const path = m.paths[i] as [number, number][];
      p.ctx.beginPath();
      path.forEach(([x, y], k) => {
        if (k === 0) p.ctx.moveTo(p.px(x), p.py(y));
        else p.ctx.lineTo(p.px(x), p.py(y));
      });
      p.ctx.stroke();
    }
    p.ctx.restore();
    for (let c = 0; c < m.centers.rows; c++) {
      const color = p.palette.css.classes[c % p.palette.css.classes.length] as string;
      p.ctx.beginPath();
      p.ctx.arc(p.px(m.centers.get(c, 0)), p.py(m.centers.get(c, 1)), 8, 0, Math.PI * 2);
      p.ctx.fillStyle = color;
      p.ctx.fill();
      p.ctx.lineWidth = 2;
      p.ctx.strokeStyle = p.palette.css.fg;
      p.ctx.stroke();
    }
  };
};

function Bandwidth({ ctx }: { ctx: ClusterContext }) {
  const bw = num(ctx.params, 'bandwidth');
  const estimate = useMemo(() => estimateBandwidth(ctx.run.data.x, 0.3), [ctx.run.data]);
  const trial = useMemo(
    () => new MeanShift({ bandwidth: estimate }).fit(ctx.run.data.x).clusters,
    [ctx.run.data, estimate],
  );
  return (
    <div className="space-y-3 text-sm text-muted" data-testid="bandwidth-note">
      <p>
        The window radius is the one real parameter: it decides how smooth the density estimate is,
        and hence how many peaks there are. You are using <strong className="text-fg">{bw}</strong>.
      </p>
      <p>
        A common rule of thumb (the average distance to the nearest 30% of the points) suggests{' '}
        <strong className="text-fg">{fix(estimate, 2)}</strong> for this data, which gives{' '}
        <strong className="text-fg">{trial}</strong> cluster{trial === 1 ? '' : 's'}. It is a
        starting point, not an answer: it ignores how spread out the clusters really are.
      </p>
    </div>
  );
}

export default function MeanShiftPage() {
  return (
    <ClusterLab
      spec={meanshiftSpec}
      text={{
        eyebrow: 'Classic · unsupervised',
        title: 'Mean shift',
        accent: 'climb to the nearest peak',
        blurb:
          'Every point repeatedly moves to the average of the points around it, drifting uphill on the density until it reaches a peak. Points that end at the same peak share a cluster.',
      }}
      showRegions={false}
      underlay={densityUnderlay}
      overlay={climbs}
      extras={[
        {
          id: 'lab-bandwidth',
          title: 'Choosing the bandwidth',
          render: (ctx) => <Bandwidth ctx={ctx} />,
        },
      ]}
      howItWorks={
        <div className="space-y-3 text-sm text-muted">
          <p>
            Mean shift treats the data as samples from an unknown density and finds its peaks
            (modes). From each point it repeats one step: replace the position by the mean of all
            points within the bandwidth. That step always points uphill, so every point converges to
            a local maximum of a kernel density estimate. The faint blue wash is that density; the
            thin lines are the climbs of a few points; the dots are the peaks found.
          </p>
          <p>
            There is no k to choose and no assumed cluster shape other than “one peak per cluster”,
            but a single bandwidth must suit the whole dataset and the cost grows quickly with the
            number of points.
          </p>
        </div>
      }
    />
  );
}
