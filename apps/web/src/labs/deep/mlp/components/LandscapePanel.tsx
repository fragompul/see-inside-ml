import { useEffect, useMemo, useRef, useState, type PointerEvent } from 'react';
import { RefreshCw } from 'lucide-react';
import { Button, Segmented } from '@/design-system';
import { mix, usePalette } from '@/shared/canvas/colors';
import { contourSegments } from '@/shared/lab-kit';
import { useMlpStore } from '../store';

const SIZE = 420;
const STEPS = 29;

type Kind = 'path' | 'random';

/** The loss as a map over a plane through weight space, with the training path drawn on it. */
export function LandscapePanel() {
  const palette = usePalette();
  const analysis = useMlpStore((s) => s.analysis);
  const running = useMlpStore((s) => s.running);
  const sceneVersion = useMlpStore((s) => s.sceneVersion);
  const epochs = useMlpStore((s) => s.history.length);
  const request = useMlpStore((s) => s.requestAnalysis);
  const [kind, setKind] = useState<Kind>('path');
  const [hover, setHover] = useState<{ a: number; b: number; loss: number } | null>(null);
  const ref = useRef<HTMLCanvasElement>(null);
  const land = analysis?.landscape ?? null;

  // ask for a fresh slice: when the lab opens, when the model changes, when training pauses
  // and (throttled) while it runs
  useEffect(() => {
    request({ landscape: { steps: STEPS, kind } });
  }, [request, kind, sceneVersion, running, epochs === 0]); // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => {
    if (!running) return;
    const id = window.setInterval(() => request({ landscape: { steps: STEPS, kind } }), 1500);
    return () => window.clearInterval(id);
  }, [running, kind, request]);

  const grid = useMemo(() => {
    if (!land) return null;
    const logs = Float64Array.from(land.values, (v) => Math.log(Math.max(v, 1e-9)));
    let lo = Infinity;
    let hi = -Infinity;
    for (const v of logs) {
      lo = Math.min(lo, v);
      hi = Math.max(hi, v);
    }
    // clip the extreme corners so the valley stays readable
    const sorted = [...logs].sort((a, b) => a - b);
    const top = sorted[Math.floor(sorted.length * 0.97)] as number;
    return { logs, lo, hi: Math.max(lo + 1e-6, Math.min(hi, top)) };
  }, [land]);

  useEffect(() => {
    const canvas = ref.current;
    const ctx = canvas?.getContext('2d');
    if (!canvas || !ctx || !land || !grid) return;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = SIZE * dpr;
    canvas.height = SIZE * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const n = land.steps;
    const cell = SIZE / n;
    // paint one pixel per grid cell on a small canvas, then enlarge it with smoothing
    const small = document.createElement('canvas');
    small.width = n;
    small.height = n;
    const sctx = small.getContext('2d');
    if (!sctx) return;
    const img = sctx.createImageData(n, n);
    for (let r = 0; r < n; r++)
      for (let c = 0; c < n; c++) {
        const t = Math.min(
          1,
          Math.max(0, ((grid.logs[r * n + c] as number) - grid.lo) / (grid.hi - grid.lo)),
        );
        const col =
          t < 0.5
            ? mix(palette.neg, palette.brand, t * 2)
            : mix(palette.brand, palette.pos, (t - 0.5) * 2);
        const soft = mix(palette.surface, col, 0.55 + 0.4 * t);
        const at = ((n - 1 - r) * n + c) * 4; // rows grow upwards on screen
        img.data[at] = soft[0];
        img.data[at + 1] = soft[1];
        img.data[at + 2] = soft[2];
        img.data[at + 3] = 255;
      }
    sctx.putImageData(img, 0, 0);
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = 'high';
    ctx.drawImage(small, 0, 0, SIZE, SIZE);
    // contour lines at a handful of loss levels
    const toPx = (v: number) => v * cell;
    ctx.save();
    ctx.lineWidth = 1;
    ctx.strokeStyle = palette.css.fg;
    ctx.globalAlpha = 0.28;
    for (const q of [0.1, 0.25, 0.4, 0.55, 0.7, 0.85]) {
      const level = grid.lo + q * (grid.hi - grid.lo);
      ctx.beginPath();
      for (const [x0, y0, x1, y1] of contourSegments(grid.logs, n, level)) {
        // marching squares works in cell units with row 0 on top; our rows grow upwards
        ctx.moveTo(toPx(x0), SIZE - toPx(y0));
        ctx.lineTo(toPx(x1), SIZE - toPx(y1));
      }
      ctx.stroke();
    }
    ctx.restore();
    // training path
    if (land.path && land.path.a.length > 1) {
      const px = (a: number) => ((a + land.radius) / (2 * land.radius)) * SIZE;
      const py = (b: number) => SIZE - ((b + land.radius) / (2 * land.radius)) * SIZE;
      ctx.lineWidth = 2;
      ctx.strokeStyle = palette.css.fg;
      ctx.beginPath();
      land.path.a.forEach((a, i) => {
        const x = px(a);
        const y = py(land.path!.b[i] as number);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.stroke();
      const first = [px(land.path.a[0] as number), py(land.path.b[0] as number)] as const;
      ctx.fillStyle = palette.css.surface;
      ctx.lineWidth = 2;
      ctx.strokeStyle = palette.css.fg;
      ctx.beginPath();
      ctx.arc(first[0], first[1], 5.5, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
    }
    // the current weights are the centre of the plane
    ctx.fillStyle = palette.css.brand;
    ctx.strokeStyle = palette.css.fg;
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(SIZE / 2, SIZE / 2, 6.5, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
  }, [land, grid, palette]);

  const onMove = (e: PointerEvent<HTMLCanvasElement>) => {
    if (!land) return;
    const rect = e.currentTarget.getBoundingClientRect();
    const fx = (e.clientX - rect.left) / rect.width;
    const fy = (e.clientY - rect.top) / rect.height;
    const c = Math.min(land.steps - 1, Math.max(0, Math.floor(fx * land.steps)));
    const r = Math.min(land.steps - 1, Math.max(0, Math.floor((1 - fy) * land.steps)));
    setHover({
      a: land.axis[c] as number,
      b: land.axis[r] as number,
      loss: land.values[r * land.steps + c] as number,
    });
  };

  return (
    <div className="grid gap-8 lg:grid-cols-[auto_minmax(0,1fr)]">
      <div className="space-y-3">
        <canvas
          ref={ref}
          role="img"
          aria-label="Map of the training loss over a plane through weight space. Blue is low loss, red is high. The line is the path training took; the coloured dot is where the weights are now and the outlined dot is where they started."
          style={{ width: SIZE, height: 'auto', aspectRatio: '1 / 1' }}
          className="block max-w-full rounded-xl border border-line-strong"
          onPointerMove={onMove}
          onPointerLeave={() => setHover(null)}
        />
        <p
          className="font-mono text-[0.72rem] text-faint tabular-nums"
          data-testid="landscape-readout"
        >
          {hover
            ? `loss ${hover.loss.toFixed(3)} at (${hover.a.toFixed(2)}, ${hover.b.toFixed(2)})`
            : land
              ? `loss now ${land.centreLoss.toFixed(3)} · hover the map`
              : 'computing…'}
        </p>
      </div>
      <div className="min-w-0 space-y-5">
        <div className="flex flex-wrap items-center gap-3">
          <Segmented<Kind>
            label="Plane"
            size="sm"
            fit
            value={kind}
            onValueChange={setKind}
            options={[
              { value: 'path', label: 'Along the training path' },
              { value: 'random', label: 'Random directions' },
            ]}
          />
          <Button
            size="sm"
            variant="secondary"
            onClick={() => request({ landscape: { steps: STEPS, kind } })}
          >
            <RefreshCw className="size-4" /> Refresh
          </Button>
        </div>
        <div className="space-y-3 text-sm text-muted">
          <p>
            The network has many weights, so its loss lives in a space with that many dimensions.
            This map slices through it: the loss at the current weights (coloured dot) and at
            weights moved along two directions. The line is the path the weights followed
            {land?.kind === 'path'
              ? ' (the plane is spanned by the two directions it moved most along)'
              : ''}
            ; its start is the outlined dot.
          </p>
          <p>
            {land?.kind === 'random'
              ? 'These two random directions are scaled to the size of each layer’s weights, so slopes are comparable between networks. A trained network sits in a basin: the loss rises in every direction.'
              : 'Training slides downhill, often along a curved valley. Switch to random directions for an unbiased slice, or pause and press Refresh after changing the learning rate to see how the next steps would behave.'}
          </p>
        </div>
      </div>
    </div>
  );
}
