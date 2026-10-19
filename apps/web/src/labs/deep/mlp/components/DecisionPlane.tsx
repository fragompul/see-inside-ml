import { type PointEdit, type SceneData, type Snapshot } from '@sim/deep';
import { useEffect, useRef, useState, type PointerEvent } from 'react';
import { cn } from '@/design-system';
import {
  divergingColor,
  paintClassification,
  paintRegression,
  sampleField,
} from '../canvas/heatmap';
import { mix, usePalette, type Palette, type Rgb } from '@/shared/canvas/colors';

interface Props {
  scene: SceneData;
  snapshot: Snapshot;
  showTest: boolean;
  className?: string;
  /** The sample currently under the step inspector (plane coordinates). */
  selected?: { x: number; y: number } | null;
  /** When set, clicking the plane picks the nearest training sample. */
  onPick?: (x: number, y: number) => void;
  /** Hand-editing of the training points (classification). */
  editTool?: EditTool;
  /** Class given to a point added with the `add` tool. */
  activeClass?: number;
  onEdit?: (edits: PointEdit[]) => void;
}

export type EditTool = 'look' | 'add' | 'move' | 'erase';

/** Pointer distance (px) within which a click grabs a training point. */
const GRAB = 14;

interface Hover {
  x: number;
  y: number;
  /** pixel position inside the canvas, for the tooltip */
  px: number;
  py: number;
}

const rgba = (c: Rgb, a: number) => `rgba(${c[0]}, ${c[1]}, ${c[2]}, ${a})`;

function targetColor(scene: SceneData, palette: Palette, value: number): Rgb {
  if (scene.problem === 'classification') {
    return palette.classes[value % palette.classes.length] as Rgb;
  }
  const maxAbs = Math.max(Math.abs(scene.targetRange[0]), Math.abs(scene.targetRange[1]), 1e-6);
  return divergingColor(value, maxAbs, palette, palette.surface);
}

/**
 * The model's output over the whole input plane, with the training (filled) and test (ringed)
 * points on top. The surface is painted at grid resolution and scaled with smoothing; the
 * overlay (crosshair) lives on a second canvas so hovering never repaints the heavy layer.
 */
export function DecisionPlane({
  scene,
  snapshot,
  showTest,
  className,
  selected,
  onPick,
  editTool = 'look',
  activeClass = 0,
  onEdit,
}: Props) {
  const palette = usePalette();
  const wrapRef = useRef<HTMLDivElement>(null);
  const baseRef = useRef<HTMLCanvasElement>(null);
  const overlayRef = useRef<HTMLCanvasElement>(null);
  const surfaceRef = useRef<HTMLCanvasElement | null>(null);
  const [size, setSize] = useState(0);
  const [hover, setHover] = useState<Hover | null>(null);
  const dragging = useRef<number | null>(null);

  useEffect(() => {
    const el = wrapRef.current;
    if (!el) return;
    const observer = new ResizeObserver(([entry]) => {
      if (entry) setSize(Math.floor(entry.contentRect.width));
    });
    observer.observe(el);
    return () => observer.disconnect();
  }, []);

  // Heavy layer: surface + grid + points.
  useEffect(() => {
    const canvas = baseRef.current;
    const ctx = canvas?.getContext('2d');
    if (!canvas || !ctx || size === 0) return;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = size * dpr;
    canvas.height = size * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, size, size);

    ctx.fillStyle = palette.css.surface;
    ctx.fillRect(0, 0, size, size);

    if (snapshot.boundary) {
      surfaceRef.current =
        scene.problem === 'classification'
          ? paintClassification(surfaceRef.current, snapshot.boundary, palette)
          : paintRegression(surfaceRef.current, snapshot.boundary, palette, scene.targetRange);
      ctx.imageSmoothingEnabled = true;
      ctx.imageSmoothingQuality = 'high';
      ctx.drawImage(surfaceRef.current, 0, 0, size, size);
    }

    // Axes: a faint grid every 2 units with the origin emphasised.
    const toPx = (v: number) => ((v + scene.domain) / (2 * scene.domain)) * size;
    ctx.lineWidth = 1;
    for (let v = -scene.domain; v <= scene.domain; v += 2) {
      ctx.strokeStyle = v === 0 ? rgba(palette.fg, 0.28) : rgba(palette.fg, 0.07);
      const p = Math.round(toPx(v)) + 0.5;
      ctx.beginPath();
      ctx.moveTo(p, 0);
      ctx.lineTo(p, size);
      ctx.moveTo(0, p);
      ctx.lineTo(size, p);
      ctx.stroke();
    }

    const drawPoints = (xy: Float32Array, target: Float32Array, test: boolean) => {
      for (let i = 0; i < target.length; i++) {
        const x = toPx(xy[2 * i] as number);
        const y = size - toPx(xy[2 * i + 1] as number);
        const color = targetColor(scene, palette, target[i] as number);
        ctx.beginPath();
        ctx.arc(x, y, test ? 3.6 : 3.4, 0, Math.PI * 2);
        if (test) {
          ctx.fillStyle = rgba(palette.surface, 0.9);
          ctx.fill();
          ctx.lineWidth = 1.8;
          ctx.strokeStyle = rgba(color, 1);
          ctx.stroke();
        } else {
          ctx.fillStyle = rgba(color, 1);
          ctx.fill();
          ctx.lineWidth = 1;
          ctx.strokeStyle = rgba(mix(palette.surface, palette.fg, 0.55), 0.85);
          ctx.stroke();
        }
      }
    };
    drawPoints(scene.train.xy, scene.train.target, false);
    if (showTest) drawPoints(scene.test.xy, scene.test.target, true);
  }, [scene, snapshot, palette, size, showTest]);

  // Light layer: crosshair.
  useEffect(() => {
    const canvas = overlayRef.current;
    const ctx = canvas?.getContext('2d');
    if (!canvas || !ctx || size === 0) return;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = size * dpr;
    canvas.height = size * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, size, size);
    if (!hover) return;
    ctx.strokeStyle = rgba(palette.fg, 0.45);
    ctx.setLineDash([4, 4]);
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(hover.px + 0.5, 0);
    ctx.lineTo(hover.px + 0.5, size);
    ctx.moveTo(0, hover.py + 0.5);
    ctx.lineTo(size, hover.py + 0.5);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.beginPath();
    ctx.arc(hover.px, hover.py, 5, 0, Math.PI * 2);
    ctx.strokeStyle = rgba(palette.fg, 0.9);
    ctx.lineWidth = 1.5;
    ctx.stroke();
  }, [hover, palette, size]);

  const plane = (event: PointerEvent<HTMLDivElement>) => {
    const rect = event.currentTarget.getBoundingClientRect();
    const px = event.clientX - rect.left;
    const py = event.clientY - rect.top;
    return {
      rect,
      px,
      py,
      x: (px / rect.width) * 2 * scene.domain - scene.domain,
      y: scene.domain - (py / rect.height) * 2 * scene.domain,
    };
  };

  /** Index of the training point under the pointer, or null. */
  const grab = (px: number, py: number, rect: DOMRect): number | null => {
    let best: number | null = null;
    let bestD = GRAB;
    const { xy } = scene.train;
    for (let i = 0; i < scene.train.target.length; i++) {
      const sx = (((xy[2 * i] as number) + scene.domain) / (2 * scene.domain)) * rect.width;
      const sy =
        rect.height -
        (((xy[2 * i + 1] as number) + scene.domain) / (2 * scene.domain)) * rect.height;
      const d = Math.hypot(sx - px, sy - py);
      if (d < bestD) {
        bestD = d;
        best = i;
      }
    }
    return best;
  };

  const onMove = (event: PointerEvent<HTMLDivElement>) => {
    const { px, py, x, y } = plane(event);
    setHover({ x, y, px, py });
    if (dragging.current !== null && onEdit)
      onEdit([{ op: 'move', index: dragging.current, x, y }]);
  };

  const onDown = (event: PointerEvent<HTMLDivElement>) => {
    if (!onEdit || editTool === 'look') return;
    const { rect, px, py, x, y } = plane(event);
    if (editTool === 'add') onEdit([{ op: 'add', x, y, label: activeClass }]);
    else {
      const hit = grab(px, py, rect);
      if (hit === null) return;
      if (editTool === 'erase') onEdit([{ op: 'erase', index: hit }]);
      else {
        dragging.current = hit;
        event.currentTarget.setPointerCapture(event.pointerId);
      }
    }
  };

  const release = () => {
    dragging.current = null;
  };

  const readout =
    hover && snapshot.boundary
      ? sampleField(snapshot.boundary, hover.x, hover.y, scene.domain)
      : null;
  const flipTooltip = hover ? hover.px > size * 0.55 : false;

  return (
    <div
      ref={wrapRef}
      data-testid="decision-plane"
      className={cn(
        'relative aspect-square w-full touch-none overflow-hidden rounded-xl border border-line-strong bg-surface',
        (onPick || editTool === 'add') && 'cursor-crosshair',
        editTool === 'move' && 'cursor-grab',
        editTool === 'erase' && 'cursor-pointer',
        className,
      )}
      onPointerMove={onMove}
      onPointerDown={onDown}
      onPointerUp={release}
      onPointerCancel={release}
      onPointerLeave={() => setHover(null)}
      onClick={(event) => {
        if (!onPick) return;
        const rect = event.currentTarget.getBoundingClientRect();
        onPick(
          ((event.clientX - rect.left) / rect.width) * 2 * scene.domain - scene.domain,
          scene.domain - ((event.clientY - rect.top) / rect.height) * 2 * scene.domain,
        );
      }}
    >
      <canvas ref={baseRef} className="absolute inset-0 size-full" />
      <canvas ref={overlayRef} className="pointer-events-none absolute inset-0 size-full" />

      {selected && size > 0 && (
        <span
          aria-hidden
          className="pointer-events-none absolute z-[5] size-6 -translate-x-1/2 -translate-y-1/2"
          style={{
            left: ((selected.x + scene.domain) / (2 * scene.domain)) * size,
            top: size - ((selected.y + scene.domain) / (2 * scene.domain)) * size,
          }}
        >
          <span className="absolute inset-0 animate-ping rounded-full border-2 border-brand opacity-70" />
          <span className="absolute inset-1 rounded-full border-2 border-brand bg-brand/20" />
        </span>
      )}

      {hover && readout && (
        <div
          className="pointer-events-none absolute z-10 min-w-36 rounded-lg border border-line-strong bg-raised/95 px-3 py-2 font-mono text-[0.7rem] shadow-[var(--shadow-card)] backdrop-blur"
          style={{
            top: Math.min(Math.max(hover.py - 12, 6), Math.max(size - 96, 6)),
            ...(flipTooltip ? { right: size - hover.px + 14 } : { left: hover.px + 14 }),
          }}
        >
          <p className="text-muted">
            x₁ {hover.x.toFixed(2)} · x₂ {hover.y.toFixed(2)}
          </p>
          {scene.problem === 'classification' ? (
            <ul className="mt-1.5 space-y-1">
              {readout.map((p, k) => (
                <li key={k} className="flex items-center gap-2">
                  <span
                    className="size-2 rounded-full"
                    style={{ background: palette.css.classes[k % palette.css.classes.length] }}
                  />
                  <span className="text-muted">class {k}</span>
                  <span className="ml-auto text-fg tabular-nums">{(p * 100).toFixed(1)}%</span>
                </li>
              ))}
            </ul>
          ) : (
            <p className="mt-1.5 text-fg">ŷ = {(readout[0] as number).toFixed(3)}</p>
          )}
        </div>
      )}
    </div>
  );
}
