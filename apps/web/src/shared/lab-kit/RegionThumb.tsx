import { useEffect, useRef } from 'react';
import { usePalette } from '@/shared/canvas/colors';
import type { Regions } from './pipeline';

/** A small picture of a model's decision regions (no points), for side-by-side comparisons. */
export function RegionThumb({
  regions,
  size = 120,
  label,
}: {
  regions: Regions;
  size?: number;
  label: string;
}) {
  const palette = usePalette();
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const canvas = ref.current;
    const ctx = canvas?.getContext('2d');
    if (!canvas || !ctx) return;
    canvas.width = regions.n;
    canvas.height = regions.n;
    const img = ctx.createImageData(regions.n, regions.n);
    for (let i = 0; i < regions.n * regions.n; i++) {
      const c = palette.classes[
        (regions.labels[i] as number) % palette.classes.length
      ] as readonly [number, number, number];
      const a =
        0.22 + 0.4 * Math.max(0, Math.min(1, ((regions.confidence[i] as number) - 0.3) / 0.7));
      img.data[i * 4] = c[0] * a + palette.surface[0] * (1 - a);
      img.data[i * 4 + 1] = c[1] * a + palette.surface[1] * (1 - a);
      img.data[i * 4 + 2] = c[2] * a + palette.surface[2] * (1 - a);
      img.data[i * 4 + 3] = 255;
    }
    ctx.putImageData(img, 0, 0);
  }, [regions, palette]);
  return (
    <figure className="space-y-1">
      <canvas
        ref={ref}
        role="img"
        aria-label={label}
        style={{ width: size, height: size, imageRendering: 'pixelated' }}
        className="block rounded-lg border border-line-strong"
      />
      <figcaption className="text-center font-mono text-[0.68rem] text-muted">{label}</figcaption>
    </figure>
  );
}
