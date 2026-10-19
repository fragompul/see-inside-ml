import { useEffect, useMemo, useRef } from 'react';
import { mix, usePalette } from '@/shared/canvas/colors';
import { toPixel } from '@/shared/canvas/plane';
import { D_GRID, EXTENT, WARP_GRID } from '../store';

const SIZE = 520;

interface Props {
  real: Float64Array | null;
  fake: Float64Array | null;
  discriminator: Float64Array | null;
  warp: Float64Array | null;
  showDiscriminator: boolean;
  showWarp: boolean;
}

/** The plane where the duel happens: what D believes, the real data, and the generator's points. */
export function DuelCanvas({
  real,
  fake,
  discriminator,
  warp,
  showDiscriminator,
  showWarp,
}: Props) {
  const palette = usePalette();
  const ref = useRef<HTMLCanvasElement>(null);

  // D(x) as one small image (D_GRID²): frost where it says "fake", ember where it says "real"
  const heat = useMemo(() => {
    if (!discriminator) return null;
    const canvas = document.createElement('canvas');
    canvas.width = D_GRID;
    canvas.height = D_GRID;
    const ctx = canvas.getContext('2d');
    if (!ctx) return null;
    const img = ctx.createImageData(D_GRID, D_GRID);
    for (let i = 0; i < D_GRID * D_GRID; i++) {
      const p = discriminator[i] as number;
      const c =
        p >= 0.5
          ? mix(palette.surface, palette.pos, (p - 0.5) * 2)
          : mix(palette.surface, palette.neg, (0.5 - p) * 2);
      img.data[i * 4] = c[0];
      img.data[i * 4 + 1] = c[1];
      img.data[i * 4 + 2] = c[2];
      img.data[i * 4 + 3] = 255;
    }
    ctx.putImageData(img, 0, 0);
    return canvas;
  }, [discriminator, palette]);

  useEffect(() => {
    const canvas = ref.current;
    const ctx = canvas?.getContext('2d');
    if (!canvas || !ctx) return;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = SIZE * dpr;
    canvas.height = SIZE * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.fillStyle = palette.css.surface;
    ctx.fillRect(0, 0, SIZE, SIZE);

    if (showDiscriminator && heat) {
      ctx.imageSmoothingEnabled = true;
      ctx.globalAlpha = 0.5;
      ctx.drawImage(heat, 0, 0, SIZE, SIZE);
      ctx.globalAlpha = 1;
    }

    const X = (v: number) => toPixel(v, EXTENT, SIZE);
    const Y = (v: number) => SIZE - toPixel(v, EXTENT, SIZE);

    // axes
    ctx.strokeStyle = palette.line;
    ctx.globalAlpha = 0.45;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(X(0), 0);
    ctx.lineTo(X(0), SIZE);
    ctx.moveTo(0, Y(0));
    ctx.lineTo(SIZE, Y(0));
    ctx.stroke();
    ctx.globalAlpha = 1;

    if (showWarp && warp) {
      ctx.strokeStyle = palette.css.brand;
      ctx.globalAlpha = 0.4;
      ctx.lineWidth = 1.2;
      ctx.beginPath();
      for (let r = 0; r < WARP_GRID; r++) {
        for (let c = 0; c < WARP_GRID; c++) {
          const i = r * WARP_GRID + c;
          const x = X(warp[2 * i] as number);
          const y = Y(warp[2 * i + 1] as number);
          if (c === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        }
      }
      for (let c = 0; c < WARP_GRID; c++) {
        for (let r = 0; r < WARP_GRID; r++) {
          const i = r * WARP_GRID + c;
          const x = X(warp[2 * i] as number);
          const y = Y(warp[2 * i + 1] as number);
          if (r === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        }
      }
      ctx.stroke();
      ctx.globalAlpha = 1;
    }

    if (real) {
      ctx.strokeStyle = palette.css.fg;
      ctx.lineWidth = 1.4;
      ctx.globalAlpha = 0.75;
      for (let i = 0; i < real.length / 2; i++) {
        ctx.beginPath();
        ctx.arc(X(real[2 * i] as number), Y(real[2 * i + 1] as number), 3.2, 0, Math.PI * 2);
        ctx.stroke();
      }
      ctx.globalAlpha = 1;
    }

    if (fake) {
      ctx.fillStyle = palette.css.brand;
      ctx.strokeStyle = palette.css.surface;
      ctx.lineWidth = 0.8;
      for (let i = 0; i < fake.length / 2; i++) {
        ctx.beginPath();
        ctx.arc(X(fake[2 * i] as number), Y(fake[2 * i + 1] as number), 3.4, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();
      }
    }
  }, [real, fake, heat, warp, showDiscriminator, showWarp, palette]);

  return (
    <canvas
      ref={ref}
      role="img"
      aria-label="The plane where the generator's points (filled) try to match the real data (rings), over the discriminator's opinion"
      style={{ width: SIZE, height: 'auto', aspectRatio: '1' }}
      className="block max-w-full rounded-xl border border-line-strong"
    />
  );
}
