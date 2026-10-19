import { IMAGE_SIZE } from '@sim/deep';
import { Crosshair } from 'lucide-react';
import { useEffect, useMemo, useRef, useState, type PointerEvent } from 'react';
import { Button, Select, Slider, Switch, cn, useTheme } from '@/design-system';
import { mix, usePalette } from '@/shared/canvas/colors';
import { PixelCanvas } from '@/shared/components/PixelCanvas';
import { digitColor, fromPixel, gridCodes, plotExtent, toPixel } from '../latent';
import { useAeStore } from '../store';

const SIZE = 440;
const GRID = 9;
const PIXELS = IMAGE_SIZE * IMAGE_SIZE;

/** Every test digit placed by its code, with a probe whose decoded picture is shown live. */
export function LatentMap() {
  const palette = usePalette();
  const dark = useTheme().resolved === 'dark';
  const s = useAeStore();
  const { scatter, config, probe, code, probeImage } = s;
  const latent = config.latent;
  const vae = config.mode === 'vae';

  const [dimX, setDimX] = useState(0);
  const [dimY, setDimY] = useState(1);
  const [showMap, setShowMap] = useState(true);
  const [highlight, setHighlight] = useState<number | null>(null);
  const pressed = useRef(false);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  // after a rebuild the chosen dimensions may no longer exist
  const dx = Math.min(dimX, latent - 1);
  const dy = Math.min(dimY === dx ? (dx === 0 ? 1 : 0) : dimY, latent - 1);

  const extent = useMemo(
    () => (scatter ? plotExtent(scatter.mu, scatter.count, latent, [dx, dy]) : 2),
    [scatter, latent, dx, dy],
  );

  // what the decoder paints over the plane (other dimensions held at the probe's values)
  const otherKey = Array.from(probe)
    .map((v, i) => (i === dx || i === dy ? '' : v.toFixed(2)))
    .join('|');
  const decoded = useMemo(() => {
    if (!showMap || !scatter) return null;
    const codes = gridCodes(extent, GRID, probe, dx, dy);
    return s.decodeMany(codes, GRID * GRID);
    // `probe` only matters through its other dimensions (otherKey) so dragging does not recompute
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [showMap, scatter, extent, dx, dy, otherKey, latent]);

  // the decoder's pictures as one small image (GRID × GRID tiles), rebuilt only when they change
  const background = useMemo(() => {
    if (!decoded) return null;
    const side = GRID * IMAGE_SIZE;
    const canvas = document.createElement('canvas');
    canvas.width = side;
    canvas.height = side;
    const bctx = canvas.getContext('2d');
    if (!bctx) return null;
    const img = bctx.createImageData(side, side);
    for (let g = 0; g < GRID * GRID; g++) {
      const ox = (g % GRID) * IMAGE_SIZE;
      const oy = Math.floor(g / GRID) * IMAGE_SIZE;
      for (let p = 0; p < PIXELS; p++) {
        const c = mix(palette.surface, palette.fg, decoded[g * PIXELS + p] as number);
        const at = ((oy + Math.floor(p / IMAGE_SIZE)) * side + ox + (p % IMAGE_SIZE)) * 4;
        img.data[at] = c[0];
        img.data[at + 1] = c[1];
        img.data[at + 2] = c[2];
        img.data[at + 3] = 255;
      }
    }
    bctx.putImageData(img, 0, 0);
    return canvas;
  }, [decoded, palette]);

  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext('2d');
    if (!canvas || !ctx || !scatter) return;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = SIZE * dpr;
    canvas.height = SIZE * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.fillStyle = palette.css.surface;
    ctx.fillRect(0, 0, SIZE, SIZE);

    if (background) {
      ctx.imageSmoothingEnabled = true;
      ctx.globalAlpha = 0.5;
      ctx.drawImage(background, 0, 0, SIZE, SIZE);
      ctx.globalAlpha = 1;
    }

    // axes through the origin
    ctx.strokeStyle = palette.line;
    ctx.globalAlpha = 0.5;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(toPixel(0, extent, SIZE), 0);
    ctx.lineTo(toPixel(0, extent, SIZE), SIZE);
    ctx.moveTo(0, SIZE - toPixel(0, extent, SIZE));
    ctx.lineTo(SIZE, SIZE - toPixel(0, extent, SIZE));
    ctx.stroke();
    ctx.globalAlpha = 1;

    if (vae) {
      // the prior the VAE is pushed towards: one and two standard deviations
      ctx.strokeStyle = palette.css.brand;
      ctx.setLineDash([5, 5]);
      ctx.lineWidth = 1.2;
      for (const r of [1, 2]) {
        ctx.beginPath();
        ctx.arc(
          toPixel(0, extent, SIZE),
          SIZE - toPixel(0, extent, SIZE),
          (r / extent) * (SIZE / 2),
          0,
          Math.PI * 2,
        );
        ctx.stroke();
      }
      ctx.setLineDash([]);
    }

    for (let i = 0; i < scatter.count; i++) {
      const digit = scatter.labels[i] as number;
      const x = toPixel(scatter.mu[i * latent + dx] as number, extent, SIZE);
      const y = SIZE - toPixel(scatter.mu[i * latent + dy] as number, extent, SIZE);
      ctx.globalAlpha = highlight === null || highlight === digit ? 0.95 : 0.12;
      ctx.fillStyle = digitColor(digit, dark);
      ctx.beginPath();
      ctx.arc(x, y, highlight === digit ? 4.4 : 3.4, 0, Math.PI * 2);
      ctx.fill();
      ctx.lineWidth = 0.8;
      ctx.strokeStyle = palette.css.surface;
      ctx.stroke();
    }
    ctx.globalAlpha = 1;

    // the code of the picture from section 01
    if (code) {
      const x = toPixel(code.mu[dx] as number, extent, SIZE);
      const y = SIZE - toPixel(code.mu[dy] as number, extent, SIZE);
      ctx.fillStyle = palette.css.fg;
      ctx.strokeStyle = palette.css.surface;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(x, y - 8);
      ctx.lineTo(x + 8, y);
      ctx.lineTo(x, y + 8);
      ctx.lineTo(x - 8, y);
      ctx.closePath();
      ctx.fill();
      ctx.stroke();
    }

    // the probe
    const px = toPixel(probe[dx] as number, extent, SIZE);
    const py = SIZE - toPixel(probe[dy] as number, extent, SIZE);
    ctx.strokeStyle = palette.css.brand;
    ctx.lineWidth = 2.5;
    ctx.shadowColor = palette.css.brand;
    ctx.shadowBlur = 10;
    ctx.beginPath();
    ctx.arc(px, py, 9, 0, Math.PI * 2);
    ctx.moveTo(px - 15, py);
    ctx.lineTo(px - 5, py);
    ctx.moveTo(px + 5, py);
    ctx.lineTo(px + 15, py);
    ctx.moveTo(px, py - 15);
    ctx.lineTo(px, py - 5);
    ctx.moveTo(px, py + 5);
    ctx.lineTo(px, py + 15);
    ctx.stroke();
    ctx.shadowBlur = 0;
  }, [scatter, background, extent, dx, dy, probe, code, palette, dark, vae, highlight, latent]);

  const place = (event: PointerEvent<HTMLCanvasElement>) => {
    const rect = event.currentTarget.getBoundingClientRect();
    const x = fromPixel(((event.clientX - rect.left) / rect.width) * SIZE, extent, SIZE);
    const y = -fromPixel(((event.clientY - rect.top) / rect.height) * SIZE, extent, SIZE);
    s.setProbePlane(dx, dy, x, y);
  };

  const dimOptions = Array.from({ length: latent }, (_, i) => ({
    value: String(i),
    label: `z${i + 1}`,
  }));

  return (
    <div className="space-y-6">
      <p className="max-w-3xl text-[0.9rem] text-muted">
        Every dot is a test digit placed by its code, coloured by the digit it really is. Nobody
        told the network about digits, yet similar ones end up close together. The faint pictures
        behind are what the decoder paints at each spot. Click or drag on the map to move the probe
        and watch a new digit appear.
      </p>

      <div className="grid gap-8 lg:grid-cols-[auto_minmax(0,1fr)]">
        <div className="space-y-3">
          <canvas
            ref={canvasRef}
            role="img"
            aria-label="Map of the latent space: every test digit placed by its code. Click or drag to move the probe."
            style={{ width: SIZE, height: 'auto', aspectRatio: '1', touchAction: 'none' }}
            className="block max-w-full cursor-crosshair rounded-xl border border-line-strong"
            onPointerDown={(e) => {
              pressed.current = true;
              e.currentTarget.setPointerCapture(e.pointerId);
              place(e);
            }}
            onPointerMove={(e) => pressed.current && place(e)}
            onPointerUp={() => (pressed.current = false)}
            onPointerCancel={() => (pressed.current = false)}
          />
          <div className="flex flex-wrap gap-1.5" role="group" aria-label="Highlight a digit">
            {Array.from({ length: 10 }, (_, d) => (
              <button
                key={d}
                type="button"
                aria-label={`Highlight digit ${d}`}
                aria-pressed={highlight === d}
                onClick={() => setHighlight(highlight === d ? null : d)}
                className={cn(
                  'flex size-7 cursor-pointer items-center justify-center rounded-md border font-mono text-xs transition-opacity',
                  highlight !== null && highlight !== d && 'opacity-40',
                )}
                style={{ borderColor: digitColor(d, dark), color: digitColor(d, dark) }}
              >
                {d}
              </button>
            ))}
          </div>
        </div>

        <div className="min-w-0 space-y-6">
          <div className="flex flex-wrap items-start gap-6">
            <figure className="space-y-2">
              <PixelCanvas
                values={probeImage ?? new Float64Array(PIXELS)}
                side={IMAGE_SIZE}
                size={176}
                ariaLabel="Picture decoded from the probe position"
              />
              <figcaption className="eyebrow">Decoded probe</figcaption>
            </figure>
            <div className="min-w-0 flex-1 space-y-3">
              <Switch
                label="Show what the decoder paints"
                checked={showMap}
                onCheckedChange={setShowMap}
              />
              <Button size="sm" variant="secondary" onClick={s.probeToImage} disabled={!code}>
                <Crosshair className="size-4" /> Move probe to my digit
              </Button>
              <p className="text-[0.74rem] leading-snug text-faint">
                The ◆ marks the code of the digit from section 01.
                {vae &&
                  ' Dashed circles show 1 and 2 standard deviations of the prior N(0, 1) that the VAE is pulled towards.'}
              </p>
            </div>
          </div>

          {latent > 2 && (
            <div className="grid max-w-md grid-cols-2 gap-4">
              <Select
                label="Horizontal axis"
                value={String(dx)}
                onValueChange={(v) => setDimX(Number(v))}
                options={dimOptions}
              />
              <Select
                label="Vertical axis"
                value={String(dy)}
                onValueChange={(v) => setDimY(Number(v))}
                options={dimOptions}
              />
            </div>
          )}

          <div className="grid gap-x-8 gap-y-4 sm:grid-cols-2">
            {Array.from({ length: latent }, (_, i) => (
              <Slider
                key={i}
                label={`Probe z${i + 1}${i === dx ? ' (horizontal)' : i === dy ? ' (vertical)' : ''}`}
                min={-extent}
                max={extent}
                step={0.05}
                value={Math.max(-extent, Math.min(extent, probe[i] ?? 0))}
                onValueChange={(v) => s.setProbe(i, v)}
                format={(v) => v.toFixed(2)}
              />
            ))}
          </div>
          {latent > 2 && (
            <p className="text-[0.74rem] text-faint">
              With more than two numbers in the code, the map shows two of them; the other sliders
              set the rest.
            </p>
          )}
        </div>
      </div>
    </div>
  );
}
