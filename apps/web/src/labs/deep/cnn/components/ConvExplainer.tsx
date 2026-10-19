import { createRng } from '@sim/core';
import { IMAGE_SIZE, convolveSingle, renderDigit } from '@sim/deep';
import { useMemo, useState } from 'react';
import { Select, Slider, cn } from '@/design-system';
import { inkOn, mix, usePalette } from '@/shared/canvas/colors';
import { maxAbs } from '@/shared/components/Heat';
import { KERNEL_PRESETS, patchAt, windowOrigin } from '../kernels';
import { effectiveImage, useCnnStore } from '../store';
import { PixelCanvas } from '@/shared/components/PixelCanvas';

const K = 3;
const SIDE = IMAGE_SIZE;
const OUT = SIDE - K + 1;

const fmt = (v: number) => (Math.abs(v) < 0.005 ? '0' : v.toFixed(2).replace(/^(-?)0\./, '$1.'));

const FALLBACK = renderDigit(4, createRng(11), { jitter: 0.02, tilt: 0.1, noise: 0 });

/** Slide a 3×3 window over the picture and watch one output number being computed. */
export function ConvExplainer() {
  const image = useCnnStore((s) => s.image);
  const center = useCnnStore((s) => s.center);
  const learned = useCnnStore((s) => s.kernels?.conv1);
  const kernelSize = useCnnStore((s) => s.config.kernel);

  const [presetId, setPresetId] = useState('edge-v');
  const [custom, setCustom] = useState<number[]>(() => [...KERNEL_PRESETS[0]!.values]);
  const [pos, setPos] = useState<[number, number]>([6, 5]);

  const options = useMemo(() => {
    const base = KERNEL_PRESETS.map((p) => ({ value: p.id, label: p.label }));
    const learnedOptions =
      kernelSize === K && learned
        ? learned.map((_, i) => ({ value: `learned-${i}`, label: `Learned filter ${i + 1}` }))
        : [];
    return [...base, ...learnedOptions, { value: 'custom', label: 'My own numbers' }];
  }, [learned, kernelSize]);

  const kernel: number[] = useMemo(() => {
    if (presetId === 'custom') return custom;
    const preset = KERNEL_PRESETS.find((p) => p.id === presetId);
    if (preset) return [...preset.values];
    if (presetId.startsWith('learned-') && learned) {
      const k = learned[Number(presetId.slice(8))];
      if (k) return Array.from(k);
    }
    return custom;
  }, [presetId, custom, learned]);

  const blank = useMemo(() => image.every((v) => v === 0), [image]);
  const source = useMemo(
    () => (blank ? FALLBACK : effectiveImage(image, center)),
    [blank, image, center],
  );
  const output = useMemo(() => convolveSingle(source, SIDE, kernel, K), [source, kernel]);
  const scale = useMemo(() => maxAbs(output), [output]);

  const [x, y] = pos;
  const patch = patchAt(source, SIDE, x, y, K);
  const products = patch.map((p, i) => p * (kernel[i] as number));
  const sum = products.reduce((a, b) => a + b, 0);
  const note = KERNEL_PRESETS.find((p) => p.id === presetId)?.note;

  const place = (px: number, py: number) => {
    const next: [number, number] = [windowOrigin(px, K, SIDE), windowOrigin(py, K, SIDE)];
    setPos((prev) => (prev[0] === next[0] && prev[1] === next[1] ? prev : next));
  };

  const pick = (id: string) => {
    if (id === 'custom') setCustom(kernel);
    setPresetId(id);
  };

  const edit = (index: number, value: number) => {
    const next = [...kernel];
    next[index] = value;
    setCustom(next);
    setPresetId('custom');
  };

  return (
    <div className="space-y-6">
      <p className="max-w-3xl text-[0.9rem] text-muted">
        A <em>kernel</em> is a tiny grid of weights. Place it over a patch of the picture, multiply
        pixel by pixel, add everything up: that single number is one pixel of the output. Slide the
        window and the output fills in. Move the pointer over the image on the left.
        {blank && ' (Showing an example digit until you draw one.)'}
      </p>

      <div className="grid items-start gap-8 lg:grid-cols-[auto_minmax(0,1fr)_auto]">
        <div className="space-y-2">
          <p className="eyebrow">Input</p>
          <PixelCanvas
            values={source}
            side={SIDE}
            size={256}
            grid
            box={{ x, y, w: K, h: K }}
            onDown={place}
            onMove={place}
            ariaLabel="Input image with the 3 by 3 convolution window. Move the pointer over it."
          />
        </div>

        <div className="min-w-0 space-y-5">
          <Select label="Kernel" value={presetId} onValueChange={pick} options={options} />
          {note && <p className="-mt-3 text-[0.78rem] text-faint">{note}</p>}

          <div className="flex flex-wrap items-start gap-x-6 gap-y-4 sm:flex-nowrap">
            <Grid3 title="Patch under the window" cells={patch.map(fmt)} tone="plain" />
            <span className="self-center pt-6 font-mono text-xl text-faint">×</span>
            <KernelEditor kernel={kernel} onEdit={edit} />
            <span className="self-center pt-6 font-mono text-xl text-faint">=</span>
            <Grid3 title="Products" cells={products.map(fmt)} tone="products" values={products} />
          </div>

          <p className="font-mono text-[0.92rem]">
            sum of the nine products ={' '}
            <span className="rounded-md bg-brand-soft px-2 py-0.5 text-fg" data-testid="conv-sum">
              {sum.toFixed(2)}
            </span>
            <span className="text-faint">
              {' '}
              → output pixel ({x}, {y})
            </span>
          </p>

          <div className="grid max-w-md gap-4 sm:grid-cols-2">
            <Slider
              label="Window column"
              min={0}
              max={SIDE - K}
              value={x}
              onValueChange={(v) => setPos([v, y])}
            />
            <Slider
              label="Window row"
              min={0}
              max={SIDE - K}
              value={y}
              onValueChange={(v) => setPos([x, v])}
            />
          </div>
        </div>

        <div className="space-y-2">
          <p className="eyebrow">
            Output ({OUT}×{OUT})
          </p>
          <PixelCanvas
            values={output}
            side={OUT}
            size={224}
            mode="diverging"
            scale={scale}
            box={{ x, y, w: 1, h: 1 }}
            ariaLabel="Result of applying the kernel everywhere. Ember is positive, frost is negative."
          />
          <p className="max-w-[224px] text-[0.72rem] leading-snug text-faint">
            Ember = the pattern is here, frost = the opposite pattern, grey = nothing. The output is
            smaller because the window cannot hang over the border.
          </p>
        </div>
      </div>
    </div>
  );
}

function Grid3({
  title,
  cells,
  tone,
  values,
}: {
  title: string;
  cells: string[];
  tone: 'plain' | 'products';
  values?: number[];
}) {
  const palette = usePalette();
  const max = values ? Math.max(1e-9, ...values.map(Math.abs)) : 1;
  return (
    <div>
      <p className="eyebrow mb-2">{title}</p>
      <div className="grid grid-cols-3 gap-1">
        {cells.map((c, i) => {
          const t = values ? (values[i] as number) / max : 0;
          return (
            <span
              key={i}
              className={cn(
                'grid h-10 w-12 place-items-center rounded-md border border-line font-mono text-[0.78rem] tabular-nums',
                tone === 'plain' && 'bg-raised/50',
              )}
              style={
                tone === 'products' && Math.abs(t) > 0.01
                  ? {
                      background: `color-mix(in srgb, var(${t > 0 ? '--pos' : '--neg'}) ${Math.round(Math.abs(t) * 55)}%, transparent)`,
                      color: inkOn(
                        mix(palette.surface, t > 0 ? palette.pos : palette.neg, Math.abs(t) * 0.55),
                      ),
                    }
                  : undefined
              }
            >
              {c}
            </span>
          );
        })}
      </div>
    </div>
  );
}

function KernelEditor({
  kernel,
  onEdit,
}: {
  kernel: number[];
  onEdit: (index: number, value: number) => void;
}) {
  return (
    <div>
      <p className="eyebrow mb-2">Kernel (editable)</p>
      <div className="grid grid-cols-3 gap-1">
        {kernel.map((v, i) => (
          <input
            key={i}
            type="number"
            step={0.25}
            aria-label={`Kernel weight row ${Math.floor(i / K) + 1}, column ${(i % K) + 1}`}
            value={Number(v.toFixed(3))}
            onChange={(e) => {
              const parsed = e.target.valueAsNumber;
              if (Number.isFinite(parsed)) onEdit(i, Math.max(-9, Math.min(9, parsed)));
            }}
            className="h-10 w-14 rounded-md border border-line-strong bg-surface text-center font-mono text-[0.78rem] tabular-nums outline-none focus:border-brand"
          />
        ))}
      </div>
    </div>
  );
}
