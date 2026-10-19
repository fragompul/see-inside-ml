import { createRng } from '@sim/core';
import {
  CLASSIFICATION_DATASETS,
  REGRESSION_DATASETS,
  datasetInfo,
  generateDataset,
  type DatasetName,
  type PlaygroundConfig,
} from '@sim/deep';
import { Dices } from 'lucide-react';
import { useEffect, useRef } from 'react';
import { Button, Segmented, Slider, cn } from '@/design-system';
import { mix, usePalette, type Palette, type Rgb } from '@/shared/canvas/colors';
import { divergingColor } from '../canvas/heatmap';

const THUMB = 52;

/** Tiny live preview of the real generator, so the tile shows exactly what you will get. */
function Thumbnail({ name, palette }: { name: DatasetName; palette: Palette }) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const canvas = ref.current;
    const ctx = canvas?.getContext('2d');
    if (!canvas || !ctx) return;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = THUMB * dpr;
    canvas.height = THUMB * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, THUMB, THUMB);
    const info = datasetInfo[name];
    const data = generateDataset(
      name,
      { samples: 110, noise: 0.08, classes: info.problem === 'classification' ? 2 : 1 },
      createRng(11),
    );
    const maxAbs =
      data.problem === 'regression' ? Math.max(...Array.from(data.values, Math.abs), 1e-6) : 1;
    for (let i = 0; i < data.X.rows; i++) {
      const x = ((data.X.get(i, 0) + 6) / 12) * THUMB;
      const y = THUMB - ((data.X.get(i, 1) + 6) / 12) * THUMB;
      const color: Rgb =
        data.problem === 'classification'
          ? (palette.classes[data.labels[i] as number] as Rgb)
          : divergingColor(
              data.values[i] as number,
              maxAbs,
              palette,
              mix(palette.surface, palette.fg, 0.2),
            );
      ctx.fillStyle = `rgb(${color[0]}, ${color[1]}, ${color[2]})`;
      ctx.beginPath();
      ctx.arc(x, y, 1.9, 0, Math.PI * 2);
      ctx.fill();
    }
  }, [name, palette]);
  return <canvas ref={ref} style={{ width: THUMB, height: THUMB }} className="rounded-md" />;
}

function DatasetTile({
  name,
  active,
  palette,
  onSelect,
}: {
  name: DatasetName;
  active: boolean;
  palette: Palette;
  onSelect: () => void;
}) {
  return (
    <button
      type="button"
      onClick={onSelect}
      aria-pressed={active}
      className={cn(
        'flex cursor-pointer flex-col items-center gap-1.5 rounded-xl border p-2 transition-all duration-200',
        active
          ? 'border-brand bg-brand-soft shadow-[0_0_0_3px_var(--brand-soft)]'
          : 'border-line bg-raised/40 hover:border-line-strong',
      )}
    >
      <Thumbnail name={name} palette={palette} />
      <span className={cn('text-[0.72rem]', active ? 'text-fg' : 'text-muted')}>
        {datasetInfo[name].label}
      </span>
    </button>
  );
}

export function DataPanel({
  config,
  onPatch,
}: {
  config: PlaygroundConfig;
  onPatch: (patch: Partial<PlaygroundConfig>) => void;
}) {
  const palette = usePalette();
  const info = datasetInfo[config.dataset];
  const [minClasses, maxClasses] = info.classes;
  const classOptions = Array.from({ length: maxClasses - minClasses + 1 }, (_, i) =>
    String(minClasses + i),
  );

  const group = (title: string, names: readonly DatasetName[]) => (
    <div>
      <p className="eyebrow mb-2">{title}</p>
      <div className="grid grid-cols-3 gap-2">
        {names.map((name) => (
          <DatasetTile
            key={name}
            name={name}
            palette={palette}
            active={config.dataset === name}
            onSelect={() => onPatch({ dataset: name })}
          />
        ))}
      </div>
    </div>
  );

  return (
    <div className="space-y-6">
      {group('Classification', CLASSIFICATION_DATASETS)}
      {group('Regression', REGRESSION_DATASETS)}

      {maxClasses > minClasses && (
        <div className="flex flex-col gap-2">
          <span className="text-[0.82rem] text-muted">Classes</span>
          <Segmented
            label="Number of classes"
            size="sm"
            value={String(Math.min(maxClasses, Math.max(minClasses, config.classes)))}
            onValueChange={(v) => onPatch({ classes: Number(v) })}
            options={classOptions.map((v) => ({ value: v, label: v }))}
          />
        </div>
      )}

      <Slider
        label="Samples"
        min={50}
        max={600}
        step={10}
        value={config.samples}
        onValueChange={(samples) => onPatch({ samples })}
      />
      <Slider
        label="Noise"
        min={0}
        max={100}
        step={1}
        value={Math.round(config.noise * 100)}
        onValueChange={(v) => onPatch({ noise: v / 100 })}
        format={(v) => `${v}%`}
      />
      <Slider
        label="Train / test split"
        min={50}
        max={90}
        step={5}
        value={config.trainRatio}
        onValueChange={(trainRatio) => onPatch({ trainRatio })}
        format={(v) => `${v} / ${100 - v}`}
      />

      <Button
        variant="secondary"
        size="sm"
        className="w-full"
        onClick={() => onPatch({ seed: Math.floor(Math.random() * 1_000_000) })}
      >
        <Dices className="size-4" /> New random seed
        <span className="ml-1 font-mono text-[0.7rem] text-muted">#{config.seed}</span>
      </Button>
    </div>
  );
}
