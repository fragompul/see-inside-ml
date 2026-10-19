import { type EpochStats, type SceneData } from '@sim/deep';
import { cn } from '@/design-system';

function Tile({ label, value, tone }: { label: string; value: string; tone?: string }) {
  return (
    <div className="flex min-w-0 flex-col gap-1 px-4 py-3">
      <span className="eyebrow truncate">{label}</span>
      <span className={cn('font-mono text-[1.35rem] leading-none font-medium tabular-nums', tone)}>
        {value}
      </span>
    </div>
  );
}

/** Fixed decimals for normal values, scientific notation when they would round to zero. */
export const fmt = (v: number | undefined) => {
  if (v === undefined || !Number.isFinite(v)) return '–';
  if (v !== 0 && Math.abs(v) < 0.001) return v.toExponential(1);
  return v.toFixed(3);
};

export function StatsStrip({ stats, scene }: { stats: EpochStats | undefined; scene: SceneData }) {
  const classification = scene.problem === 'classification';
  return (
    <div className="grid grid-cols-2 divide-x divide-y divide-line overflow-hidden rounded-xl border border-line-strong bg-raised/40">
      <Tile label="Train loss" value={fmt(stats?.trainLoss)} tone="text-brand" />
      <Tile label="Test loss" value={fmt(stats?.testLoss)} tone="text-class-2" />
      <Tile
        label={classification ? 'Test accuracy' : 'Test MSE'}
        value={
          classification
            ? stats
              ? `${(stats.testMetric * 100).toFixed(1)}%`
              : '–'
            : fmt(stats?.testMetric)
        }
      />
      <Tile label="Parameters" value={String(scene.parameterCount)} />
    </div>
  );
}
