import { cn } from '@/design-system';

export interface Metric {
  label: string;
  value: string;
  hint?: string;
  tone?: 'good' | 'bad' | 'neutral';
  testId?: string;
}

/** A row of labelled numbers: scores, counts, fitted values. */
export function MetricsStrip({ metrics, className }: { metrics: Metric[]; className?: string }) {
  return (
    <dl
      className={cn(
        'grid grid-cols-2 gap-x-6 gap-y-3 rounded-xl border border-line bg-raised/30 p-4 sm:grid-cols-3 lg:grid-cols-4',
        className,
      )}
    >
      {metrics.map((m) => (
        <div key={m.label} title={m.hint}>
          <dt className="text-[0.68rem] text-faint">{m.label}</dt>
          <dd
            data-testid={m.testId}
            className={cn(
              'font-mono text-lg tabular-nums',
              m.tone === 'good' ? 'text-ok' : m.tone === 'bad' ? 'text-neg' : 'text-fg',
            )}
          >
            {m.value}
          </dd>
        </div>
      ))}
    </dl>
  );
}

/** Formats a share as a whole percentage ("–" when missing). */
export const pct = (v: number | null | undefined): string =>
  v === null || v === undefined ? '–' : `${Math.round(v * 100)}%`;

/** Formats a number with a sensible precision ("–" when missing or not finite). */
export const fix = (v: number | null | undefined, digits = 3): string =>
  v === null || v === undefined || !Number.isFinite(v) ? '–' : v.toFixed(digits);
