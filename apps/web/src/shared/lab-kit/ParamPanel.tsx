import { Segmented, Select, Slider, Switch } from '@/design-system';
import { ladderIndex, num, type ParamSpec, type ParamValues } from './params';

/** Draws the controls of a model from its parameter specs. */
export function ParamPanel({
  specs,
  values,
  onChange,
  columns = 2,
}: {
  specs: readonly ParamSpec[];
  values: ParamValues;
  onChange: (key: string, value: number | string | boolean) => void;
  columns?: 1 | 2 | 3;
}) {
  const visible = specs.filter((s) => !s.showIf || s.showIf(values));
  const cols =
    columns === 1 ? '' : columns === 2 ? 'sm:grid-cols-2' : 'sm:grid-cols-2 lg:grid-cols-3';
  return (
    <div className={`grid gap-x-10 gap-y-6 ${cols}`}>
      {visible.map((spec) => (
        <div key={spec.key} className="min-w-0">
          {spec.kind === 'slider' && (
            <Slider
              label={spec.label}
              min={spec.min}
              max={spec.max}
              step={spec.step ?? 1}
              value={num(values, spec.key)}
              onValueChange={(v) => onChange(spec.key, v)}
              {...(spec.format ? { format: spec.format } : {})}
            />
          )}
          {spec.kind === 'ladder' && (
            <Slider
              label={spec.label}
              min={0}
              max={spec.values.length - 1}
              value={ladderIndex(spec.values, num(values, spec.key))}
              onValueChange={(i) => onChange(spec.key, spec.values[i] as number)}
              format={() => (spec.format ?? String)(num(values, spec.key))}
            />
          )}
          {spec.kind === 'select' && (
            <Select
              label={spec.label}
              value={String(values[spec.key])}
              onValueChange={(v) => onChange(spec.key, v)}
              options={spec.options}
            />
          )}
          {spec.kind === 'segmented' && (
            <div className="flex flex-col gap-2">
              <span className="text-[0.82rem] text-muted">{spec.label}</span>
              <Segmented
                label={spec.label}
                size="sm"
                value={String(values[spec.key])}
                onValueChange={(v) => onChange(spec.key, v)}
                options={spec.options}
              />
            </div>
          )}
          {spec.kind === 'switch' && (
            <Switch
              label={spec.label}
              checked={Boolean(values[spec.key])}
              onCheckedChange={(v) => onChange(spec.key, v)}
            />
          )}
          {spec.hint && (
            <p className="mt-1.5 text-[0.72rem] leading-snug text-faint">{spec.hint}</p>
          )}
        </div>
      ))}
    </div>
  );
}
