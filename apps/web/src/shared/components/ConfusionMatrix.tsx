import { useState } from 'react';
import { cn } from '@/design-system';
import { contrast, mix, usePalette, type Palette, type Rgb } from '@/shared/canvas/colors';

const INKS: Rgb[] = [
  [10, 10, 20],
  [255, 255, 255],
];

/** Tint and text colour for a cell: the tint fades until one of the two inks reads at 4.5:1 or better. */
function cellPaint(palette: Palette, diagonal: boolean, share: number) {
  const base = diagonal ? palette.brand : palette.neg;
  const unders = [palette.surface, palette.raised, palette.bg];
  let alpha = 0.18 + share * 0.7;
  const score = (a: number) =>
    INKS.map((ink) => Math.min(...unders.map((u) => contrast(mix(u, base, a), ink))));
  while (alpha > 0.12 && Math.max(...score(alpha)) < 4.6) alpha -= 0.02;
  const [dark, light] = score(alpha) as [number, number];
  return {
    background: `color-mix(in srgb, var(${diagonal ? '--brand' : '--neg'}) ${Math.round(alpha * 100)}%, transparent)`,
    color: dark >= light ? '#0a0a14' : '#ffffff',
  };
}

/** Rows = the true class, columns = what the network said, counted over the test examples. */
export function ConfusionMatrix({
  matrix,
  labels,
  noun = 'digit',
}: {
  matrix: number[][];
  /** Names of the classes (defaults to 0, 1, 2…). */
  labels?: readonly string[];
  /** What one class is called in the sentences ("digit", "answer"). */
  noun?: string;
}) {
  const palette = usePalette();
  const name = (i: number) => labels?.[i] ?? String(i);
  const [hover, setHover] = useState<[number, number] | null>(null);
  const hovered = hover ? (matrix[hover[0]]?.[hover[1]] ?? 0) : null;
  const rowTotal = hover ? (matrix[hover[0]]?.reduce((a, b) => a + b, 0) ?? 0) : 0;

  return (
    <div>
      <div className="flex gap-2">
        <span className="eyebrow self-center [writing-mode:vertical-rl] rotate-180" aria-hidden>
          True {noun}
        </span>
        <div className="min-w-0 grow">
          <div
            className="grid gap-[2px]"
            style={{ gridTemplateColumns: `auto repeat(${matrix.length}, minmax(0, 1fr))` }}
            role="table"
            aria-label="Confusion matrix"
          >
            <div role="row" className="contents">
              <span role="columnheader" aria-label="True class" />
              {matrix.map((_, c) => (
                <span
                  key={c}
                  role="columnheader"
                  className="text-center font-mono text-[0.68rem] text-faint"
                >
                  {name(c)}
                </span>
              ))}
            </div>
            {matrix.map((row, r) => {
              const total = row.reduce((a, b) => a + b, 0) || 1;
              return (
                <div key={r} role="row" className="contents">
                  <span
                    role="rowheader"
                    className="pr-1.5 text-right font-mono text-[0.68rem] leading-6 text-faint"
                  >
                    {name(r)}
                  </span>
                  {row.map((count, c) => {
                    const share = count / total;
                    const diagonal = r === c;
                    return (
                      <span
                        key={c}
                        role="cell"
                        onPointerEnter={() => setHover([r, c])}
                        onPointerLeave={() => setHover(null)}
                        className={cn(
                          'grid aspect-square min-h-6 place-items-center rounded-[4px] border font-mono text-[0.64rem] tabular-nums',
                          count === 0 ? 'border-line text-transparent' : 'border-transparent',
                          hover && hover[0] === r && hover[1] === c && 'ring-2 ring-fg',
                        )}
                        style={count > 0 ? cellPaint(palette, diagonal, share) : undefined}
                      >
                        {count}
                      </span>
                    );
                  })}
                </div>
              );
            })}
          </div>
          <p className="eyebrow mt-2 text-center">Network said</p>
        </div>
      </div>
      <p className="mt-3 min-h-5 text-[0.78rem] text-muted" aria-live="polite">
        {hover && hovered !== null
          ? hover[0] === hover[1]
            ? `${hovered} of ${rowTotal} ${noun}s that really were ${name(hover[0])} were recognised.`
            : `${hovered} of ${rowTotal} ${noun}s that really were ${name(hover[0])} were mistaken for ${name(hover[1])}.`
          : 'Bright diagonal = correct. Any colour off the diagonal is a confusion: hover to read it.'}
      </p>
    </div>
  );
}
