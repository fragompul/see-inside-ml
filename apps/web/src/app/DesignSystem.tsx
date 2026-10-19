import { useState, type ReactNode } from 'react';
import { Badge, Button, Card, Kbd, Slider, Switch, Tabs, Tooltip, cn } from '@/design-system';

const swatches = [
  ['bg', 'Canvas'],
  ['bg-soft', 'Canvas soft'],
  ['surface', 'Surface'],
  ['raised', 'Raised'],
  ['fg', 'Ink'],
  ['muted', 'Muted'],
  ['brand', 'Brand · iris'],
  ['pos', 'Positive · ember'],
  ['neg', 'Negative · frost'],
  ['ok', 'Success'],
  ['danger', 'Danger'],
] as const;

function Section({ title, note, children }: { title: string; note?: string; children: ReactNode }) {
  return (
    <section className="border-t border-line py-12">
      <div className="mb-8 grid gap-2 md:grid-cols-[14rem_1fr] md:gap-10">
        <div>
          <h2 className="font-display text-xl font-semibold tracking-tight">{title}</h2>
          {note && <p className="mt-2 text-sm text-muted">{note}</p>}
        </div>
        <div>{children}</div>
      </div>
    </section>
  );
}

export function DesignSystem() {
  const [lr, setLr] = useState(0.03);
  const [noise, setNoise] = useState(25);
  const [skip, setSkip] = useState(true);

  return (
    <div className="mx-auto max-w-5xl px-5 py-16 sm:px-8">
      <p className="eyebrow">Design system</p>
      <h1 className="mt-4 font-display text-5xl font-semibold tracking-[-0.04em]">
        Parchment <span className="font-serif font-normal italic text-muted">&</span> Obsidian
      </h1>
      <p className="mt-5 max-w-2xl text-muted">
        One semantic system in two palettes. Colour always carries meaning: ember for positive
        values, frost for negative, iris for interaction. Switch the theme in the header to compare.
      </p>

      <div className="mt-12" />

      <Section title="Colour" note="Tokens are CSS variables; Tailwind utilities map to them.">
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 md:grid-cols-4">
          {swatches.map(([token, label]) => (
            <div key={token} className="overflow-hidden rounded-xl border border-line">
              <div className="h-16" style={{ background: `var(--${token})` }} />
              <div className="bg-surface px-3 py-2">
                <p className="text-[0.8rem]">{label}</p>
                <p className="font-mono text-[0.68rem] text-faint">--{token}</p>
              </div>
            </div>
          ))}
        </div>
        <div className="mt-5 h-3 rounded-full bg-gradient-to-r from-neg via-brand to-pos" />
        <p className="mt-2 font-mono text-[0.7rem] text-muted">
          weight scale: −1 (frost) → 0 (iris) → +1 (ember)
        </p>
      </Section>

      <Section
        title="Typography"
        note="Bricolage Grotesque · Instrument Serif · Geist · JetBrains Mono"
      >
        <div className="space-y-4">
          <p className="font-display text-5xl font-semibold tracking-[-0.04em]">Display heading</p>
          <p className="font-serif text-4xl italic">An editorial accent.</p>
          <p className="max-w-xl text-muted">
            Body text is set in Geist: neutral and highly legible, so the visualizations carry the
            personality.
          </p>
          <p className="font-mono text-sm tabular-nums">
            loss = 0.2941 &nbsp; acc = 94.7% &nbsp; lr = 3.0e-2
          </p>
        </div>
      </Section>

      <Section title="Buttons & badges">
        <div className="flex flex-wrap items-center gap-3">
          <Button>Train</Button>
          <Button variant="secondary">Step</Button>
          <Button variant="ghost">Reset</Button>
          <Button disabled>Disabled</Button>
          <Button size="sm">Small</Button>
          <Button size="lg">Large</Button>
        </div>
        <div className="mt-5 flex flex-wrap items-center gap-2">
          <Badge>Neutral</Badge>
          <Badge tone="brand">Brand</Badge>
          <Badge tone="pos">Positive</Badge>
          <Badge tone="neg">Negative</Badge>
          <Badge tone="ok">Live</Badge>
          <span className="ml-2 text-sm text-muted">
            Press <Kbd>Space</Kbd> to play, <Kbd>→</Kbd> to step
          </span>
        </div>
      </Section>

      <Section title="Controls" note="Radix primitives: fully keyboard accessible.">
        <Card>
          <div className="grid gap-8 p-6 sm:grid-cols-2">
            <div className="space-y-6">
              <Slider
                label="Learning rate"
                value={lr}
                onValueChange={setLr}
                min={0.001}
                max={0.1}
                step={0.001}
                format={(v) => v.toFixed(3)}
              />
              <Slider
                label="Noise"
                value={noise}
                onValueChange={setNoise}
                min={0}
                max={50}
                format={(v) => `${v}%`}
              />
              <Switch label="Skip connections" checked={skip} onCheckedChange={setSkip} />
            </div>
            <div>
              <Tabs
                items={[
                  { value: 'forward', label: 'Forward', content: <Pane>ŷ = σ(W·x + b)</Pane> },
                  { value: 'loss', label: 'Loss', content: <Pane>L = −Σ y · log ŷ</Pane> },
                  { value: 'backward', label: 'Backward', content: <Pane>∂L/∂W = δ · xᵀ</Pane> },
                ]}
              />
              <div className="mt-6">
                <Tooltip content="Hover or focus to reveal context, formulas and live values.">
                  <button
                    className={cn(
                      'cursor-help rounded-md border border-dashed border-line-strong px-3 py-1.5',
                      'font-mono text-xs text-muted transition-colors hover:text-fg',
                    )}
                  >
                    Hover me
                  </button>
                </Tooltip>
              </div>
            </div>
          </div>
        </Card>
      </Section>
    </div>
  );
}

function Pane({ children }: { children: ReactNode }) {
  return <p className="font-mono text-sm text-fg">{children}</p>;
}
