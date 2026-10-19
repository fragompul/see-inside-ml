import { type Matrix } from '@sim/core';
import { gelu, type BpeTokenizer, type ForwardTrace } from '@sim/deep';
import { useMemo, useState } from 'react';
import { Segmented, Tex } from '@/design-system';
import { usePalette } from '@/shared/canvas/colors';
import { visible } from '../tokens';
import { Heat, maxAbs } from '@/shared/components/Heat';

interface Props {
  trace: ForwardTrace;
  tokenizer: BpeTokenizer;
  layer: number;
  position: number;
  onLayer: (layer: number) => void;
}

const row = (m: Matrix, i: number) => m.data.subarray(i * m.cols, (i + 1) * m.cols);
const norm = (v: ArrayLike<number>) => {
  let s = 0;
  for (let i = 0; i < v.length; i++) s += (v[i] as number) ** 2;
  return Math.sqrt(s);
};

interface Stage {
  key: string;
  title: string;
  hint: string;
  values: Float64Array;
  /** Stages that are *added* to the stream (attention / MLP outputs) are drawn lighter. */
  delta: boolean;
}

/**
 * The residual stream: one vector per token that every sub-layer reads from and *adds to*.
 * Watching it for a single token shows what attention and the MLP each contribute.
 */
export function BlocksPanel({ trace, tokenizer, layer, position, onLayer }: Props) {
  const palette = usePalette();
  const [hoverNeuron, setHoverNeuron] = useState<number | null>(null);
  const d = trace.embed.cols;
  const p = Math.min(position, trace.ids.length - 1);

  const stages = useMemo<Stage[]>(() => {
    const out: Stage[] = [
      { key: 'x0', title: 'Embedding', hint: 'x₀', values: row(trace.embed, p), delta: false },
    ];
    trace.layers.forEach((l, i) => {
      out.push(
        {
          key: `a${i}`,
          title: `+ Attention (layer ${i + 1})`,
          hint: 'added',
          values: row(l.attnOut, p),
          delta: true,
        },
        {
          key: `m${i}`,
          title: '= Residual stream',
          hint: `after attention ${i + 1}`,
          values: row(l.residMid, p),
          delta: false,
        },
        {
          key: `p${i}`,
          title: `+ MLP (layer ${i + 1})`,
          hint: 'added',
          values: row(l.mlpOut, p),
          delta: true,
        },
        {
          key: `r${i}`,
          title: '= Residual stream',
          hint: `after block ${i + 1}`,
          values: row(l.residOut, p),
          delta: false,
        },
      );
    });
    out.push({
      key: 'f',
      title: 'Final LayerNorm',
      hint: 'ready to read out',
      values: row(trace.finalLn, p),
      delta: false,
    });
    return out;
  }, [trace, p]);

  const longest = Math.max(...stages.map((s) => norm(s.values)), 1e-9);
  const block = trace.layers[layer];
  const pre = block ? row(block.mlpPre, p) : new Float64Array(0);
  const act = block ? row(block.mlpAct, p) : new Float64Array(0);
  const hidden = pre.length;
  const active = act.reduce((n, v) => n + (Math.abs(v) > 0.1 ? 1 : 0), 0);
  const mlpScale = maxAbs(pre, act);

  // GELU curve with this token's neurons plotted on it
  const lo = Math.min(-3, ...pre);
  const hi = Math.max(3, ...pre);
  const W = 360;
  const H = 230;
  const px = (x: number) => 28 + ((x - lo) / (hi - lo)) * (W - 40);
  const gy = (v: number) => H - 24 - ((v - gelu(-0.75)) / (hi - gelu(-0.75))) * (H - 44);
  const curve = Array.from({ length: 121 }, (_, i) => {
    const x = lo + ((hi - lo) * i) / 120;
    return `${i === 0 ? 'M' : 'L'}${px(x).toFixed(1)},${gy(gelu(x)).toFixed(1)}`;
  }).join(' ');

  return (
    <div className="grid grid-cols-[minmax(0,1fr)] gap-10 lg:grid-cols-[minmax(0,1.05fr)_minmax(0,1fr)]">
      <div>
        <p className="mb-4 text-[0.88rem] leading-relaxed text-muted">
          Each block <em className="text-fg">adds</em> its result to a running vector instead of
          replacing it: first what attention gathered from other tokens, then what the MLP computed
          from it. This is the residual stream of{' '}
          <span className="font-mono text-fg">
            “{visible(tokenizer.tokenString(trace.ids[p] as number))}”
          </span>
          .
        </p>
        <ol
          tabIndex={0}
          aria-label="Blocks of the model"
          className="space-y-2.5 overflow-x-auto rounded-xl border border-line bg-raised/30 p-4"
        >
          {stages.map((s) => (
            <li
              key={s.key}
              className="grid min-w-[26rem] grid-cols-[10.5rem_auto_3.6rem] items-center gap-3"
            >
              <div className="min-w-0">
                <p className={s.delta ? 'text-[0.78rem] text-muted' : 'text-[0.82rem] text-fg'}>
                  {s.title}
                </p>
                <p className="font-mono text-[0.64rem] text-faint">{s.hint}</p>
              </div>
              <Heat
                values={s.values}
                cols={d}
                scale={maxAbs(s.values)}
                cellW={7}
                cellH={s.delta ? 12 : 18}
                ariaLabel={`${s.title}: ${d} numbers`}
              />
              <span className="block text-right" title="length of the vector">
                <span className="block font-mono text-[0.68rem] text-muted tabular-nums">
                  {norm(s.values).toFixed(1)}
                </span>
                <span className="mt-1 block h-1 overflow-hidden rounded-full bg-line">
                  <span
                    className="block h-full rounded-full bg-brand"
                    style={{ width: `${Math.max(3, (norm(s.values) / longest) * 100)}%` }}
                  />
                </span>
              </span>
            </li>
          ))}
        </ol>
        <p className="mt-2 font-mono text-[0.68rem] text-faint">
          each row has its own colour scale · the bar is the vector's length relative to the longest
        </p>
      </div>

      <div className="space-y-5">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <p className="eyebrow">The MLP of layer {layer + 1}</p>
          <Segmented
            label="Layer"
            size="sm"
            fit
            value={String(layer)}
            onValueChange={(v) => onLayer(Number(v))}
            options={trace.layers.map((_, i) => ({ value: String(i), label: `Layer ${i + 1}` }))}
          />
        </div>
        <p className="text-[0.85rem] leading-relaxed text-muted">
          The MLP widens each token's vector to {hidden} numbers, applies a non-linearity (GELU),
          then narrows it back to {d}. Most of its neurons stay quiet for any one token:{' '}
          <span className="text-fg">
            {active} of {hidden}
          </span>{' '}
          are active here.
        </p>

        <div className="space-y-3 overflow-x-auto rounded-xl border border-line bg-raised/30 p-4">
          <div>
            <p className="mb-1.5 text-[0.78rem] text-muted">Before GELU</p>
            <Heat
              values={pre}
              cols={16}
              scale={mlpScale}
              cellW={14}
              cellH={7}
              onHover={setHoverNeuron}
              ariaLabel="MLP neurons before the non-linearity"
            />
          </div>
          <div>
            <p className="mb-1.5 text-[0.78rem] text-muted">After GELU</p>
            <Heat
              values={act}
              cols={16}
              scale={mlpScale}
              cellW={14}
              cellH={7}
              onHover={setHoverNeuron}
              ariaLabel="MLP neurons after the non-linearity"
            />
          </div>
          <p className="h-4 font-mono text-[0.7rem] text-muted">
            {hoverNeuron !== null
              ? `neuron ${hoverNeuron}: ${(pre[hoverNeuron] as number).toFixed(2)} → ${(act[hoverNeuron] as number).toFixed(2)}`
              : 'hover a neuron'}
          </p>
        </div>

        <div>
          <svg
            viewBox={`0 0 ${W} ${H}`}
            className="w-full rounded-xl border border-line bg-raised/30"
            role="img"
            aria-label="GELU curve with the neurons of this token on it"
          >
            <line x1={px(lo)} x2={px(hi)} y1={gy(0)} y2={gy(0)} stroke="var(--line-strong)" />
            <line x1={px(0)} x2={px(0)} y1={12} y2={H - 24} stroke="var(--line-strong)" />
            <path d={curve} fill="none" stroke={palette.css.brand} strokeWidth={2} />
            {Array.from(pre, (x, i) => (
              <circle
                key={i}
                cx={px(x)}
                cy={gy(act[i] as number)}
                r={hoverNeuron === i ? 5 : 2.6}
                fill={x >= 0 ? palette.css.pos : palette.css.neg}
                fillOpacity={hoverNeuron === null || hoverNeuron === i ? 0.75 : 0.25}
              />
            ))}
            <text
              x={W - 10}
              y={gy(0) - 6}
              textAnchor="end"
              className="fill-faint font-mono text-[10px]"
            >
              input
            </text>
            <text x={px(0) + 6} y={20} className="fill-faint font-mono text-[10px]">
              output
            </text>
          </svg>
          <div className="mt-1 text-[0.82rem] text-muted">
            <Tex tex="\mathrm{GELU}(x)\approx x\,\Phi(x)" /> lets positive values through and mutes
            negative ones, smoothly. Each dot is one neuron of this token.
          </div>
        </div>
      </div>
    </div>
  );
}
