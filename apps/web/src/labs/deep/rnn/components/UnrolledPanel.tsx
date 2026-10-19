import { cellInfo, taskInfo, type CellName } from '@sim/deep';
import { useMemo, useState, type ReactNode } from 'react';
import { Tex, cn } from '@/design-system';
import { Heat, maxAbs } from '@/shared/components/Heat';
import { useRnnStore } from '../store';
import { logPosition, logRange, stepFacts, unitMajor } from '../view';
import { tokenColor } from './SequencePanel';

const LABEL_W = 108;
const BAR_H = 64;

const EQUATIONS: Record<CellName, string[]> = {
  rnn: [String.raw`h_t=\tanh\!\left(W_x\,x_t+W_h\,h_{t-1}+b\right)`],
  lstm: [
    String.raw`f_t=\sigma(\cdot),\;\; i_t=\sigma(\cdot),\;\; o_t=\sigma(\cdot),\;\; g_t=\tanh(\cdot)`,
    String.raw`c_t=f_t\odot c_{t-1}+i_t\odot g_t\qquad h_t=o_t\odot\tanh(c_t)`,
  ],
  gru: [
    String.raw`z_t=\sigma(\cdot),\;\; r_t=\sigma(\cdot),\;\; n_t=\tanh\!\left(W_x x_t+r_t\odot (W_h h_{t-1})\right)`,
    String.raw`h_t=(1-z_t)\odot n_t+z_t\odot h_{t-1}`,
  ],
};

/** The network unrolled over time: one column per step, one row of heat per kind of signal. */
export function UnrolledPanel() {
  const trace = useRnnStore((s) => s.trace);
  const norms = useRnnStore((s) => s.gradientNorms);
  const config = useRnnStore((s) => s.config);
  const [picked, setPicked] = useState<number | null>(null);

  const steps = trace?.steps ?? 0;
  const hidden = trace?.hidden ?? 0;
  const step = picked !== null && picked < steps ? picked : steps - 1;
  const cellW = Math.max(14, Math.min(44, Math.floor(620 / Math.max(1, steps))));
  const cellH = Math.max(3, Math.min(11, Math.floor(150 / Math.max(1, hidden))));

  const rows = useMemo(() => {
    if (!trace) return [];
    const list: {
      key: string;
      title: string;
      hint: string;
      values: Float64Array;
      scale: number;
    }[] = [
      {
        key: 'h',
        title: 'Hidden state',
        hint: 'memory passed to the next step',
        values: unitMajor(trace.h, steps, hidden, 1),
        scale: 1,
      },
    ];
    if (trace.c) {
      const values = unitMajor(trace.c, steps, hidden, 1);
      list.push({
        key: 'c',
        title: 'Cell state',
        hint: 'long-term lane, edited by gates',
        values,
        scale: Math.max(1, maxAbs(values)),
      });
    }
    const gateRows: Record<string, [string, string]> = {
      forget: ['Forget gate', 'bright = keep the old memory'],
      input: ['Input gate', 'bright = write new information'],
      output: ['Output gate', 'bright = reveal the memory'],
      update: ['Update gate', 'bright = keep the old state'],
      reset: ['Reset gate', 'bright = let the past feed the candidate'],
    };
    for (const [name, [title, hint]] of Object.entries(gateRows)) {
      const g = trace.gates[name];
      if (g) list.push({ key: name, title, hint, values: unitMajor(g, steps, hidden), scale: 1 });
    }
    return list;
  }, [trace, steps, hidden]);

  const range = useMemo(() => (norms ? logRange(norms, 3) : null), [norms]);

  if (!trace) return null;
  const info = taskInfo[config.task];
  const facts = stepFacts(trace, step);
  const left = LABEL_W + step * cellW;

  return (
    <div className="space-y-6">
      <p className="max-w-3xl text-[0.9rem] text-muted">
        A recurrent network is one small block of weights used again at every step. Drawn out in
        time it looks like a long chain: each step reads one token and the memory left by the
        previous step. Hover or click a column to see what happened there.
      </p>

      <div className="rounded-xl border border-line bg-raised/30 px-4 py-3 text-[0.95rem]">
        {EQUATIONS[config.cell].map((tex) => (
          <div key={tex} className="overflow-x-auto py-0.5">
            <Tex tex={tex} />
          </div>
        ))}
        <p className="mt-1 text-[0.78rem] text-faint">{cellInfo[config.cell].note}</p>
      </div>

      <div className="overflow-x-auto pb-2">
        <div className="relative w-max" onPointerLeave={() => setPicked(null)}>
          <Row label="Input" hint="read at each step" height={44}>
            <div className="flex">
              {Array.from(trace.tokens, (token, t) => (
                <button
                  key={t}
                  type="button"
                  onPointerEnter={() => setPicked(t)}
                  onFocus={() => setPicked(t)}
                  aria-label={`Step ${t + 1}, token ${info.symbols[token]}`}
                  style={{ width: cellW, ['--c' as string]: tokenColor(token) }}
                  className="grid h-9 cursor-pointer place-items-center font-mono text-[0.78rem] text-fg"
                >
                  <span className="grid size-[calc(100%-4px)] place-items-center rounded-md border border-[color:var(--c)] bg-[color-mix(in_srgb,var(--c)_18%,transparent)]">
                    {info.symbols[token]}
                  </span>
                </button>
              ))}
            </div>
          </Row>

          {rows.map((row) => (
            <Row key={row.key} label={row.title} hint={row.hint} height={hidden * cellH + 12}>
              <Heat
                values={row.values}
                cols={steps}
                scale={row.scale}
                cellW={cellW}
                cellH={cellH}
                onHover={(i) => i !== null && setPicked(i % steps)}
                onSelect={(i) => setPicked(i % steps)}
                ariaLabel={`${row.title} of every neuron at every step`}
              />
            </Row>
          ))}

          {norms && range && (
            <Row
              label="Learning signal"
              hint="‖dLoss/dh‖, log scale: how much this step can still learn"
              height={BAR_H + 22}
            >
              <div className="flex items-end" style={{ height: BAR_H }}>
                {Array.from(norms, (n, t) => (
                  <div
                    key={t}
                    className="flex items-end justify-center"
                    style={{ width: cellW, height: BAR_H }}
                    onPointerEnter={() => setPicked(t)}
                  >
                    <div
                      className="w-[60%] rounded-t-sm bg-brand/80"
                      style={{ height: Math.max(2, logPosition(n, range.lo, range.hi) * BAR_H) }}
                    />
                  </div>
                ))}
              </div>
              <div className="mt-1 flex justify-between font-mono text-[0.6rem] text-faint">
                <span>1e{range.lo}</span>
                <span>1e{range.hi}</span>
              </div>
            </Row>
          )}

          <div
            aria-hidden
            className="pointer-events-none absolute top-0 bottom-0 rounded-md border-2 border-brand/80 transition-[left] duration-150"
            style={{ left, width: cellW }}
          />
        </div>
      </div>

      <div className="rounded-xl border border-line bg-surface/60 px-4 py-3 text-[0.86rem] leading-relaxed">
        <p>
          <span className="eyebrow mr-2">Step {step + 1}</span>
          Reads <span className="font-mono text-fg">{info.symbols[facts.token]}</span>; the hidden
          state moves by {facts.change.toFixed(2)} on average.
          {facts.gates.length === 0 &&
            ' A vanilla RNN has no gates: it rewrites its whole memory with a tanh every step.'}
        </p>
        {facts.gates.length > 0 && (
          <ul className="mt-1.5 grid gap-x-8 gap-y-0.5 text-muted sm:grid-cols-2">
            {facts.gates.map((g) => (
              <li key={g.name}>
                <span className="font-mono text-fg">
                  {g.name} {g.mean.toFixed(2)}
                </span>{' '}
                · {g.meaning}
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  );
}

function Row({
  label,
  hint,
  height,
  children,
}: {
  label: string;
  hint: string;
  height: number;
  children: ReactNode;
}) {
  return (
    <div className={cn('flex items-start')} style={{ minHeight: height }}>
      <div className="shrink-0 pr-3" style={{ width: LABEL_W }}>
        <p className="font-mono text-[0.68rem] leading-tight text-fg">{label}</p>
        <p className="mt-0.5 text-[0.6rem] leading-tight text-faint">{hint}</p>
      </div>
      <div>{children}</div>
    </div>
  );
}
