import type { Histogram } from '@sim/deep';
import { useEffect } from 'react';
import { fix } from '@/shared/lab-kit';
import { useMlpStore } from '../store';

const BINS = 24;

function Bars({ h, tone, label }: { h: Histogram; tone: 'brand' | 'pos'; label: string }) {
  const peak = Math.max(...h.counts, 1);
  return (
    <figure className="space-y-1">
      <svg viewBox={`0 0 ${BINS * 8} 56`} role="img" aria-label={label} className="block w-full">
        {h.counts.map((c, i) => (
          <rect
            key={i}
            x={i * 8 + 0.5}
            y={52 - (c / peak) * 50}
            width={7}
            height={(c / peak) * 50}
            rx={1}
            className={tone === 'brand' ? 'fill-brand/80' : 'fill-pos/80'}
          />
        ))}
        <line x1={0} x2={BINS * 8} y1={52.5} y2={52.5} className="stroke-line-strong" />
        <line
          x1={BINS * 4}
          x2={BINS * 4}
          y1={0}
          y2={54}
          className="stroke-line"
          strokeDasharray="2 3"
        />
      </svg>
      <figcaption className="flex justify-between font-mono text-[0.62rem] text-faint tabular-nums">
        <span>{h.min.toExponential(1)}</span>
        <span>0</span>
        <span>{h.max.toExponential(1)}</span>
      </figcaption>
    </figure>
  );
}

/** How the weights of each layer are spread, and how big their latest gradients are. */
export function HistogramPanel() {
  const analysis = useMlpStore((s) => s.analysis);
  const running = useMlpStore((s) => s.running);
  const sceneVersion = useMlpStore((s) => s.sceneVersion);
  const epochs = useMlpStore((s) => s.history.length);
  const request = useMlpStore((s) => s.requestAnalysis);

  useEffect(() => {
    request({ histograms: BINS });
  }, [request, sceneVersion, running, epochs === 0]); // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => {
    if (!running) return;
    const id = window.setInterval(() => request({ histograms: BINS }), 800);
    return () => window.clearInterval(id);
  }, [running, request]);

  const layers = analysis?.histograms;
  if (!layers) return <p className="text-sm text-muted">Computing…</p>;
  return (
    <div className="space-y-6">
      <div className="grid gap-x-8 gap-y-6 md:grid-cols-2 xl:grid-cols-3">
        {layers.map((l, i) => (
          <div
            key={i}
            className="space-y-3 rounded-xl border border-line p-3"
            data-testid={`hist-layer-${i}`}
          >
            <p className="eyebrow">Layer {i + 1}</p>
            <div>
              <p className="mb-1 text-[0.72rem] text-muted">
                Weights · RMS <span className="font-mono text-fg">{fix(l.weightRms)}</span>
                {l.zeroShare > 0 && (
                  <span className="ml-2 font-mono text-faint">
                    {Math.round(l.zeroShare * 100)}% exactly zero
                  </span>
                )}
              </p>
              <Bars
                h={l.weights}
                tone="brand"
                label={`Distribution of the weights of layer ${i + 1}`}
              />
            </div>
            <div>
              <p className="mb-1 text-[0.72rem] text-muted">
                Latest gradients · RMS{' '}
                <span className="font-mono text-fg">{fix(l.gradientRms, 4)}</span>
              </p>
              <Bars
                h={l.gradients}
                tone="pos"
                label={`Distribution of the gradients of layer ${i + 1}`}
              />
            </div>
          </div>
        ))}
      </div>
      <p className="max-w-3xl text-sm text-muted">
        Healthy training keeps the weights spread out and the gradients neither vanishing (a spike
        at zero, tiny RMS in the early layers) nor exploding. L1 regularisation piles weights up at
        exactly zero; L2 squeezes the whole distribution towards the centre. Watch the gradients
        shrink as the loss flattens out.
      </p>
    </div>
  );
}
