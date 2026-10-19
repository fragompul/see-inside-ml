import { NEURON_DATASETS, bestLine, neuronData, neuronDatasetInfo } from '@sim/deep';
import { ArrowRight } from 'lucide-react';
import { useMemo } from 'react';
import { Link } from 'react-router-dom';
import { cn } from '@/design-system';

/** What one line can and cannot do: the reason networks have layers. */
export function LimitsPanel() {
  const results = useMemo(
    () => NEURON_DATASETS.map((d) => ({ dataset: d, accuracy: bestLine(neuronData(d)).accuracy })),
    [],
  );
  return (
    <div className="space-y-6">
      <p className="max-w-3xl text-[0.9rem] text-muted">
        A single neuron draws one straight line. That is enough for AND, OR and any pair of
        well-separated clouds, but some patterns cannot be cut by a line at all. The table is the
        best accuracy <em>any</em> line can reach, found by trying every direction and every cut
        (not by gradient descent).
      </p>
      <ul className="max-w-xl space-y-3" aria-label="Best accuracy of a single line">
        {results.map(({ dataset, accuracy }) => (
          <li key={dataset} className="flex items-center gap-4">
            <span className="w-24 shrink-0 text-[0.84rem] text-fg">
              {neuronDatasetInfo[dataset].label}
            </span>
            <div className="h-4 grow overflow-hidden rounded bg-raised/70">
              <div
                className={cn('h-full rounded', accuracy === 1 ? 'bg-brand' : 'bg-neg')}
                style={{ width: `${accuracy * 100}%` }}
              />
            </div>
            <span className="w-12 text-right font-mono text-[0.8rem] text-muted tabular-nums">
              {Math.round(accuracy * 100)}%
            </span>
          </li>
        ))}
      </ul>
      <p className="max-w-3xl text-[0.9rem] text-muted">
        The fix is to stack neurons: the first layer draws several lines, and the next layer
        combines them, so the network can carve out regions of any shape. That is a multi-layer
        perceptron, and the next lab lets you build one, add layers and watch it solve XOR and a
        spiral.
      </p>
      <Link
        to="/mlp"
        className="inline-flex items-center gap-2 rounded-full border border-line-strong px-4 py-2 text-sm text-fg transition-colors hover:border-brand hover:bg-brand-soft"
      >
        Open the MLP playground <ArrowRight className="size-4" />
      </Link>
    </div>
  );
}
