import { topographicError, uMatrix } from '@sim/classic';
import { useMemo, useState } from 'react';
import { Segmented } from '@/design-system';
import { usePalette } from '@/shared/canvas/colors';
import { PixelCanvas } from '@/shared/components/PixelCanvas';
import { ProjectorLab, fix, pct, type ProjectorContext } from '@/shared/lab-kit';
import { fitSom, somSpec } from './somSpec';

/** The trained map for the current data and settings (the same fit the embedding came from). */
function useMap(ctx: ProjectorContext) {
  const side = Number(ctx.params.side);
  return useMemo(() => {
    const som = fitSom(ctx.params, ctx.data.x);
    const cells = som.assign(ctx.data.x);
    const hits = new Int32Array(side * side);
    const votes = Array.from({ length: side * side }, () => new Map<number, number>());
    cells.forEach(({ row, col }, i) => {
      const u = row * side + col;
      hits[u] = (hits[u] as number) + 1;
      const cls = Math.round(ctx.data.y[i] as number);
      votes[u]!.set(cls, (votes[u]!.get(cls) ?? 0) + 1);
    });
    const majority = votes.map((v) => {
      let best = -1;
      let count = 0;
      v.forEach((c, k) => {
        if (c > count) {
          count = c;
          best = k;
        }
      });
      return best;
    });
    return {
      som,
      side,
      hits,
      majority,
      u: uMatrix(som.weights, side, side),
      te: topographicError(som.weights, ctx.data.x, side),
      empty: hits.reduce((a, h) => a + (h === 0 ? 1 : 0), 0),
    };
  }, [ctx.params, ctx.data, side]);
}

function MapView({ ctx }: { ctx: ProjectorContext }) {
  const palette = usePalette();
  const m = useMap(ctx);
  const { side } = m;
  const cell = Math.min(46, Math.floor(480 / side));
  const size = cell * side;
  const uMax = Math.max(...m.u, 1e-9);
  const digits = ctx.data.dimensions === 256;
  return (
    <div className="grid gap-8 lg:grid-cols-[auto_minmax(0,1fr)]">
      <div className="max-w-full overflow-x-auto">
        {digits ? (
          <div
            className="grid"
            style={{ gridTemplateColumns: `repeat(${side}, ${cell}px)` }}
            role="img"
            aria-label="The picture each map unit has learned, laid out on the grid"
          >
            {Array.from({ length: side * side }, (_, u) => (
              <PixelCanvas
                key={u}
                values={Array.from({ length: 256 }, (_, j) =>
                  Math.min(1, Math.max(0, m.som.weights.get(u, j))),
                )}
                side={16}
                size={cell}
                ariaLabel={`Unit ${u + 1}`}
              />
            ))}
          </div>
        ) : (
          <svg
            width={size}
            height={size}
            viewBox={`0 0 ${size} ${size}`}
            role="img"
            aria-label="U-matrix: shading shows how different each unit is from its neighbours; circles show how many samples it won"
          >
            {Array.from({ length: side * side }, (_, u) => {
              const r = Math.floor(u / side);
              const c = u % side;
              const hit = m.hits[u] as number;
              const maxHits = Math.max(...m.hits, 1);
              const cls = m.majority[u] as number;
              return (
                <g key={u}>
                  <rect
                    x={c * cell}
                    y={r * cell}
                    width={cell}
                    height={cell}
                    fill={palette.css.fg}
                    fillOpacity={0.05 + 0.5 * ((m.u[u] as number) / uMax)}
                    stroke={palette.css.surface}
                  />
                  {hit > 0 && (
                    <circle
                      cx={c * cell + cell / 2}
                      cy={r * cell + cell / 2}
                      r={3 + (cell / 2 - 5) * Math.sqrt(hit / maxHits)}
                      fill={
                        ctx.data.colour === 'classes'
                          ? palette.css.classes[cls % palette.css.classes.length]
                          : palette.css.brand
                      }
                      opacity={0.85}
                    />
                  )}
                </g>
              );
            })}
          </svg>
        )}
      </div>
      <div className="min-w-0 space-y-3 text-sm text-muted" data-testid="som-note">
        <p>
          <strong className="text-fg">Topographic error {pct(m.te)}</strong>: the share of samples
          whose two best-matching units are not neighbours on the grid. Low means the grid order
          really reflects the data.{' '}
          {m.empty > 0 && (
            <>
              <strong className="text-fg">{m.empty}</strong> of {side * side} units won no sample at
              all: they sit between clusters, stretching the sheet.
            </>
          )}
        </p>
        {digits ? (
          <p>
            Each tile is the picture its unit has learned. Neighbouring tiles look alike, and digits
            of the same kind collect in the same region.
          </p>
        ) : (
          <p>
            Shading is the <em>U-matrix</em>: how different a unit is from its neighbours. Dark
            ridges mark borders between groups; circles show how many samples each unit won
            {ctx.data.colour === 'classes' ? ' and the class that wins most often there' : ''}.
          </p>
        )}
      </div>
    </div>
  );
}

function Planes({ ctx }: { ctx: ProjectorContext }) {
  const palette = usePalette();
  const m = useMap(ctx);
  const [feature, setFeature] = useState(0);
  const d = ctx.data.dimensions;
  if (d > 13) {
    return (
      <p className="text-sm text-muted">
        Component planes need a handful of features: pick Iris, Wine or one of the 3-D surfaces.
      </p>
    );
  }
  const f = Math.min(feature, d - 1);
  const values = Array.from({ length: m.side * m.side }, (_, u) => m.som.weights.get(u, f));
  const lo = Math.min(...values);
  const hi = Math.max(...values);
  const cell = Math.min(40, Math.floor(420 / m.side));
  return (
    <div className="space-y-4">
      <Segmented
        label="Feature"
        size="sm"
        fit
        value={String(f)}
        onValueChange={(v) => setFeature(Number(v))}
        options={Array.from({ length: d }, (_, j) => ({ value: String(j), label: `x${j + 1}` }))}
      />
      <div className="max-w-full overflow-x-auto">
        <svg
          width={cell * m.side}
          height={cell * m.side}
          role="img"
          aria-label={`Value of feature x${f + 1} stored in each map unit`}
        >
          {values.map((v, u) => (
            <rect
              key={u}
              x={(u % m.side) * cell}
              y={Math.floor(u / m.side) * cell}
              width={cell}
              height={cell}
              fill={palette.css.brand}
              fillOpacity={0.08 + 0.85 * ((v - lo) / (hi - lo || 1))}
              stroke={palette.css.surface}
            />
          ))}
        </svg>
      </div>
      <p className="text-sm text-muted">
        One slice of what every unit has learned: how large feature x{f + 1} is in that unit&apos;s
        prototype (from {fix(lo, 2)} to {fix(hi, 2)}). A smooth gradient across the map is the
        signature of a well-ordered SOM; comparing planes shows which features move together.
      </p>
    </div>
  );
}

export default function SomPage() {
  return (
    <ProjectorLab
      spec={somSpec}
      iterative
      text={{
        eyebrow: 'Classic · unsupervised',
        title: 'Self-organising map',
        accent: 'a sheet that learns the data',
        blurb:
          'A grid of units is pulled towards the data, winners and their neighbours together. Watch the sheet unfold, then read the map: who landed where, and where the borders are.',
      }}
      extras={[
        { id: 'lab-map', title: 'Reading the map', render: (ctx) => <MapView ctx={ctx} /> },
        { id: 'lab-planes', title: 'Component planes', render: (ctx) => <Planes ctx={ctx} /> },
      ]}
      howItWorks={
        <div className="space-y-3 text-sm text-muted">
          <p>
            Every grid cell holds a weight vector in data space. For each sample the closest unit
            (the <em>best-matching unit</em>) wins, and it and its grid neighbours move towards the
            sample, the winner most and distant neighbours barely at all. The learning rate and the
            neighbourhood radius shrink as training goes on: first the whole sheet orders itself,
            then the units fine-tune.
          </p>
          <p>
            Because neighbours are always dragged together, the grid ends up a topology-preserving
            picture of the data. Two numbers judge it: the{' '}
            <strong className="text-fg">quantization error</strong> (how far samples are from their
            unit: fit) and the <strong className="text-fg">topographic error</strong> (how often the
            two closest units are not adjacent: order).
          </p>
        </div>
      }
    />
  );
}
