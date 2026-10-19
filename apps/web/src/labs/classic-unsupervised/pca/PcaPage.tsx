import { PCA } from '@sim/classic';
import { useMemo, useState } from 'react';
import { Slider } from '@/design-system';
import { usePalette } from '@/shared/canvas/colors';
import { MiniChart } from '@/shared/components/MiniChart';
import { PixelCanvas } from '@/shared/components/PixelCanvas';
import { ProjectorLab, fix, type ProjectorContext } from '@/shared/lab-kit';
import { pcaSpec } from './pcaSpec';

function Scree({ ctx }: { ctx: ProjectorContext }) {
  const palette = usePalette();
  const p = useMemo(() => new PCA().fit(ctx.data.x), [ctx.data]);
  const shown = Math.min(12, p.allVariance.length);
  const ratio = Array.from(p.explainedVarianceRatio).slice(0, shown);
  const cumulative = ratio.map((_, i) => ratio.slice(0, i + 1).reduce((a, b) => a + b, 0));
  const for90 = p.componentsFor(0.9);
  return (
    <div className="grid gap-8 lg:grid-cols-[minmax(0,1fr)_20rem]">
      <MiniChart
        title="Share of the variance carried by each component"
        x={ratio.map((_, i) => i + 1)}
        xLabel="component"
        yRange={{ lo: 0, hi: 1 }}
        series={[
          { name: 'each component', color: palette.css.brand, values: ratio },
          { name: 'cumulative', color: palette.css.pos, values: cumulative },
        ]}
        guide={0.9}
        height={200}
      />
      <div className="space-y-3 text-sm text-muted" data-testid="scree-note">
        <p>
          The first component is the direction along which the data vary most; the second is the
          best direction perpendicular to it, and so on. A steep fall followed by a flat tail (the
          “elbow”) says how many directions really matter.
        </p>
        <p>
          Here <strong className="text-fg">{for90}</strong> of {ctx.data.dimensions} components
          reach 90% of the variance (dashed line); the first two carry{' '}
          <strong className="text-fg">{fix(((ratio[0] ?? 0) + (ratio[1] ?? 0)) * 100, 1)}%</strong>.
        </p>
      </div>
    </div>
  );
}

function Loadings({ ctx }: { ctx: ProjectorContext }) {
  const p = useMemo(() => new PCA({ components: 6 }).fit(ctx.data.x), [ctx.data]);
  if (ctx.data.dimensions === 256) {
    return (
      <div className="space-y-4">
        <div className="flex flex-wrap gap-3">
          {Array.from({ length: 6 }, (_, c) => (
            <figure key={c} className="space-y-1">
              <PixelCanvas
                values={Array.from({ length: 256 }, (_, j) => p.components.get(c, j))}
                side={16}
                size={84}
                mode="diverging"
                scale={0.25}
                ariaLabel={`Principal component ${c + 1} drawn as an image`}
              />
              <figcaption className="text-center font-mono text-[0.68rem] text-faint">
                PC {c + 1} · {fix((p.explainedVarianceRatio[c] ?? 0) * 100, 1)}%
              </figcaption>
            </figure>
          ))}
        </div>
        <p className="text-sm text-muted">
          Each component is itself a 16 × 16 picture: ember pixels are added, frost pixels
          subtracted. Every digit is the average digit plus a weighted sum of these “eigen-digits”.
        </p>
      </div>
    );
  }
  const names =
    ctx.data.dimensions <= 13
      ? Array.from({ length: ctx.data.dimensions }, (_, j) => `x${j + 1}`)
      : [];
  return (
    <div className="space-y-4">
      <div className="overflow-x-auto">
        <table className="w-full min-w-[22rem] text-left font-mono text-[0.78rem] tabular-nums">
          <thead className="text-muted">
            <tr>
              <th className="pr-4 pb-2 font-medium">feature</th>
              {[0, 1].map((c) => (
                <th key={c} className="pr-4 pb-2 font-medium">
                  PC {c + 1}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {names.map((n, j) => (
              <tr key={n} className="border-t border-line">
                <td className="py-1 pr-4">{n}</td>
                {[0, 1].map((c) => {
                  const v = p.components.get(c, j);
                  return (
                    <td key={c} className="pr-4">
                      <span
                        className="mr-2 inline-block h-2 rounded-full align-middle"
                        style={{
                          width: `${Math.abs(v) * 70}px`,
                          background: v >= 0 ? 'var(--pos)' : 'var(--neg)',
                        }}
                      />
                      {fix(v, 2)}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="text-sm text-muted">
        A component is a recipe: how much of each original feature to add. Large entries (ember =
        positive, frost = negative) mark the features that drive that direction.
      </p>
    </div>
  );
}

function Reconstruction({ ctx }: { ctx: ProjectorContext }) {
  const [k, setK] = useState(5);
  const [index, setIndex] = useState(0);
  const pFull = useMemo(() => new PCA().fit(ctx.data.x), [ctx.data]);
  const max = Math.min(60, pFull.allVariance.length);
  const model = useMemo(() => new PCA({ components: k }).fit(ctx.data.x), [ctx.data, k]);
  if (ctx.data.dimensions !== 256) {
    return (
      <p className="text-sm text-muted">
        Reconstruction is easiest to see on images: pick the Digits dataset and slide the number of
        components.
      </p>
    );
  }
  const row = ctx.data.x.data.slice(index * 256, (index + 1) * 256);
  const back = model.inverseTransform(
    model.transform({
      ...ctx.data.x,
      rows: 1,
      data: row,
      get: (_r: number, c: number) => row[c] as number,
    } as never),
  );
  const err = model.reconstructionError({
    ...ctx.data.x,
    rows: 1,
    data: row,
    get: (_r: number, c: number) => row[c] as number,
  } as never);
  return (
    <div className="space-y-5">
      <div className="grid gap-x-10 gap-y-5 sm:grid-cols-2">
        <Slider label="Components kept" min={1} max={max} value={k} onValueChange={setK} />
        <Slider
          label="Digit"
          min={0}
          max={ctx.data.x.rows - 1}
          value={index}
          onValueChange={setIndex}
        />
      </div>
      <div className="flex flex-wrap items-end gap-8">
        <figure className="space-y-1">
          <PixelCanvas values={row} side={16} size={128} ariaLabel="The original digit" />
          <figcaption className="text-center font-mono text-[0.7rem] text-faint">
            original (256 numbers)
          </figcaption>
        </figure>
        <figure className="space-y-1">
          <PixelCanvas
            values={Array.from(back.data, (v) => Math.min(1, Math.max(0, v)))}
            side={16}
            size={128}
            ariaLabel="The digit rebuilt from the kept components"
          />
          <figcaption
            className="text-center font-mono text-[0.7rem] text-faint"
            data-testid="recon-caption"
          >
            from {k} numbers · error {fix(err, 4)}
          </figcaption>
        </figure>
      </div>
      <p className="text-sm text-muted">
        Compression in action: keep only the first {k} components (so {k} numbers per picture) and
        rebuild the picture. The more you keep, the sharper it gets; the error is what the dropped
        components carried.
      </p>
    </div>
  );
}

export default function PcaPage() {
  return (
    <ProjectorLab
      spec={pcaSpec}
      text={{
        eyebrow: 'Classic · unsupervised',
        title: 'PCA & kernel PCA',
        accent: 'the directions that matter',
        blurb:
          'Rotate the data to the directions of greatest variance and keep the first few. See how much each carries, what they look like, and where a linear view breaks down.',
      }}
      extras={[
        {
          id: 'lab-scree',
          title: 'How much each direction carries',
          render: (ctx) => <Scree ctx={ctx} />,
        },
        {
          id: 'lab-loadings',
          title: 'What the directions are',
          render: (ctx) => <Loadings ctx={ctx} />,
        },
        {
          id: 'lab-reconstruction',
          title: 'Compression and reconstruction',
          render: (ctx) => <Reconstruction ctx={ctx} />,
        },
      ]}
      howItWorks={
        <div className="space-y-3 text-sm text-muted">
          <p>
            PCA centres the data and finds the eigenvectors of its covariance matrix. Each
            eigenvector is a direction; its eigenvalue is the variance of the data along it.
            Projecting onto the first k eigenvectors is the best possible k-dimensional linear
            summary, in the sense of least squared reconstruction error.
          </p>
          <p>
            Kernel PCA does the same thing in an implicit, much larger feature space defined by a
            kernel (it eigen-decomposes the centred kernel matrix instead of the covariance), which
            lets it follow curved structure. Both keep global variance, not neighbourhoods: for
            that, see t-SNE and UMAP.
          </p>
        </div>
      }
    />
  );
}
