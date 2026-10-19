import { nearZeroEigenvalues } from '@sim/classic';
import { Matrix, plotBounds, type Table } from '@sim/core';
import { useMemo } from 'react';
import { usePalette } from '@/shared/canvas/colors';
import { MiniChart } from '@/shared/components/MiniChart';
import { ClusterLab, DataPlane, fix, num, type ClusterContext } from '@/shared/lab-kit';
import { spectralOf, spectralSpec } from './spectralSpec';

function Spectrum({ ctx }: { ctx: ClusterContext }) {
  const palette = usePalette();
  const m = spectralOf(ctx.run.model);
  const k = num(ctx.params, 'k');
  if (!m) return null;
  const shown = Array.from(m.eigenvalues).slice(0, 10);
  const pieces = nearZeroEigenvalues(m.eigenvalues, 0.02);
  return (
    <div className="grid gap-8 lg:grid-cols-[minmax(0,1fr)_20rem]">
      <MiniChart
        title="Smallest eigenvalues of the graph Laplacian"
        x={shown.map((_, i) => i + 1)}
        xLabel="eigenvalue number"
        series={[{ name: 'eigenvalue', color: palette.css.brand, values: shown }]}
        height={190}
      />
      <div className="space-y-3 text-sm text-muted" data-testid="spectrum-note">
        <p>
          If the graph falls into <em>c</em> separate pieces, exactly <em>c</em> eigenvalues are
          zero; if the pieces are only weakly linked, <em>c</em> eigenvalues are close to zero and
          then there is a jump (the eigengap).
        </p>
        <p>
          Here <strong className="text-fg">{pieces}</strong> eigenvalue
          {pieces === 1 ? ' is' : 's are'} almost zero and the next is{' '}
          {fix(m.eigenvalues[Math.min(pieces, m.eigenvalues.length - 1)])}. You asked for k = {k}.
        </p>
      </div>
    </div>
  );
}

function Embedding({ ctx }: { ctx: ClusterContext }) {
  const m = spectralOf(ctx.run.model);
  const { data } = ctx.run;
  const table = useMemo<Table | null>(() => {
    if (!m) return null;
    const e = m.embedding;
    const rows = e.rows;
    const flat = new Float64Array(rows * 2);
    for (let i = 0; i < rows; i++) {
      flat[2 * i] = e.get(i, 0);
      flat[2 * i + 1] = e.cols > 1 ? e.get(i, 1) : 0;
    }
    return { ...data, x: new Matrix(rows, 2, flat), y: Float64Array.from(ctx.run.labels) };
  }, [m, data, ctx.run.labels]);
  const bounds = useMemo(
    () => (table ? plotBounds(table.x, 0.15) : { x0: -1, x1: 1, y0: -1, y1: 1 }),
    [table],
  );
  const noTest = useMemo(() => new Uint8Array(data.x.rows), [data]);
  if (!table) return null;
  return (
    <div className="grid gap-8 lg:grid-cols-[auto_minmax(0,1fr)]">
      <DataPlane
        all={table}
        testMask={noTest}
        bounds={bounds}
        width={360}
        height={300}
        ariaLabel="The points in spectral coordinates: the first two eigenvector coordinates, after row normalisation"
      />
      <p className="text-sm text-muted">
        Same points, new coordinates: each point is described by its entries in the first
        eigenvectors of the graph Laplacian (row-normalised). Whatever tangled shape the clusters
        had before, here points of one cluster sit on top of each other, so plain k-means finishes
        the job in this space.
      </p>
    </div>
  );
}

export default function SpectralPage() {
  return (
    <ClusterLab
      spec={spectralSpec}
      text={{
        eyebrow: 'Classic · unsupervised',
        title: 'Spectral clustering',
        accent: 'clusters from the shape of a graph',
        blurb:
          'Connect each point to its neighbours, then use the eigenvectors of that graph to unroll tangled shapes into clusters that k-means can separate.',
      }}
      showRegions={false}
      extras={[
        { id: 'lab-spectrum', title: 'The spectrum', render: (ctx) => <Spectrum ctx={ctx} /> },
        {
          id: 'lab-embedding',
          title: 'The spectral embedding',
          render: (ctx) => <Embedding ctx={ctx} />,
        },
      ]}
      howItWorks={
        <div className="space-y-3 text-sm text-muted">
          <p>
            Build a similarity graph (each point linked to its nearest neighbours, or weighted by a
            Gaussian of distance). The normalised graph Laplacian L = I − D^(−1/2) W D^(−1/2)
            encodes how well connected each region is; its smallest eigenvalues and their
            eigenvectors describe the best ways to cut the graph. Using the first k eigenvectors as
            new coordinates makes clusters compact even when they are curved, and k-means then
            separates them.
          </p>
          <p>
            It excels where clusters are defined by connectivity (crescents, rings), but needs the
            graph to be built well (too few neighbours disconnects it, too many bridges the gaps)
            and costs an eigen-decomposition of an n × n matrix, so it suits a few hundred points.
          </p>
        </div>
      }
    />
  );
}
