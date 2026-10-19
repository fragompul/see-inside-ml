import { Columns3 } from 'lucide-react';
import { useState } from 'react';
import { Button } from '@/design-system';
import {
  EmbeddingPlane,
  ProjectorLab,
  fix,
  runProjector,
  type ProjectorContext,
  type ProjectorRun,
} from '@/shared/lab-kit';
import { METHODS, makeEmbedder, manifoldSpec, type ManifoldMethod } from './manifoldSpec';

interface Entry {
  method: ManifoldMethod;
  run: ProjectorRun | null;
  error: string | null;
}

function Compare({ ctx }: { ctx: ProjectorContext }) {
  const [results, setResults] = useState<Entry[] | null>(null);
  const [busy, setBusy] = useState(false);
  const start = () => {
    setBusy(true);
    // let the button repaint before the (few-second) computation blocks the page
    window.setTimeout(() => {
      const entries: Entry[] = METHODS.map(({ value }) => {
        try {
          const model = makeEmbedder({
            method: value,
            neighbors: 10,
            perplexity: 20,
            minDist: 0.1,
            seed: 3,
          });
          return { method: value, run: runProjector(model, ctx.data), error: null };
        } catch (error) {
          return {
            method: value,
            run: null,
            error: error instanceof Error ? error.message : String(error),
          };
        }
      });
      setResults(entries);
      setBusy(false);
    }, 40);
  };
  return (
    <div className="space-y-5">
      <div className="flex flex-wrap items-center gap-3">
        <Button size="sm" variant="secondary" onClick={start} disabled={busy}>
          <Columns3 className="size-4" /> {busy ? 'Computing…' : 'Compare all six methods'}
        </Button>
        <span className="text-[0.8rem] text-muted">
          Runs every method on the current data with sensible default settings (takes a few
          seconds).
        </span>
      </div>
      {results && (
        <div className="grid grid-cols-2 gap-5 lg:grid-cols-3" data-testid="comparison">
          {results.map((e) => (
            <figure key={e.method} className="space-y-1.5">
              {e.run ? (
                <EmbeddingPlane
                  points={e.run.embedding}
                  values={ctx.data.y}
                  colour={ctx.data.colour}
                  width={260}
                  height={200}
                  ariaLabel={`${METHODS.find((m) => m.value === e.method)?.label} embedding`}
                />
              ) : (
                <div className="grid h-[200px] place-items-center rounded-xl border border-line-strong p-3 text-center text-[0.78rem] text-muted">
                  {e.error}
                </div>
              )}
              <figcaption className="font-mono text-[0.7rem] text-muted">
                <span className="text-fg">{METHODS.find((m) => m.value === e.method)?.label}</span>
                {e.run
                  ? ` · trust ${fix(e.run.scores.trustworthiness, 3)} · kept ${fix(e.run.scores.preservation, 2)}`
                  : ''}
              </figcaption>
            </figure>
          ))}
        </div>
      )}
    </div>
  );
}

const NOTES: Record<ManifoldMethod, string> = {
  mds: 'Classical MDS finds coordinates whose pairwise distances best match the original ones, in closed form. For ordinary (Euclidean) distances it is exactly PCA.',
  smacof:
    'Metric MDS minimises the stress Σ(dᵢⱼ − δᵢⱼ)² iteratively (SMACOF): every step moves each point to the average of where its distances say it should be.',
  isomap:
    'Isomap replaces straight-line distances by shortest paths through the k-neighbour graph (geodesics) and then runs classical MDS. It unrolls smooth sheets but a single shortcut edge between layers ruins it.',
  lle: 'Locally linear embedding writes every point as a weighted mix of its neighbours, then finds 2-D points that satisfy the same recipes. It keeps local geometry but can collapse or distort globally.',
  tsne: 't-SNE turns distances into neighbour probabilities and matches them in 2-D with a heavy-tailed kernel, which pushes groups apart. Distances between groups and cluster sizes in the picture are not meaningful.',
  umap: 'UMAP builds a fuzzy neighbour graph and lays it out so linked points attract and unlinked ones repel. Like t-SNE it shows clusters well; it often preserves a little more of the global layout and is faster.',
};

function Notes({ ctx }: { ctx: ProjectorContext }) {
  const method = ctx.params.method as ManifoldMethod;
  return (
    <div className="space-y-3 text-sm text-muted" data-testid="method-note">
      <p>
        <strong className="text-fg">{METHODS.find((m) => m.value === method)?.label}.</strong>{' '}
        {NOTES[method]}
      </p>
      <p>
        Judge an embedding by its neighbours, not by how pretty it is:{' '}
        <strong className="text-fg">trustworthiness</strong> penalises false neighbours (points that
        look close in the picture but are not), <strong className="text-fg">continuity</strong>{' '}
        penalises torn neighbourhoods (points close in the data that the picture separates).
      </p>
    </div>
  );
}

export default function ManifoldPage() {
  return (
    <ProjectorLab
      spec={manifoldSpec}
      iterative
      text={{
        eyebrow: 'Classic · unsupervised',
        title: 'Manifold learning',
        accent: 'MDS, Isomap, LLE, t-SNE and UMAP',
        blurb:
          'Squeeze high-dimensional data into a picture while keeping what matters. Watch the iterative methods settle, and ask whether neighbours stayed neighbours.',
      }}
      extras={[
        { id: 'lab-method', title: 'About this method', render: (ctx) => <Notes ctx={ctx} /> },
        {
          id: 'lab-compare',
          title: 'All the methods side by side',
          render: (ctx) => <Compare ctx={ctx} />,
        },
      ]}
      howItWorks={
        <div className="space-y-3 text-sm text-muted">
          <p>
            Real data often lie close to a low-dimensional surface (a manifold) curled up inside a
            high-dimensional space. Linear methods like PCA can only look at it from one side.
            Manifold learners try to recover the surface itself, each keeping a different thing: all
            distances (MDS), distances along the surface (Isomap), local linear recipes (LLE), or
            neighbour probabilities (t-SNE, UMAP).
          </p>
          <p>
            Hover a point to ring its true nearest neighbours in the original space: a faithful
            embedding keeps them close to it. The two scores in the strip measure the same thing
            over all points.
          </p>
        </div>
      }
    />
  );
}
