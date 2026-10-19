import { categoryOf } from '@sim/deep';
import { CATEGORY_COLOR } from '../colors';
import { useNlpStore } from '../store';
import { EmbeddingMap } from './EmbeddingMap';

/** The whole vocabulary on one plane, evolving as the model trains. */
export function MapPanel() {
  const { map, selected, neighbours, step } = useNlpStore();
  const select = useNlpStore((s) => s.select);
  if (!map) return null;
  const explained = map.explained.reduce((a, b) => a + b, 0);

  return (
    <div className="space-y-6">
      <p className="max-w-3xl text-[0.9rem] text-muted">
        Each word ends up as a list of numbers (its <em>embedding</em>). Here the vectors are
        squashed onto the two directions in which they vary the most, so that every word becomes a
        dot. Untrained, the dots are scattered at random. Press Train (section 05) and words that
        keep the same company drift together into families. Click any word to inspect it.
      </p>
      <div className="grid gap-8 lg:grid-cols-[minmax(0,40rem)_minmax(0,1fr)]">
        <EmbeddingMap map={map} selected={selected} onSelect={select} />
        <div className="min-w-0 space-y-4">
          <div>
            <p className="eyebrow mb-1">Selected</p>
            <p
              className="font-display text-3xl font-semibold tracking-tight"
              style={{ color: CATEGORY_COLOR[categoryOf(selected)] }}
            >
              {selected}
            </p>
          </div>
          <div>
            <p className="eyebrow mb-2">Closest words</p>
            <ol
              className="space-y-1 font-mono text-[0.84rem]"
              aria-label={`Words closest to ${selected}`}
            >
              {neighbours.slice(0, 5).map((n) => (
                <li key={n.word} className="flex items-center justify-between gap-4">
                  <button
                    type="button"
                    onClick={() => select(n.word)}
                    className="cursor-pointer text-fg underline-offset-2 hover:underline"
                  >
                    {n.word}
                  </button>
                  <span className="text-muted tabular-nums">{n.similarity.toFixed(2)}</span>
                </li>
              ))}
            </ol>
          </div>
          <p className="text-[0.76rem] leading-snug text-faint">
            This 2-D picture keeps {(explained * 100).toFixed(0)}% of how the vectors differ; the
            rest is lost in the flattening, which is why some neighbours on the map are not the true
            nearest words. {step === 0 && 'The model has not been trained yet.'}
          </p>
        </div>
      </div>
    </div>
  );
}
