import { IMAGE_SIZE } from '@sim/deep';
import { ArrowRight } from 'lucide-react';
import { useState, type ReactNode } from 'react';
import { Segmented } from '@/design-system';
import { Heat, maxAbs } from '@/shared/components/Heat';
import { effectiveImage, useCnnStore } from '../store';
import { PixelCanvas } from '@/shared/components/PixelCanvas';

type Stage = 'relu' | 'raw';

const cellFor = (side: number) => Math.max(3, Math.round(72 / side));

/** The maps each layer produces for the picture on the pad, with the kernels that made them. */
export function FeatureMaps() {
  const trace = useCnnStore((s) => s.trace);
  const kernels = useCnnStore((s) => s.kernels);
  const image = useCnnStore((s) => s.image);
  const center = useCnnStore((s) => s.center);
  const k = useCnnStore((s) => s.config.kernel);
  const [stage, setStage] = useState<Stage>('relu');

  if (!trace || !kernels) return null;
  const [f1, h1] = trace.shapes.conv1;
  const [, p1] = trace.shapes.pool1;
  const [f2, h2] = trace.shapes.conv2;
  const input = effectiveImage(image, center);

  const conv1 = stage === 'relu' ? trace.conv1 : trace.conv1Pre;
  const conv2 = stage === 'relu' ? trace.conv2 : trace.conv2Pre;
  const scale1 = maxAbs(conv1);
  const scalePool = maxAbs(trace.pool1);
  const scale2 = maxAbs(conv2);
  const kernelScale = maxAbs(...kernels.conv1);

  const slice = (data: Float64Array, index: number, side: number) =>
    data.subarray(index * side * side, (index + 1) * side * side);

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-center justify-between gap-4">
        <p className="max-w-2xl text-[0.9rem] text-muted">
          Each filter scans the whole picture and produces a <em>feature map</em>: where its pattern
          appears. The second layer looks for combinations of the first layer&apos;s patterns, so
          the maps get smaller and more abstract. Draw something above and watch them react.
        </p>
        <Segmented<Stage>
          label="Show feature maps"
          size="sm"
          fit
          value={stage}
          onValueChange={setStage}
          options={[
            { value: 'relu', label: 'After ReLU' },
            { value: 'raw', label: 'Before ReLU' },
          ]}
        />
      </div>

      <div className="flex flex-col gap-8 xl:flex-row xl:items-start xl:gap-5">
        <Column title="Input" caption={`${IMAGE_SIZE}×${IMAGE_SIZE}`}>
          <PixelCanvas values={input} side={IMAGE_SIZE} size={84} ariaLabel="Network input" />
        </Column>
        <Arrow />
        <Column
          title="Convolution 1"
          caption={`${f1} filters of ${k}×${k} → ${h1}×${h1} maps`}
          wide
        >
          {Array.from({ length: f1 }, (_, f) => (
            <figure key={f} className="flex flex-col items-center gap-1.5">
              <Heat
                values={kernels.conv1[f] as Float64Array}
                cols={k}
                scale={kernelScale}
                cellW={9}
                cellH={9}
                ariaLabel={`Learned ${k} by ${k} kernel of filter ${f + 1}`}
              />
              <Heat
                values={slice(conv1, f, h1)}
                cols={h1}
                scale={scale1}
                cellW={cellFor(h1)}
                cellH={cellFor(h1)}
                ariaLabel={`Feature map of filter ${f + 1} in the first convolution`}
              />
            </figure>
          ))}
        </Column>
        <Arrow />
        <Column title="Max pool" caption={`keep the strongest of each 2×2 → ${p1}×${p1}`} wide>
          {Array.from({ length: f1 }, (_, f) => (
            <Heat
              key={f}
              values={slice(trace.pool1, f, p1)}
              cols={p1}
              scale={scalePool}
              cellW={cellFor(p1)}
              cellH={cellFor(p1)}
              ariaLabel={`Pooled map ${f + 1}`}
            />
          ))}
        </Column>
        <Arrow />
        <Column
          title="Convolution 2"
          caption={`${f2} filters over all ${f1} maps → ${h2}×${h2}`}
          wide
        >
          {Array.from({ length: f2 }, (_, f) => (
            <Heat
              key={f}
              values={slice(conv2, f, h2)}
              cols={h2}
              scale={scale2}
              cellW={cellFor(h2)}
              cellH={cellFor(h2)}
              ariaLabel={`Feature map of filter ${f + 1} in the second convolution`}
            />
          ))}
        </Column>
      </div>
      <p className="text-[0.78rem] text-faint">
        Finally every value of the last maps is wired into ten output neurons (a dense layer), and a
        softmax turns their scores into the probabilities you saw above.
      </p>
    </div>
  );
}

function Column({
  title,
  caption,
  wide,
  children,
}: {
  title: string;
  caption: string;
  wide?: boolean;
  children: ReactNode;
}) {
  return (
    <div className={wide ? 'min-w-0 flex-1' : 'shrink-0'}>
      <p className="eyebrow">{title}</p>
      <p className="mb-3 text-[0.72rem] text-faint">{caption}</p>
      <div className="flex flex-wrap items-start gap-3">{children}</div>
    </div>
  );
}

function Arrow() {
  return (
    <ArrowRight
      aria-hidden
      className="hidden size-5 shrink-0 self-center text-faint xl:block"
      strokeWidth={1.5}
    />
  );
}
