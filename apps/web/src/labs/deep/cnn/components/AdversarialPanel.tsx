import { IMAGE_SIZE, argmax, attack, robustAccuracy, type AttackMethod } from '@sim/deep';
import { Gauge } from 'lucide-react';
import { useMemo, useState } from 'react';
import { Button, Segmented, Select, Slider } from '@/design-system';
import { usePalette } from '@/shared/canvas/colors';
import { MiniChart } from '@/shared/components/MiniChart';
import { PixelCanvas } from '@/shared/components/PixelCanvas';
import { cnnSession, effectiveImage, useCnnStore } from '../store';

const BUDGETS = [0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4];

/** Nudge the picture by less than the eye notices and watch the answer change. */
export function AdversarialPanel() {
  const palette = usePalette();
  const image = useCnnStore((s) => s.image);
  const center = useCnnStore((s) => s.center);
  const source = useCnnStore((s) => s.source);
  const sampleLabel = useCnnStore((s) => s.sampleLabel);
  const version = useCnnStore((s) => s.version);
  const trace = useCnnStore((s) => s.trace);
  const [method, setMethod] = useState<AttackMethod>('fgsm');
  const [epsilon, setEpsilon] = useState(0.12);
  const [target, setTarget] = useState('any');
  const [curve, setCurve] = useState<{ fgsm: number[]; pgd: number[] } | null>(null);
  const [measuring, setMeasuring] = useState(false);

  const input = useMemo(() => effectiveImage(image, center), [image, center]);
  const predicted = trace ? argmax(trace.probs) : 0;
  const label = sampleLabel ?? predicted;

  const result = useMemo(() => {
    if (source === 'blank') return null;
    const net = cnnSession().network;
    return attack(net, input, label, {
      method,
      epsilon,
      steps: 12,
      ...(target === 'any' ? {} : { targetClass: Number(target) }),
    });
    // `version` bumps when the weights change while training
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [input, label, method, epsilon, target, source, version]);

  const measure = () => {
    setMeasuring(true);
    // let the button repaint before the (second-long) loop runs
    window.setTimeout(() => {
      const s = cnnSession();
      const n = Math.min(120, s.testCount);
      const imgs = new Float64Array(n * IMAGE_SIZE * IMAGE_SIZE);
      const labels = new Int32Array(n);
      for (let i = 0; i < n; i++) {
        const sample = s.sample(i, 'test');
        imgs.set(sample.image, i * IMAGE_SIZE * IMAGE_SIZE);
        labels[i] = sample.label;
      }
      const run = (m: AttackMethod) =>
        BUDGETS.map((e) =>
          robustAccuracy(s.network, imgs, labels, { method: m, epsilon: e, steps: 8 }),
        );
      setCurve({ fgsm: run('fgsm'), pgd: run('pgd') });
      setMeasuring(false);
    }, 30);
  };

  const blank = source === 'blank' || !result;
  return (
    <div className="space-y-8">
      <div className="grid gap-x-10 gap-y-5 sm:grid-cols-2 lg:grid-cols-3">
        <Segmented<AttackMethod>
          label="Attack"
          size="sm"
          fit
          value={method}
          onValueChange={setMethod}
          options={[
            { value: 'fgsm', label: 'FGSM (one step)' },
            { value: 'pgd', label: 'PGD (12 steps)' },
          ]}
        />
        <Slider
          label="Budget ε per pixel"
          min={0}
          max={0.5}
          step={0.01}
          value={epsilon}
          onValueChange={setEpsilon}
          format={(v) => v.toFixed(2)}
        />
        <Select
          label="Make it say"
          value={target}
          onValueChange={setTarget}
          options={[
            { value: 'any', label: 'Anything but the truth' },
            ...Array.from({ length: 10 }, (_, d) => ({
              value: String(d),
              label: `The digit ${d}`,
            })),
          ]}
        />
      </div>

      {blank ? (
        <p className="text-sm text-muted">
          Draw a digit or load a test digit above, then attack it.
        </p>
      ) : (
        <div className="space-y-5">
          <div className="flex flex-wrap items-end gap-6">
            <figure className="space-y-1.5">
              <PixelCanvas values={input} side={IMAGE_SIZE} size={120} ariaLabel="Original image" />
              <figcaption className="text-center font-mono text-[0.7rem] text-faint">
                original → {result.before.label}
              </figcaption>
            </figure>
            <span className="pb-10 font-mono text-faint">+</span>
            <figure className="space-y-1.5">
              <PixelCanvas
                values={result.perturbation}
                side={IMAGE_SIZE}
                size={120}
                mode="diverging"
                scale={Math.max(epsilon, 0.01)}
                ariaLabel="The perturbation, drawn with ember where pixels were brightened and frost where they were darkened"
              />
              <figcaption className="text-center font-mono text-[0.7rem] text-faint">
                the nudge (≤ {epsilon.toFixed(2)})
              </figcaption>
            </figure>
            <span className="pb-10 font-mono text-faint">=</span>
            <figure className="space-y-1.5">
              <PixelCanvas
                values={result.adversarial}
                side={IMAGE_SIZE}
                size={120}
                ariaLabel="Adversarial image"
              />
              <figcaption
                className="text-center font-mono text-[0.7rem] text-faint"
                data-testid="adv-answer"
              >
                attacked → {result.after.label}
              </figcaption>
            </figure>
            <p
              data-testid="adv-verdict"
              className={`pb-9 font-display text-lg font-semibold ${result.fooled ? 'text-neg' : 'text-ok'}`}
            >
              {result.fooled ? 'Fooled!' : 'Still right'}
            </p>
          </div>
          <p className="text-sm text-muted">
            Confidence in {result.after.label}:{' '}
            <strong className="text-fg">
              {((result.after.probs[result.after.label] ?? 0) * 100).toFixed(0)}%
            </strong>{' '}
            (was {((result.before.probs[result.before.label] ?? 0) * 100).toFixed(0)}% for{' '}
            {result.before.label}). The nudge moves each pixel by at most {epsilon.toFixed(2)} on a
            0 to 1 scale, in the direction that most increases the loss.
          </p>
        </div>
      )}

      <div className="space-y-4 border-t border-line pt-6">
        <div className="flex flex-wrap items-center gap-3">
          <Button size="sm" variant="secondary" onClick={measure} disabled={measuring}>
            <Gauge className="size-4" /> {measuring ? 'Testing…' : 'Test robustness'}
          </Button>
          <span className="text-[0.8rem] text-muted">
            Attacks 120 test digits at each budget with the current weights.
          </span>
        </div>
        {curve && (
          <MiniChart
            title="Accuracy under attack"
            x={BUDGETS}
            xLabel="ε (budget per pixel)"
            yRange={{ lo: 0, hi: 1 }}
            series={[
              { name: 'FGSM', color: palette.css.brand, values: curve.fgsm },
              { name: 'PGD', color: palette.css.neg, values: curve.pgd },
            ]}
            note="Accuracy collapses with a tiny budget: the network is accurate on typical digits but not robust. Iterating (PGD) is stronger than a single step (FGSM)."
            height={200}
          />
        )}
      </div>
    </div>
  );
}
