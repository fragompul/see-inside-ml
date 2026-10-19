import { HIDDEN_ACTIVATIONS, Matrix, activate, activationInfo, derivative } from '@sim/core';
import { useMemo } from 'react';
import { Segmented, Tex } from '@/design-system';
import { usePalette } from '@/shared/canvas/colors';
import { MiniChart } from '@/shared/components/MiniChart';
import { useFoundationsStore } from '../store';

const XS = Array.from({ length: 241 }, (_, i) => -6 + i * 0.05);

/** The activation functions and their slopes: the slope is what backpropagation multiplies. */
export function ActivationsPanel() {
  const palette = usePalette();
  const name = useFoundationsStore((s) => s.activationView);
  const setName = useFoundationsStore((s) => s.setActivationView);

  const curves = useMemo(() => {
    const z = new Matrix(1, XS.length, Float64Array.from(XS));
    const a = activate(name, z);
    const slope = derivative(name, z, a);
    return { f: Array.from(a.data), df: Array.from(slope.data) };
  }, [name]);

  const maxSlope = Math.max(...curves.df);
  const tenLayers = maxSlope ** 10;
  const info = activationInfo[name];

  return (
    <div className="space-y-6">
      <p className="max-w-3xl text-[0.9rem] text-muted">
        Without an activation function a stack of neurons would just be one big linear function. The
        bend it adds is what lets networks learn curved boundaries. But there is a price: when
        training, the error is passed back through every activation multiplied by its <em>slope</em>
        , so the shape of that slope decides whether learning signals survive or fade.
      </p>
      <div className="-mx-1 overflow-x-auto px-1 pb-1">
        <Segmented
          className="min-w-[34rem]"
          label="Activation function"
          size="sm"
          value={name}
          onValueChange={setName}
          options={HIDDEN_ACTIVATIONS.map((a) => ({ value: a, label: activationInfo[a].label }))}
        />
      </div>
      <div className="grid gap-8 lg:grid-cols-[minmax(0,1fr)_minmax(0,22rem)]">
        <MiniChart
          title={`${info.label}: output and slope`}
          x={XS.map((v) => Number(v.toFixed(2)))}
          xLabel="z"
          height={240}
          series={[
            { name: 'f(z)', color: palette.css.brand, values: curves.f },
            { name: "f'(z) slope", color: palette.css.classes[0] as string, values: curves.df },
          ]}
          guide={0}
          note={info.note}
        />
        <div className="space-y-4">
          <div className="overflow-x-auto rounded-xl border border-line bg-raised/30 px-4 py-3 text-[1rem]">
            <Tex tex={`f(z)=${info.formula}`} />
          </div>
          <dl className="space-y-3 rounded-xl border border-line px-4 py-3 text-[0.84rem]">
            <div>
              <dt className="text-[0.7rem] text-faint">Steepest slope</dt>
              <dd className="font-mono text-lg text-fg tabular-nums" data-testid="max-slope">
                {maxSlope.toFixed(2)}
              </dd>
            </div>
            <div>
              <dt className="text-[0.7rem] text-faint">Signal left after ten layers, at best</dt>
              <dd className="font-mono text-lg text-fg tabular-nums" data-testid="ten-layers">
                {tenLayers >= 0.01 ? tenLayers.toFixed(2) : tenLayers.toExponential(1)}
              </dd>
            </div>
          </dl>
          <p className="text-[0.76rem] leading-snug text-faint">
            Slopes below 1 shrink the error signal at every layer it passes through, so networks
            built from sigmoids learn very slowly in their early layers. This is why ReLU, whose
            slope is exactly 1 for positive inputs, replaced them.
          </p>
        </div>
      </div>
    </div>
  );
}
