import {
  NEURON_ACTIVATIONS,
  NEURON_DATASETS,
  neuronActivate,
  neuronActivationInfo,
  neuronData,
  neuronDatasetInfo,
  weightedSum,
} from '@sim/deep';
import { Wand2 } from 'lucide-react';
import { useEffect, useMemo, useRef, type PointerEvent } from 'react';
import { Button, Segmented, Slider, Tex } from '@/design-system';
import { mix, usePalette } from '@/shared/canvas/colors';
import { currentAccuracy, useFoundationsStore } from '../store';

const SIZE = 440;
const LO = -0.6;
const HI = 1.6;
const px = (v: number) => ((v - LO) / (HI - LO)) * SIZE;
const unpx = (p: number) => LO + (p / SIZE) * (HI - LO);

const fmt = (v: number) => (Math.abs(v) < 0.005 ? '0' : v.toFixed(2));

/** One neuron: weights, bias, an activation, and the line it draws through the plane. */
export function NeuronPanel() {
  const palette = usePalette();
  const { dataset, neuron, activation, best, inspected } = useFoundationsStore();
  const setNeuron = useFoundationsStore((s) => s.setNeuron);
  const setActivation = useFoundationsStore((s) => s.setActivation);
  const setDataset = useFoundationsStore((s) => s.setDataset);
  const findBest = useFoundationsStore((s) => s.findBest);
  const inspect = useFoundationsStore((s) => s.inspect);
  const ref = useRef<HTMLCanvasElement>(null);

  const data = useMemo(() => neuronData(dataset), [dataset]);
  const accuracy = currentAccuracy(dataset, neuron);

  useEffect(() => {
    const canvas = ref.current;
    const ctx = canvas?.getContext('2d');
    if (!canvas || !ctx) return;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = SIZE * dpr;
    canvas.height = SIZE * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    // what the neuron outputs over the whole plane
    const N = 44;
    const tile = document.createElement('canvas');
    tile.width = N;
    tile.height = N;
    const tctx = tile.getContext('2d');
    if (tctx) {
      const img = tctx.createImageData(N, N);
      for (let row = 0; row < N; row++) {
        for (let col = 0; col < N; col++) {
          const x = LO + ((col + 0.5) / N) * (HI - LO);
          const y = HI - ((row + 0.5) / N) * (HI - LO);
          const a = neuronActivate(activation, weightedSum(neuron, x, y));
          const t =
            activation === 'tanh' ? (a + 1) / 2 : activation === 'relu' ? Math.min(1, a / 2) : a;
          const c =
            t >= 0.5
              ? mix(palette.surface, palette.pos, (t - 0.5) * 2)
              : mix(palette.surface, palette.neg, (0.5 - t) * 2);
          const i = (row * N + col) * 4;
          img.data[i] = c[0];
          img.data[i + 1] = c[1];
          img.data[i + 2] = c[2];
          img.data[i + 3] = 255;
        }
      }
      tctx.putImageData(img, 0, 0);
      ctx.imageSmoothingEnabled = activation !== 'step';
      ctx.globalAlpha = 0.55;
      ctx.drawImage(tile, 0, 0, SIZE, SIZE);
      ctx.globalAlpha = 1;
    }

    // axes at 0 and 1
    ctx.strokeStyle = palette.line;
    ctx.globalAlpha = 0.5;
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (const v of [0, 1]) {
      ctx.moveTo(px(v), 0);
      ctx.lineTo(px(v), SIZE);
      ctx.moveTo(0, SIZE - px(v));
      ctx.lineTo(SIZE, SIZE - px(v));
    }
    ctx.stroke();
    ctx.globalAlpha = 1;

    // the decision line w1·x + w2·y + b = 0, with a short arrow towards the positive side
    const { w1, w2, b } = neuron;
    if (Math.hypot(w1, w2) > 1e-9) {
      const pts: [number, number][] = [];
      for (const x of [LO, HI]) if (Math.abs(w2) > 1e-9) pts.push([x, -(w1 * x + b) / w2]);
      for (const y of [LO, HI]) if (Math.abs(w1) > 1e-9) pts.push([-(w2 * y + b) / w1, y]);
      const inside = pts.filter(
        ([x, y]) => x >= LO - 1e-9 && x <= HI + 1e-9 && y >= LO - 1e-9 && y <= HI + 1e-9,
      );
      if (inside.length >= 2) {
        const [p, q] = inside as [[number, number], [number, number]];
        ctx.strokeStyle = palette.css.fg;
        ctx.lineWidth = 2.4;
        ctx.setLineDash([8, 6]);
        ctx.beginPath();
        ctx.moveTo(px(p[0]), SIZE - px(p[1]));
        ctx.lineTo(px(q[0]), SIZE - px(q[1]));
        ctx.stroke();
        ctx.setLineDash([]);
        const mx = (p[0] + q[0]) / 2;
        const my = (p[1] + q[1]) / 2;
        const len = Math.hypot(w1, w2);
        const tx = mx + (w1 / len) * 0.18;
        const ty = my + (w2 / len) * 0.18;
        ctx.beginPath();
        ctx.moveTo(px(mx), SIZE - px(my));
        ctx.lineTo(px(tx), SIZE - px(ty));
        ctx.stroke();
        ctx.fillStyle = palette.css.fg;
        ctx.beginPath();
        const ax = px(tx);
        const ay = SIZE - px(ty);
        const ang = Math.atan2(-(ty - my), tx - mx);
        ctx.moveTo(ax + 7 * Math.cos(ang), ay - 7 * Math.sin(ang));
        ctx.lineTo(ax + 6 * Math.cos(ang + 2.5), ay - 6 * Math.sin(ang + 2.5));
        ctx.lineTo(ax + 6 * Math.cos(ang - 2.5), ay - 6 * Math.sin(ang - 2.5));
        ctx.closePath();
        ctx.fill();
      }
    }

    // data
    for (let i = 0; i < data.count; i++) {
      const x = data.x[2 * i] as number;
      const y = data.y[i] as number;
      const yy = data.x[2 * i + 1] as number;
      const z = weightedSum(neuron, x, yy);
      const wrong = z > 0 !== (y === 1);
      ctx.beginPath();
      ctx.arc(px(x), SIZE - px(yy), data.count <= 4 ? 13 : 7, 0, Math.PI * 2);
      ctx.fillStyle = y === 1 ? palette.css.pos : palette.css.neg;
      ctx.fill();
      ctx.lineWidth = i === inspected ? 4 : wrong ? 3 : 1.5;
      ctx.strokeStyle =
        i === inspected ? palette.css.fg : wrong ? palette.css.fg : palette.css.surface;
      ctx.stroke();
      if (wrong) {
        ctx.font = `600 ${data.count <= 4 ? 14 : 9}px ui-monospace, monospace`;
        ctx.fillStyle = palette.css.surface;
        ctx.textAlign = 'center';
        ctx.fillText('×', px(x), SIZE - px(yy) + (data.count <= 4 ? 5 : 3));
      }
    }
  }, [data, neuron, activation, palette, inspected]);

  const pick = (event: PointerEvent<HTMLCanvasElement>) => {
    const rect = event.currentTarget.getBoundingClientRect();
    const x = unpx(((event.clientX - rect.left) / rect.width) * SIZE);
    const y = unpx(SIZE - ((event.clientY - rect.top) / rect.height) * SIZE);
    let nearest = -1;
    let bestD = 0.12 * 0.12 * (data.count <= 4 ? 1.8 : 0.6);
    for (let i = 0; i < data.count; i++) {
      const d = ((data.x[2 * i] as number) - x) ** 2 + ((data.x[2 * i + 1] as number) - y) ** 2;
      if (d < bestD) {
        bestD = d;
        nearest = i;
      }
    }
    inspect(nearest >= 0 ? nearest : null);
  };

  const i = inspected;
  const x1 = i === null ? 0 : (data.x[2 * i] as number);
  const x2 = i === null ? 0 : (data.x[2 * i + 1] as number);
  const z = weightedSum(neuron, x1, x2);

  return (
    <div className="space-y-6">
      <p className="max-w-3xl text-[0.9rem] text-muted">
        A neuron is the smallest unit of a neural network: it multiplies each input by a{' '}
        <em>weight</em>, adds a <em>bias</em>, and passes the total through an{' '}
        <em>activation function</em>. With two inputs that is a line drawn through the plane: one
        side says &ldquo;class 1&rdquo;, the other &ldquo;class 0&rdquo;. Move the sliders and watch
        the line tilt and shift; try to get every point on its right side.
      </p>
      <div className="space-y-2">
        <Segmented
          label="Dataset"
          size="sm"
          value={dataset}
          onValueChange={setDataset}
          options={NEURON_DATASETS.map((d) => ({ value: d, label: neuronDatasetInfo[d].label }))}
        />
        <p className="text-[0.8rem] text-muted">{neuronDatasetInfo[dataset].note}</p>
      </div>

      <div className="grid gap-8 lg:grid-cols-[auto_minmax(0,1fr)]">
        <canvas
          ref={ref}
          role="img"
          aria-label="The plane with the data points and the line drawn by the neuron. Click a point to see its arithmetic."
          style={{ width: SIZE, height: 'auto', aspectRatio: '1' }}
          className="block max-w-full cursor-pointer rounded-xl border border-line-strong"
          onPointerDown={pick}
        />
        <div className="min-w-0 space-y-5">
          <div className="grid gap-x-8 gap-y-4 sm:grid-cols-3 lg:grid-cols-1 xl:grid-cols-3">
            <Slider
              label="Weight w₁ (for x₁)"
              min={-8}
              max={8}
              step={0.1}
              value={Math.max(-8, Math.min(8, neuron.w1))}
              onValueChange={(w1) => setNeuron({ w1 })}
              format={fmt}
            />
            <Slider
              label="Weight w₂ (for x₂)"
              min={-8}
              max={8}
              step={0.1}
              value={Math.max(-8, Math.min(8, neuron.w2))}
              onValueChange={(w2) => setNeuron({ w2 })}
              format={fmt}
            />
            <Slider
              label="Bias b"
              min={-8}
              max={8}
              step={0.1}
              value={Math.max(-8, Math.min(8, neuron.b))}
              onValueChange={(b) => setNeuron({ b })}
              format={fmt}
            />
          </div>

          <div className="space-y-2">
            <Segmented
              label="Activation function"
              size="sm"
              value={activation}
              onValueChange={setActivation}
              options={NEURON_ACTIVATIONS.map((a) => ({
                value: a,
                label: neuronActivationInfo[a].label,
              }))}
            />
            <p className="text-[0.76rem] text-faint">{neuronActivationInfo[activation].note}</p>
          </div>

          <div className="rounded-xl border border-line bg-raised/30 px-4 py-3">
            <div className="overflow-x-auto text-[0.95rem]">
              <Tex tex={String.raw`a=f\left(w_1x_1+w_2x_2+b\right)`} />
            </div>
            <p
              className="mt-2 font-mono text-[0.78rem] text-muted tabular-nums"
              data-testid="accuracy"
            >
              {Math.round(accuracy * 100)}% of the points are on the right side
              {accuracy === 1 && <span className="text-ok"> · solved</span>}
            </p>
          </div>

          <div className="flex flex-wrap items-center gap-3">
            <Button size="sm" variant="secondary" onClick={findBest}>
              <Wand2 className="size-4" /> Find the best line
            </Button>
            {best && (
              <p className="text-[0.8rem] text-muted" data-testid="best-result">
                {best.accuracy === 1
                  ? 'A line separates every point: found it.'
                  : `The best any single line can do here is ${Math.round(best.accuracy * 100)}%.`}
              </p>
            )}
          </div>

          <div
            className="rounded-xl border border-line px-4 py-3 text-[0.82rem]"
            aria-label="Arithmetic of one point"
          >
            {i === null ? (
              <p className="text-muted">
                Click a point on the plane to see the arithmetic the neuron does for it.
              </p>
            ) : (
              <p className="font-mono leading-relaxed text-muted">
                point ({x1.toFixed(2)}, {x2.toFixed(2)}): z = {fmt(neuron.w1)}·{x1.toFixed(2)} +{' '}
                {fmt(neuron.w2)}·{x2.toFixed(2)} + {fmt(neuron.b)} ={' '}
                <span className="text-fg">{z.toFixed(2)}</span>
                {' → '}a = {activation}(z) ={' '}
                <span className="text-fg">{neuronActivate(activation, z).toFixed(2)}</span>
                <br />
                the neuron says <span className="text-fg">{z > 0 ? 'class 1' : 'class 0'}</span>;
                the truth is <span className="text-fg">class {data.y[i]}</span>
              </p>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
