import { HIDDEN_ACTIVATIONS, activationInfo } from '@sim/core';
import {
  LIMITS,
  type HiddenActivation,
  type HiddenLayerConfig,
  type PlaygroundConfig,
  type SceneData,
  type Snapshot,
  type StepInspection,
} from '@sim/deep';
import { Minus, Plus, X } from 'lucide-react';
import { useEffect, useMemo, useRef, useState } from 'react';
import { Tex, cn } from '@/design-system';
import { usePalette, type Palette, type Rgb } from '@/shared/canvas/colors';
import { paintNeuron } from '../canvas/heatmap';
import { buildEdges, computeLayout, edgePath, maxAbsWeight } from '../graph';
import { edgeValue, formatValue, maxEdgeValue, nodeValue } from '../inspectView';
import { NEURON_RESOLUTION } from '../store';

interface Props {
  scene: SceneData;
  snapshot: Snapshot;
  config: PlaygroundConfig;
  running: boolean;
  onPatch: (patch: Partial<PlaygroundConfig>) => void;
  /** When set, the graph shows one dissected training step instead of the live weights. */
  inspection?: StepInspection | null;
  /** Stage of the inspected step: 0 forward · 1 loss · 2 backward · 3 update. */
  stage?: number;
}

type Hover =
  { kind: 'node'; layer: number; index: number } | { kind: 'edge'; index: number } | null;

const HEIGHT = 470;
const HEADER = 92;

function NeuronCanvas({
  values,
  palette,
  tint,
  size,
}: {
  values: Float32Array | undefined;
  palette: Palette;
  tint?: Rgb;
  size: number;
}) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    if (ref.current && values) paintNeuron(ref.current, values, NEURON_RESOLUTION, palette, tint);
  }, [values, palette, tint]);
  return <canvas ref={ref} style={{ width: size, height: size }} className="block" />;
}

const sign = (v: number) => (v >= 0 ? '+' : '−');

export function NetworkGraph({
  scene,
  snapshot,
  config,
  running,
  onPatch,
  inspection = null,
  stage = 0,
}: Props) {
  const palette = usePalette();
  const wrapRef = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(720);
  const [hover, setHover] = useState<Hover>(null);
  const [pointer, setPointer] = useState({ x: 0, y: 0 });

  useEffect(() => {
    const el = wrapRef.current;
    if (!el) return;
    const observer = new ResizeObserver(([entry]) => {
      if (entry) setWidth(Math.floor(entry.contentRect.width));
    });
    observer.observe(el);
    return () => observer.disconnect();
  }, []);

  const sizes = scene.layerSizes;
  const columns = sizes.length;
  const effectiveWidth = Math.max(width, columns * 118);
  const layout = useMemo(
    () => computeLayout({ sizes, width: effectiveWidth, height: HEIGHT, headerHeight: HEADER }),
    [sizes, effectiveWidth],
  );
  const edges = useMemo(
    () => buildEdges(snapshot.weights, sizes, config.skipConnections),
    [snapshot.weights, sizes, config.skipConnections],
  );
  const maxW = maxAbsWeight(edges) || 1;
  const maxV = inspection ? maxEdgeValue(edges, inspection, stage) : 1;
  const d = layout.diameter;

  const nodePos = (layer: number, index: number) => ({
    x: layout.columnX[layer] as number,
    y: (layout.nodeY[layer]?.[index] ?? 0) as number,
  });

  const mapFor = (layer: number, index: number): Float32Array | undefined =>
    layer === 0 ? snapshot.inputMaps?.[index] : snapshot.neuronMaps?.[layer - 1]?.[index];

  const labelFor = (layer: number, index: number): string => {
    if (layer === 0) return scene.featureLabels[index] ?? `x${index + 1}`;
    if (layer === columns - 1) return scene.problem === 'classification' ? `class ${index}` : 'ŷ';
    return `L${layer} · n${index + 1}`;
  };

  const isConnected = (edge: (typeof edges)[number]) =>
    hover?.kind === 'node' &&
    ((edge.fromLayer === hover.layer && edge.fromIndex === hover.index) ||
      (edge.toLayer === hover.layer && edge.toIndex === hover.index));

  const edgeOpacity = (edge: (typeof edges)[number], i: number) => {
    const base = 0.16 + 0.7 * (Math.abs(edge.weight) / maxW);
    const skipFactor = edge.skip ? 0.55 : 1;
    if (hover?.kind === 'node') return isConnected(edge) ? 1 : 0.06;
    if (hover?.kind === 'edge') return hover.index === i ? 1 : base * 0.25;
    return base * skipFactor;
  };

  const setHidden = (hidden: HiddenLayerConfig[]) => onPatch({ hidden });
  const updateLayer = (i: number, patch: Partial<HiddenLayerConfig>) =>
    setHidden(config.hidden.map((h, k) => (k === i ? { ...h, ...patch } : h)));

  const track = (event: React.PointerEvent) => {
    const rect = wrapRef.current?.getBoundingClientRect();
    if (rect) setPointer({ x: event.clientX - rect.left, y: event.clientY - rect.top });
  };

  const nodePopover = () => {
    if (hover?.kind !== 'node') return null;
    const { layer, index } = hover;
    const isInput = layer === 0;
    const isOutput = layer === columns - 1;
    const weights = isInput ? null : snapshot.weights[layer - 1];
    const incoming = weights
      ? weights.W.map((row, i) => ({ i, w: row[index] as number })).sort(
          (a, b) => Math.abs(b.w) - Math.abs(a.w),
        )
      : [];
    const activation = isInput
      ? null
      : isOutput
        ? scene.problem === 'classification'
          ? 'softmax'
          : 'linear'
        : config.hidden[layer - 1]?.activation;
    const info = activation ? activationInfo[activation] : null;
    const src = (i: number) => {
      const prev = sizes[layer - 1] as number;
      if (config.skipConnections && layer > 1 && i >= prev)
        return scene.featureLabels[i - prev] ?? `x${i - prev + 1}`;
      return labelFor(layer - 1, i);
    };
    const strongest = incoming.slice(0, 4);
    const maxIn = Math.max(...incoming.map((r) => Math.abs(r.w)), 1e-9);
    return (
      <div className="w-60">
        <div className="flex items-center gap-3">
          <div className="overflow-hidden rounded-lg border border-line-strong">
            <NeuronCanvas
              values={mapFor(layer, index)}
              palette={palette}
              size={72}
              tint={
                isOutput && scene.problem === 'classification'
                  ? (palette.classes[index % 5] as Rgb)
                  : undefined
              }
            />
          </div>
          <div className="min-w-0">
            <p className="font-display text-sm font-semibold">
              {isInput
                ? `Feature ${labelFor(0, index)}`
                : isOutput
                  ? `Output ${labelFor(layer, index)}`
                  : `Hidden ${layer} · neuron ${index + 1}`}
            </p>
            <p className="text-[0.7rem] leading-snug text-muted">
              {isInput
                ? 'What this input looks like across the plane.'
                : 'What this neuron responds to across the plane.'}
            </p>
          </div>
        </div>
        {inspection && (
          <div className="mt-3 space-y-1 border-t border-line pt-3 font-mono text-[0.7rem]">
            <p className="eyebrow">In this step</p>
            {!isInput && (
              <p className="flex justify-between">
                <span className="text-muted">z (before f)</span>
                <span>{formatValue(inspection.preActivations[layer - 1]?.[index] ?? 0)}</span>
              </p>
            )}
            <p className="flex justify-between">
              <span className="text-muted">a (output)</span>
              <span>{formatValue(inspection.activations[layer]?.[index] ?? 0)}</span>
            </p>
            {!isInput && (
              <p className="flex justify-between">
                <span className="text-muted">δ = ∂L/∂z</span>
                <span className="text-brand">
                  {formatValue(inspection.deltas[layer - 1]?.[index] ?? 0)}
                </span>
              </p>
            )}
          </div>
        )}
        {info && weights && (
          <div className="mt-3 space-y-2 border-t border-line pt-3">
            <div className="flex items-baseline justify-between gap-2 text-[0.72rem]">
              <span className="text-muted">{info.label}</span>
              <span className="text-fg">
                <Tex
                  tex={`a = ${activation === 'softmax' ? 'softmax' : 'f'}\\!\\left(\\textstyle\\sum_i w_i x_i + b\\right)`}
                />
              </span>
            </div>
            <div className="flex justify-between font-mono text-[0.7rem]">
              <span className="text-muted">bias b</span>
              <span className={weights.b[index]! >= 0 ? 'text-pos' : 'text-neg'}>
                {sign(weights.b[index] as number)}
                {Math.abs(weights.b[index] as number).toFixed(3)}
              </span>
            </div>
            <ul className="space-y-1">
              {strongest.map(({ i, w }) => (
                <li key={i} className="flex items-center gap-2 font-mono text-[0.68rem]">
                  <span className="w-14 shrink-0 truncate text-muted">{src(i)}</span>
                  <span className="relative h-1.5 flex-1 overflow-hidden rounded-full bg-line">
                    <span
                      className={cn(
                        'absolute inset-y-0 rounded-full',
                        w >= 0 ? 'bg-pos' : 'bg-neg',
                      )}
                      style={{
                        width: `${(Math.abs(w) / maxIn) * 100}%`,
                        [w >= 0 ? 'left' : 'right']: 0,
                      }}
                    />
                  </span>
                  <span
                    className={cn('w-12 text-right tabular-nums', w >= 0 ? 'text-pos' : 'text-neg')}
                  >
                    {sign(w)}
                    {Math.abs(w).toFixed(2)}
                  </span>
                </li>
              ))}
            </ul>
            {incoming.length > strongest.length && (
              <p className="text-[0.65rem] text-faint">
                +{incoming.length - strongest.length} more incoming weights
              </p>
            )}
            {info.note && <p className="text-[0.68rem] leading-snug text-muted">{info.note}</p>}
          </div>
        )}
      </div>
    );
  };

  const hoveredEdge = hover?.kind === 'edge' ? edges[hover.index] : undefined;

  return (
    <div className="relative">
      <div className="overflow-x-auto overflow-y-hidden" ref={wrapRef}>
        <div
          className="relative"
          style={{ width: effectiveWidth, height: HEIGHT }}
          onPointerMove={track}
          onPointerLeave={() => setHover(null)}
        >
          {/* Connections */}
          <svg className="absolute inset-0" width={effectiveWidth} height={HEIGHT} aria-hidden>
            {edges.map((edge, i) => {
              const a = nodePos(edge.fromLayer, edge.fromIndex);
              const b = nodePos(edge.toLayer, edge.toIndex);
              const path = edgePath(a.x + d / 2, a.y, b.x - d / 2, b.y);
              // Live view: weight sign and size. Inspect view: the stage's own quantity.
              const value = inspection ? edgeValue(edge, inspection, stage) : edge.weight;
              const strength = inspection ? Math.abs(value) / maxV : Math.abs(edge.weight) / maxW;
              const color = value >= 0 ? palette.css.pos : palette.css.neg;
              const opacity = inspection
                ? hover?.kind === 'node'
                  ? isConnected(edge)
                    ? 1
                    : 0.05
                  : hover?.kind === 'edge'
                    ? hover.index === i
                      ? 1
                      : 0.08 + 0.2 * strength
                    : 0.1 + 0.85 * strength
                : edgeOpacity(edge, i);
              const flowing = inspection
                ? stage !== 3 && strength > 0.3
                : running && strength > 0.3;
              const backwards = inspection !== null && stage === 2;
              return (
                <g key={i}>
                  <path
                    d={path}
                    fill="none"
                    stroke={color}
                    strokeWidth={0.5 + 3.6 * strength}
                    strokeDasharray={edge.skip ? '5 5' : undefined}
                    strokeLinecap="round"
                    opacity={opacity}
                    style={{ transition: 'opacity 0.2s, stroke-width 0.3s' }}
                  />
                  {flowing && !edge.skip && (
                    <path
                      d={path}
                      fill="none"
                      className={backwards ? 'flow-edge flow-back' : 'flow-edge'}
                      stroke={backwards ? palette.css.brand : color}
                      strokeWidth={1.4 + 1.6 * strength}
                      opacity={hover ? 0.1 : 0.9}
                    />
                  )}
                  <path
                    d={path}
                    fill="none"
                    stroke="transparent"
                    strokeWidth={9}
                    style={{ pointerEvents: 'stroke', cursor: 'crosshair' }}
                    onPointerEnter={() => setHover({ kind: 'edge', index: i })}
                  />
                </g>
              );
            })}
          </svg>

          {/* Column headers */}
          {sizes.map((_, c) => {
            const isInput = c === 0;
            const isOutput = c === columns - 1;
            const layerConfig = config.hidden[c - 1];
            return (
              <div
                key={c}
                className="absolute top-3 flex -translate-x-1/2 flex-col items-center gap-1.5"
                style={{
                  left: layout.columnX[c],
                  width: Math.min(124, effectiveWidth / columns - 6),
                }}
              >
                <p className="eyebrow text-center">
                  {isInput ? 'Features' : isOutput ? 'Output' : `Hidden ${c}`}
                </p>
                {layerConfig ? (
                  <>
                    <div className="flex items-center gap-1 rounded-full border border-line-strong bg-raised/70 p-0.5">
                      <button
                        aria-label={`Remove a neuron from hidden layer ${c}`}
                        disabled={layerConfig.neurons <= LIMITS.neurons[0]}
                        onClick={() => updateLayer(c - 1, { neurons: layerConfig.neurons - 1 })}
                        className="grid size-5 cursor-pointer place-items-center rounded-full text-muted hover:bg-brand-soft hover:text-fg disabled:opacity-30"
                      >
                        <Minus className="size-3" />
                      </button>
                      <span className="w-5 text-center font-mono text-[0.72rem] tabular-nums">
                        {layerConfig.neurons}
                      </span>
                      <button
                        aria-label={`Add a neuron to hidden layer ${c}`}
                        disabled={layerConfig.neurons >= LIMITS.neurons[1]}
                        onClick={() => updateLayer(c - 1, { neurons: layerConfig.neurons + 1 })}
                        className="grid size-5 cursor-pointer place-items-center rounded-full text-muted hover:bg-brand-soft hover:text-fg disabled:opacity-30"
                      >
                        <Plus className="size-3" />
                      </button>
                    </div>
                    <div className="flex items-center gap-1">
                      <select
                        aria-label={`Activation of hidden layer ${c}`}
                        value={layerConfig.activation}
                        onChange={(e) =>
                          updateLayer(c - 1, { activation: e.target.value as HiddenActivation })
                        }
                        className="h-6 max-w-[92px] cursor-pointer appearance-none rounded-md border border-line-strong bg-raised px-1.5 text-center font-mono text-[0.66rem] text-fg"
                      >
                        {HIDDEN_ACTIVATIONS.map((a) => (
                          <option key={a} value={a}>
                            {activationInfo[a].label}
                          </option>
                        ))}
                      </select>
                      <button
                        aria-label={`Remove hidden layer ${c}`}
                        onClick={() => setHidden(config.hidden.filter((_, k) => k !== c - 1))}
                        className="grid size-5 shrink-0 cursor-pointer place-items-center rounded-full text-faint hover:bg-brand-soft hover:text-danger"
                      >
                        <X className="size-3" />
                      </button>
                    </div>
                  </>
                ) : (
                  <p className="font-mono text-[0.66rem] text-faint">
                    {isInput
                      ? `${sizes[0]} input${sizes[0] === 1 ? '' : 's'}`
                      : scene.problem === 'classification'
                        ? 'softmax'
                        : 'linear'}
                  </p>
                )}
              </div>
            );
          })}

          {/* Neurons */}
          {sizes.map((count, layer) =>
            Array.from({ length: count }, (_, index) => {
              const { x, y } = nodePos(layer, index);
              const isOutput = layer === columns - 1;
              const tint =
                isOutput && scene.problem === 'classification'
                  ? (palette.classes[index % 5] as Rgb)
                  : undefined;
              const active =
                hover?.kind === 'node' && hover.layer === layer && hover.index === index;
              return (
                <button
                  key={`${layer}-${index}`}
                  type="button"
                  aria-label={`${labelFor(layer, index)}`}
                  onPointerEnter={() => setHover({ kind: 'node', layer, index })}
                  onFocus={() => setHover({ kind: 'node', layer, index })}
                  onBlur={() => setHover(null)}
                  className={cn(
                    'absolute cursor-pointer overflow-hidden rounded-full border-2 bg-surface p-0 transition-[transform,box-shadow] duration-200',
                    active ? 'scale-125 shadow-[0_0_0_4px_var(--brand-soft)]' : '',
                  )}
                  style={{
                    left: x - d / 2,
                    top: y - d / 2,
                    width: d,
                    height: d,
                    borderColor: tint
                      ? palette.css.classes[index % 5]
                      : active
                        ? 'var(--brand)'
                        : 'var(--line-strong)',
                  }}
                >
                  <NeuronCanvas
                    values={mapFor(layer, index)}
                    palette={palette}
                    tint={tint}
                    size={d - 4}
                  />
                </button>
              );
            }),
          )}

          {/* Step inspector: numbers on the neurons and the target marker */}
          {inspection &&
            sizes.map((count, layer) =>
              Array.from({ length: count }, (_, index) => {
                const value = nodeValue(layer, index, inspection, stage);
                const { x, y } = nodePos(layer, index);
                const isTarget =
                  layer === columns - 1 &&
                  stage === 1 &&
                  scene.problem === 'classification' &&
                  inspection.targets[index] === 1;
                return (
                  <div key={`v-${layer}-${index}`} className="pointer-events-none absolute z-10">
                    {isTarget && (
                      <span
                        className="absolute rounded-full border-2 border-dashed border-fg/70"
                        style={{
                          left: x - d / 2 - 6,
                          top: y - d / 2 - 6,
                          width: d + 12,
                          height: d + 12,
                        }}
                      />
                    )}
                    {value !== null && (
                      <span
                        className={cn(
                          'absolute -translate-x-1/2 -translate-y-1/2 rounded-md bg-bg/85 px-1 py-px font-mono whitespace-nowrap text-[9.5px] leading-tight font-medium shadow-sm backdrop-blur-sm',
                          value >= 0 ? 'text-pos' : 'text-neg',
                        )}
                        style={{ left: x, top: y }}
                      >
                        {formatValue(value)}
                      </span>
                    )}
                  </div>
                );
              }),
            )}

          {/* Side captions for inputs / outputs */}
          {Array.from({ length: sizes[0] as number }, (_, i) => (
            <span
              key={`in-${i}`}
              className="pointer-events-none absolute -translate-y-1/2 text-right font-mono text-[0.68rem] text-muted"
              style={{
                left: 2,
                width: (layout.columnX[0] as number) - d / 2 - 8,
                top: nodePos(0, i).y,
              }}
            >
              {scene.featureLabels[i]}
            </span>
          ))}
          {Array.from({ length: sizes[columns - 1] as number }, (_, k) => (
            <span
              key={`out-${k}`}
              className="pointer-events-none absolute -translate-y-1/2 font-mono text-[0.66rem] text-muted"
              style={{
                left: (layout.columnX[columns - 1] as number) + d / 2 + 7,
                top: nodePos(columns - 1, k).y,
              }}
            >
              {scene.problem === 'classification' ? (
                <span style={{ color: palette.css.classes[k % 5] }}>●</span>
              ) : (
                'ŷ'
              )}
            </span>
          ))}

          {/* Popovers */}
          {hover?.kind === 'node' && (
            <div
              className="pointer-events-none absolute z-20 rounded-xl border border-line-strong bg-raised/95 p-3 shadow-[var(--shadow-card)] backdrop-blur"
              style={{
                left: Math.min(
                  Math.max(nodePos(hover.layer, hover.index).x + d / 2 + 18, 8),
                  effectiveWidth - 260,
                ),
                top: Math.min(Math.max(nodePos(hover.layer, hover.index).y - 40, 8), HEIGHT - 330),
              }}
            >
              {nodePopover()}
            </div>
          )}
          {hoveredEdge && (
            <div
              className="pointer-events-none absolute z-20 rounded-lg border border-line-strong bg-raised/95 px-2.5 py-1.5 font-mono text-[0.7rem] shadow-[var(--shadow-card)] backdrop-blur"
              style={{
                left: Math.min(pointer.x + 14, effectiveWidth - 190),
                top: Math.max(pointer.y - (inspection ? 120 : 44), 4),
              }}
            >
              <p className="text-muted">
                {labelFor(hoveredEdge.fromLayer, hoveredEdge.fromIndex)} →{' '}
                {labelFor(hoveredEdge.toLayer, hoveredEdge.toIndex)}
                {hoveredEdge.skip ? ' (skip)' : ''}
              </p>
              <p className={hoveredEdge.weight >= 0 ? 'text-pos' : 'text-neg'}>
                w = {sign(hoveredEdge.weight)}
                {Math.abs(hoveredEdge.weight).toFixed(4)}
              </p>
              {inspection && (
                <div className="mt-1 space-y-0.5 border-t border-line pt-1 text-muted">
                  <p>
                    input a ={' '}
                    <span className="text-fg">
                      {formatValue(
                        inspection.layerInputs[hoveredEdge.layerIndex]?.[hoveredEdge.row] ?? 0,
                      )}
                    </span>
                  </p>
                  <p>
                    δ ={' '}
                    <span className="text-brand">
                      {formatValue(
                        inspection.deltas[hoveredEdge.layerIndex]?.[hoveredEdge.col] ?? 0,
                      )}
                    </span>
                  </p>
                  <p>
                    ∂L/∂w = a·δ ={' '}
                    <span className="text-fg">
                      {formatValue(edgeValue(hoveredEdge, inspection, 2))}
                    </span>
                  </p>
                  <p>
                    Δw = −η·∂L/∂w ={' '}
                    <span className="text-fg">
                      {formatValue(edgeValue(hoveredEdge, inspection, 3))}
                    </span>
                  </p>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

/** Add / remove hidden layers (lives in the card header, next to the graph). */
export function LayerStepper({
  hidden,
  onChange,
}: {
  hidden: HiddenLayerConfig[];
  onChange: (hidden: HiddenLayerConfig[]) => void;
}) {
  const last = hidden.at(-1);
  return (
    <div className="flex items-center gap-2">
      <span className="eyebrow">Hidden layers</span>
      <div className="flex items-center gap-1 rounded-full border border-line-strong bg-raised/70 p-0.5">
        <button
          aria-label="Remove a hidden layer"
          disabled={hidden.length <= LIMITS.hiddenLayers[0]}
          onClick={() => onChange(hidden.slice(0, -1))}
          className="grid size-6 cursor-pointer place-items-center rounded-full text-muted hover:bg-brand-soft hover:text-fg disabled:opacity-30"
        >
          <Minus className="size-3.5" />
        </button>
        <span className="w-4 text-center font-mono text-xs tabular-nums">{hidden.length}</span>
        <button
          aria-label="Add a hidden layer"
          disabled={hidden.length >= LIMITS.hiddenLayers[1]}
          onClick={() =>
            onChange([
              ...hidden,
              { neurons: last?.neurons ?? 4, activation: last?.activation ?? 'tanh' },
            ])
          }
          className="grid size-6 cursor-pointer place-items-center rounded-full text-muted hover:bg-brand-soft hover:text-fg disabled:opacity-30"
        >
          <Plus className="size-3.5" />
        </button>
      </div>
    </div>
  );
}
