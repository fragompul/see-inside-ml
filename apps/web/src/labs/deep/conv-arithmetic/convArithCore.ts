import {
  analyseStack,
  conv,
  outputSize,
  convParameters,
  pool,
  samePadding,
  tracedField,
  type StackStats,
  type Window,
} from '@sim/deep';

export type Design = 'plain' | 'pooled' | 'strided' | 'dilated';

export const DESIGN_LABEL: Record<Design, string> = {
  plain: 'Plain: same-size convolutions',
  pooled: 'Pooling after every convolution',
  strided: 'Stride 2 on every second convolution',
  dilated: 'Dilation doubling at each layer',
};

export interface ConvArithScenario {
  design: Design;
  depth: number;
  kernel: number;
  input: number;
}

export const DEFAULT_CONV_ARITH: ConvArithScenario = {
  design: 'plain',
  depth: 3,
  kernel: 3,
  input: 64,
};

export const CHANNELS = 16;

export interface NamedLayer {
  name: string;
  window: Window;
}

export function buildStack(s: ConvArithScenario): NamedLayer[] {
  const out: NamedLayer[] = [];
  let size = s.input;
  const push = (name: string, window: Window) => {
    out.push({ name, window });
    size = outputSize(size, window);
  };
  for (let i = 0; i < s.depth; i++) {
    const n = i + 1;
    if (s.design === 'dilated') {
      const d = 2 ** i;
      push(`conv ${n} (dilation ${d})`, conv(s.kernel, 1, samePadding(s.kernel, d), d));
    } else if (s.design === 'strided' && i % 2 === 1 && size >= 4) {
      push(`conv ${n} (stride 2)`, conv(s.kernel, 2, samePadding(s.kernel)));
    } else {
      push(`conv ${n}`, conv(s.kernel, 1, samePadding(s.kernel)));
    }
    // pooling stops once the picture is down to a few values
    if (s.design === 'pooled' && size >= 4) push(`pool ${n}`, pool(2));
  }
  return out;
}

export interface ConvArithRun {
  scenario: ConvArithScenario;
  layers: NamedLayer[];
  stats: StackStats;
  /** The receptive field found by tracing dependencies back; equals the formula whenever the input is large enough to hold it. */
  traced: number;
  finalSize: number;
  finalField: number;
  /** Share of the input the central output value can see. */
  coverage: number;
  parameters: number;
  /** Weights of one convolution wide enough to see as much (stride 1, same channels). */
  singleKernelParameters: number;
}

export function runConvArith(s: ConvArithScenario): ConvArithRun {
  const layers = buildStack(s);
  const stats = analyseStack(
    s.input,
    layers.map((l) => l.window),
  );
  const finalField = stats.field[stats.field.length - 1] as number;
  const convs = layers.filter((l) => l.window.kind === 'conv');
  return {
    scenario: s,
    layers,
    stats,
    traced: tracedField(
      s.input,
      layers.map((l) => l.window),
    ),
    finalSize: stats.sizes[stats.sizes.length - 1] as number,
    finalField,
    coverage: Math.min(1, finalField / s.input),
    parameters: convs.reduce(
      (a, l) => a + convParameters(l.window.kernel, CHANNELS, CHANNELS, 2),
      0,
    ),
    singleKernelParameters: convParameters(finalField, CHANNELS, CHANNELS, 2),
  };
}
