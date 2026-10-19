import { createRng } from '@sim/core';
import {
  AutoencoderSession,
  DEFAULT_AE_LAB,
  IMAGE_SIZE,
  type AutoencoderEpochStats,
  type AutoencoderLabConfig,
} from '@sim/deep';
import { create } from 'zustand';
import { inkStats, normalizeDrawing, stroke } from '@/shared/canvas/draw';

export const SPEEDS = [1, 2, 4, 8] as const;
export const IMAGINED = 12;
export const STRIP_STEPS = 9;
/** While training, the views are republished at most this often (and at the end of an epoch). */
const PUBLISH_EVERY_MS = 100;

const PIXELS = IMAGE_SIZE * IMAGE_SIZE;

export type ImageSource = 'blank' | 'drawn' | 'sample';

export interface Latents {
  mu: Float64Array;
  labels: Uint8Array;
  count: number;
}

export interface Pair {
  a: { image: Float64Array; label: number };
  b: { image: Float64Array; label: number };
}

interface AeState {
  config: AutoencoderLabConfig;
  history: AutoencoderEpochStats[];
  epoch: number;
  progress: number;
  running: boolean;
  speed: number;

  /** What is on the pad (raw), where it came from, and how the network reads it. */
  image: Float64Array;
  source: ImageSource;
  sampleLabel: number | null;
  center: boolean;
  /** Gaussian noise added to the input before encoding (to see a denoiser at work). */
  inputNoise: number;
  /** The image after centring and noise: exactly what the encoder receives. */
  input: Float64Array;
  code: { mu: Float64Array; logvar: Float64Array | null } | null;
  recon: Float64Array | null;

  /** A point of the latent space to decode, and the picture it gives. */
  probe: Float64Array;
  probeImage: Float64Array | null;
  scatter: Latents | null;

  pair: Pair | null;
  /** `STRIP_STEPS` decoded images from the code of `pair.a` to the code of `pair.b`. */
  strip: Float64Array | null;
  /** `IMAGINED` images decoded from codes drawn from the prior N(0, I). */
  imagined: Float64Array | null;

  notice: string | null;
  version: number;

  init: () => void;
  beginStroke: (x: number, y: number) => void;
  moveStroke: (x: number, y: number) => void;
  endStroke: () => void;
  clear: () => void;
  loadSample: (index: number) => void;
  loadDigit: (digit: number) => void;
  randomSample: () => void;
  setInputNoise: (level: number) => void;
  setCenter: (center: boolean) => void;
  setProbe: (index: number, value: number) => void;
  setProbePlane: (dimX: number, dimY: number, x: number, y: number) => void;
  probeToImage: () => void;
  setPairDigit: (which: 'a' | 'b', digit: number | null) => void;
  imagine: () => void;
  /** Decodes `count` codes (`count × latent`) with the current network. */
  decodeMany: (codes: Float64Array, count: number) => Float64Array;
  patch: (patch: Partial<AutoencoderLabConfig>) => void;
  setRunning: (running: boolean) => void;
  toggleRunning: () => void;
  stepEpoch: () => void;
  reset: () => void;
  setSpeed: (speed: number) => void;
}

let session: AutoencoderSession | null = null;
let rafId = 0;
let lastPoint: [number, number] | null = null;
let lastPublish = 0;
let imagineSeed = 1;
let imagineCodes = new Float64Array(0);
/** A fixed noise pattern, so the noisy input does not flicker while a slider moves. */
const NOISE = (() => {
  const rng = createRng(2024);
  return Float64Array.from({ length: PIXELS }, () => rng.normal());
})();

const blankImage = () => new Float64Array(PIXELS);

/** Centres/scales a drawing and adds the fixed noise pattern at the given strength. */
export function networkInput(image: Float64Array, center: boolean, noise: number): Float64Array {
  const base = center ? normalizeDrawing(image) : image;
  if (noise <= 0) return base;
  return base.map((v, i) => Math.min(1, Math.max(0, v + noise * (NOISE[i] as number))));
}

const hasInk = (image: Float64Array) => inkStats(image).mass > 1;

export const useAeStore = create<AeState>((set, get) => {
  const ensure = (): AutoencoderSession => {
    session ??= new AutoencoderSession(DEFAULT_AE_LAB);
    return session;
  };

  const drawCodes = () => {
    const { latent } = ensure().config;
    const rng = createRng(imagineSeed++ * 104729);
    imagineCodes = Float64Array.from({ length: IMAGINED * latent }, () => rng.normal());
  };

  const randomPair = (): Pair => {
    const s = ensure();
    const pick = () => {
      const index = Math.floor(Math.random() * s.testCount);
      return s.sample(index);
    };
    return { a: pick(), b: pick() };
  };

  /** Everything that depends on (network, image) and is cheap enough for every frame. */
  const deriveLight = (
    state: Pick<AeState, 'image' | 'center' | 'inputNoise' | 'source' | 'probe' | 'pair'>,
  ) => {
    const s = ensure();
    const input = networkInput(state.image, state.center, state.inputNoise);
    const blank = state.source === 'blank' || !hasInk(state.image);
    const code = blank ? null : s.encode(input);
    const recon = code ? s.decode(code.mu) : null;
    const latent = s.config.latent;
    let strip: Float64Array | null = null;
    if (state.pair) {
      const za = s.encode(state.pair.a.image).mu;
      const zb = s.encode(state.pair.b.image).mu;
      const codes = new Float64Array(STRIP_STEPS * latent);
      for (let k = 0; k < STRIP_STEPS; k++) {
        const t = k / (STRIP_STEPS - 1);
        for (let d = 0; d < latent; d++)
          codes[k * latent + d] = (za[d] as number) * (1 - t) + (zb[d] as number) * t;
      }
      strip = s.decodeMany(codes, STRIP_STEPS);
    }
    const probeImage =
      state.probe.length === latent ? s.decode(state.probe) : s.decode(new Float64Array(latent));
    const imagined =
      imagineCodes.length === IMAGINED * latent ? s.decodeMany(imagineCodes, IMAGINED) : null;
    return { input, code, recon, strip, probeImage, imagined };
  };

  const stopLoop = () => {
    cancelAnimationFrame(rafId);
    rafId = 0;
  };

  const frame = () => {
    const state = get();
    if (!state.running) return;
    const s = ensure();
    const completed = s.trainBatches(state.speed);
    const now = performance.now();
    // Training runs every frame; redrawing the many views is the expensive part, so it is throttled.
    if (completed.length > 0 || now - lastPublish > PUBLISH_EVERY_MS) {
      lastPublish = now;
      const current = get();
      set({
        ...deriveLight(current),
        scatter: s.latents('test'),
        history: completed.length ? [...s.history] : current.history,
        epoch: s.epoch,
        progress: s.epochFraction,
        version: current.version + 1,
      });
    }
    rafId = requestAnimationFrame(frame);
  };

  const refresh = (extra: Partial<AeState> = {}) => {
    const s = ensure();
    const state = { ...get(), ...extra };
    set({
      ...extra,
      ...deriveLight(state),
      scatter: s.latents('test'),
      history: [...s.history],
      epoch: s.epoch,
      progress: s.epochFraction,
      version: get().version + 1,
    });
  };

  return {
    config: DEFAULT_AE_LAB,
    history: [],
    epoch: 0,
    progress: 0,
    running: false,
    speed: 1,
    image: blankImage(),
    source: 'blank',
    sampleLabel: null,
    center: true,
    inputNoise: 0,
    input: blankImage(),
    code: null,
    recon: null,
    probe: new Float64Array(DEFAULT_AE_LAB.latent),
    probeImage: null,
    scatter: null,
    pair: null,
    strip: null,
    imagined: null,
    notice: null,
    version: 0,

    init: () => {
      const s = ensure();
      if (get().scatter) return;
      drawCodes();
      const first = s.sample(0);
      refresh({
        image: first.image,
        source: 'sample',
        sampleLabel: first.label,
        config: { ...s.config },
        pair: {
          a: s.sample(s.testIndicesOf(3)[0] as number),
          b: s.sample(s.testIndicesOf(8)[0] as number),
        },
      });
    },

    beginStroke: (x, y) => {
      const image = Float64Array.from(get().image);
      if (get().source !== 'drawn') image.fill(0); // drawing over a sample starts a fresh picture
      lastPoint = [x, y];
      stroke(image, [x, y], [x, y]);
      refresh({ image, source: 'drawn', sampleLabel: null });
    },

    moveStroke: (x, y) => {
      if (!lastPoint) return;
      const image = Float64Array.from(get().image);
      stroke(image, lastPoint, [x, y]);
      lastPoint = [x, y];
      refresh({ image, source: 'drawn' });
    },

    endStroke: () => {
      lastPoint = null;
    },

    clear: () => {
      lastPoint = null;
      refresh({ image: blankImage(), source: 'blank', sampleLabel: null });
    },

    loadSample: (index) => {
      const { image, label } = ensure().sample(index);
      lastPoint = null;
      refresh({ image, source: 'sample', sampleLabel: label });
    },

    loadDigit: (digit) => {
      const s = ensure();
      const indices = s.testIndicesOf(digit);
      get().loadSample(indices[Math.floor(Math.random() * indices.length)] as number);
    },

    randomSample: () => get().loadSample(Math.floor(Math.random() * ensure().testCount)),

    setInputNoise: (inputNoise) => refresh({ inputNoise }),
    setCenter: (center) => refresh({ center }),

    setProbe: (index, value) => {
      const probe = Float64Array.from(get().probe);
      if (index < 0 || index >= probe.length) return;
      probe[index] = value;
      refresh({ probe });
    },

    setProbePlane: (dimX, dimY, x, y) => {
      const probe = Float64Array.from(get().probe);
      if (dimX < probe.length) probe[dimX] = x;
      if (dimY < probe.length) probe[dimY] = y;
      refresh({ probe });
    },

    probeToImage: () => {
      const code = get().code;
      if (code) refresh({ probe: Float64Array.from(code.mu) });
    },

    setPairDigit: (which, digit) => {
      const s = ensure();
      const pair = get().pair ?? randomPair();
      const pick = () => {
        const indices = digit === null ? null : s.testIndicesOf(digit);
        const index = indices
          ? (indices[Math.floor(Math.random() * indices.length)] as number)
          : Math.floor(Math.random() * s.testCount);
        return s.sample(index);
      };
      refresh({ pair: { ...pair, [which]: pick() } });
    },

    decodeMany: (codes, count) => ensure().decodeMany(codes, count),

    imagine: () => {
      drawCodes();
      refresh();
    },

    patch: (patch) => {
      const s = ensure();
      try {
        const mode = s.update(patch);
        if (mode === 'rebuilt') stopLoop();
        const extra: Partial<AeState> = {
          config: { ...s.config },
          notice: null,
          running: mode === 'rebuilt' ? false : get().running,
        };
        if (mode === 'rebuilt') {
          extra.probe = new Float64Array(s.config.latent);
          drawCodes();
        }
        set(extra);
        refresh();
        if (mode === 'live' && get().running && !rafId) rafId = requestAnimationFrame(frame);
      } catch (error) {
        set({
          notice: error instanceof Error ? error.message : String(error),
          config: { ...s.config },
        });
      }
    },

    setRunning: (running) => {
      stopLoop();
      set({ running });
      if (running) rafId = requestAnimationFrame(frame);
    },

    toggleRunning: () => get().setRunning(!get().running),

    stepEpoch: () => {
      stopLoop();
      set({ running: false });
      ensure().step(1);
      refresh();
    },

    reset: () => {
      stopLoop();
      ensure().reset();
      set({ running: false });
      refresh();
    },

    setSpeed: (speed) => set({ speed }),
  };
});

/** Test hook: drop the module-level session so each test starts clean. */
export function resetAeSession(): void {
  cancelAnimationFrame(rafId);
  rafId = 0;
  session = null;
  lastPoint = null;
  imagineSeed = 1;
  imagineCodes = new Float64Array(0);
  useAeStore.setState({
    config: DEFAULT_AE_LAB,
    history: [],
    epoch: 0,
    progress: 0,
    running: false,
    image: blankImage(),
    source: 'blank',
    sampleLabel: null,
    inputNoise: 0,
    input: blankImage(),
    code: null,
    recon: null,
    probe: new Float64Array(DEFAULT_AE_LAB.latent),
    probeImage: null,
    scatter: null,
    pair: null,
    strip: null,
    imagined: null,
    notice: null,
    version: 0,
  });
}
