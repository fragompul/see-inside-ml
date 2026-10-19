import { readFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { act, fireEvent, render, screen } from '@testing-library/react';
import {
  createTransformer,
  nextTokenDistribution,
  type TokenizerData,
  type TransformerManifest,
} from '@sim/deep';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

// WebGL is not available in jsdom: the scene is replaced by a stub that exposes what it receives.
vi.mock('./Scene', () => ({
  default: (props: { stage: number | null; selected: number }) => (
    <div data-testid="scene" data-stage={String(props.stage)} data-selected={props.selected} />
  ),
}));

const { ThreeDPanel } = await import('./ThreeDPanel');
const { buildSlabs, buildStages } = await import('./layout');

const dir = resolve(
  dirname(fileURLToPath(import.meta.url)),
  '../../../../../../../packages/deep/models/tiny-gpt',
);
const read = (name: string) => readFileSync(resolve(dir, name));
const json = <T,>(name: string) => JSON.parse(read(name).toString('utf-8')) as T;
const raw = read('weights.bin');
const { model, tokenizer } = createTransformer(
  json<TransformerManifest>('model.json'),
  raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength) as ArrayBuffer,
  json<TokenizerData>('tokenizer.json'),
);
const ids = tokenizer.encode('Anna found a ball. She');
const trace = model.forward(ids);
const distribution = nextTokenDistribution(
  trace.logits.data.slice((ids.length - 1) * trace.logits.cols),
  { temperature: 1, topK: 0, topP: 1 },
);
const stages = buildStages(trace, buildSlabs(trace));

const mount = (onSelect = vi.fn()) =>
  render(
    <ThreeDPanel
      trace={trace}
      tokenizer={tokenizer}
      distribution={distribution}
      position={ids.length - 1}
      onSelect={onSelect}
    />,
  );

describe('<ThreeDPanel />', () => {
  beforeEach(() => {
    vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue({} as never);
  });
  afterEach(() => {
    vi.restoreAllMocks();
    vi.useRealTimers();
  });

  it('explains itself and falls back gracefully without WebGL', () => {
    vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue(null);
    mount();
    expect(screen.getByRole('status')).toHaveTextContent(/WebGL/);
    expect(screen.queryByTestId('scene')).toBeNull();
  });

  it('starts on the overview with one chip per stage', async () => {
    mount();
    expect(await screen.findByTestId('scene')).toHaveAttribute('data-stage', 'null');
    expect(screen.getByTestId('three-host')).toHaveAttribute('data-stage', 'overview');
    expect(screen.getByRole('button', { name: 'Overview' })).toHaveAttribute(
      'aria-pressed',
      'true',
    );
    // Overview + one chip per stage + the Tour toggle (it is a pressed-state button too)
    expect(
      screen
        .getByRole('group', { name: 'Tour of the model' })
        .querySelectorAll('button[aria-pressed]'),
    ).toHaveLength(stages.length + 2);
    expect(screen.getByText(/residual stream/)).toBeInTheDocument();
  });

  it('flies to a stage, narrates it and tells the scene', async () => {
    mount();
    fireEvent.click(await screen.findByRole('button', { name: /Attention · layer 1/ }));
    expect(screen.getByTestId('scene')).toHaveAttribute('data-stage', '1');
    expect(screen.getByTestId('three-host')).toHaveAttribute('data-stage', 'attention-0');
    expect(screen.getByRole('heading', { name: /2\. Attention · layer 1/ })).toBeInTheDocument();
    expect(screen.getByText(stages[1]?.text as string)).toBeInTheDocument();

    fireEvent.click(screen.getByRole('button', { name: 'Overview' }));
    expect(screen.getByTestId('scene')).toHaveAttribute('data-stage', 'null');
  });

  it('steps with previous / next and stops at both ends', async () => {
    mount();
    const prev = await screen.findByRole('button', { name: 'Previous stage' });
    const next = screen.getByRole('button', { name: 'Next stage' });
    expect(prev).toBeDisabled(); // overview: nothing before it

    fireEvent.click(next); // overview -> stage 1
    expect(screen.getByTestId('scene')).toHaveAttribute('data-stage', '0');
    expect(prev).toBeDisabled();
    fireEvent.click(next);
    fireEvent.click(next);
    expect(screen.getByTestId('scene')).toHaveAttribute('data-stage', '2');
    fireEvent.click(prev);
    expect(screen.getByTestId('scene')).toHaveAttribute('data-stage', '1');

    fireEvent.click(screen.getByRole('button', { name: /Next-token probabilities/ }));
    expect(next).toBeDisabled();
    expect(
      screen.getByRole('heading', { name: new RegExp(`${stages.length}\\. `) }),
    ).toBeInTheDocument();
  });

  it('the tour auto-advances, wraps around and can be paused', async () => {
    vi.useFakeTimers();
    mount();
    await act(async () => {
      await vi.dynamicImportSettled();
    });
    fireEvent.click(screen.getByRole('button', { name: 'Play the tour' }));
    expect(screen.getByTestId('scene')).toHaveAttribute('data-stage', '0');
    act(() => {
      vi.advanceTimersByTime(3200);
    });
    expect(screen.getByTestId('scene')).toHaveAttribute('data-stage', '1');
    act(() => {
      vi.advanceTimersByTime(3200 * (stages.length - 1));
    });
    expect(screen.getByTestId('scene')).toHaveAttribute('data-stage', '0'); // wrapped

    fireEvent.click(screen.getByRole('button', { name: 'Pause the tour' }));
    act(() => {
      vi.advanceTimersByTime(10_000);
    });
    expect(screen.getByTestId('scene')).toHaveAttribute('data-stage', '0');
  });

  it('manual navigation interrupts the tour', async () => {
    vi.useFakeTimers();
    mount();
    await act(async () => {
      await vi.dynamicImportSettled();
    });
    fireEvent.click(screen.getByRole('button', { name: 'Play the tour' }));
    fireEvent.click(screen.getByRole('button', { name: /MLP · layer 1/ }));
    expect(screen.getByRole('button', { name: 'Play the tour' })).toBeInTheDocument();
    act(() => {
      vi.advanceTimersByTime(10_000);
    });
    expect(screen.getByTestId('scene')).toHaveAttribute('data-stage', '3'); // stayed where the user went
  });

  it('follows the token in focus', async () => {
    mount();
    expect(await screen.findByTestId('scene')).toHaveAttribute(
      'data-selected',
      String(ids.length - 1),
    );
  });
});
