import { ACTIONS, ACTION_COUNT, GRID_PRESETS, parseGrid } from '@sim/deep';
import { useMemo, useState } from 'react';
import { Segmented, Switch, cn } from '@/design-system';
import { useRlStore } from '../store';
import { cellXY } from '../view';
import { GridView, type PaintTool, type ViewMode } from './GridView';

const ARROW = ['↑', '→', '↓', '←'] as const;

const TOOLS: { value: PaintTool; label: string }[] = [
  { value: 'look', label: 'Inspect' },
  { value: 'wall', label: 'Wall' },
  { value: 'pit', label: 'Pit' },
  { value: 'goal', label: 'Goal' },
  { value: 'start', label: 'Start' },
  { value: 'empty', label: 'Erase' },
];

const MODES: { value: ViewMode; label: string }[] = [
  { value: 'values', label: 'Values' },
  { value: 'policy', label: 'Policy' },
  { value: 'q', label: 'Action values' },
  { value: 'visits', label: 'Visits' },
];

const MODE_NOTE: Record<ViewMode, string> = {
  values:
    'Each cell shows how good the agent thinks it is to stand there: frost = bad, ember = good (the number is the estimate).',
  policy:
    'Arrows point where the agent would go from each cell. No arrow means it has learned nothing about that cell yet.',
  q: 'Each cell is split into four triangles, one per move; the colour is how good the agent thinks that move is.',
  visits: 'How often the agent has stood in each cell: brighter = more visits.',
};

/** The paintable gridworld with the learned values, plus the details of one inspected cell. */
export function WorldPanel() {
  const { config, snapshot, inspected } = useRlStore();
  const paint = useRlStore((s) => s.paint);
  const inspect = useRlStore((s) => s.inspect);
  const loadPreset = useRlStore((s) => s.loadPreset);
  const [tool, setTool] = useState<PaintTool>('look');
  const [mode, setMode] = useState<ViewMode>('values');
  const [path, setPath] = useState(true);

  const layout = useMemo(() => parseGrid(config.grid), [config.grid]);
  if (!snapshot) return null;
  const presetId = Object.entries(GRID_PRESETS).find(
    ([, p]) => p.rows.join() === config.grid.join(),
  )?.[0];
  const cell = inspected;
  const [cx, cy] = cell === null ? [0, 0] : cellXY(cell, layout.width);
  const kind = cell === null ? null : layout.kinds[cell];

  return (
    <div className="space-y-6">
      <p className="max-w-3xl text-[0.9rem] text-muted">
        The agent (the purple dot) is dropped on <b className="font-medium text-fg">S</b>. At every
        step it picks one of four moves and gets a reward: −1 for each step, +10 for the goal ★,
        −100 for falling into a pit ✕. It knows nothing about the map or the rules; it only learns
        from what happens. Paint your own world, then press Train in section 02.
      </p>

      <div className="flex flex-wrap items-center gap-x-6 gap-y-3">
        <Segmented
          label="Layout"
          size="sm"
          fit
          value={presetId ?? 'custom'}
          onValueChange={(id) => id !== 'custom' && loadPreset(id)}
          options={[
            ...Object.entries(GRID_PRESETS).map(([id, p]) => ({ value: id, label: p.label })),
            ...(presetId ? [] : [{ value: 'custom', label: 'Custom' }]),
          ]}
        />
        <Segmented<PaintTool>
          label="Tool"
          size="sm"
          fit
          value={tool}
          onValueChange={setTool}
          options={TOOLS}
        />
      </div>
      {presetId && <p className="-mt-3 text-[0.8rem] text-faint">{GRID_PRESETS[presetId]?.note}</p>}

      <div className="grid gap-8 lg:grid-cols-[minmax(0,40rem)_minmax(0,1fr)]">
        <div className="min-w-0 space-y-3">
          <GridView
            rows={config.grid}
            snapshot={snapshot}
            mode={mode}
            showPath={path}
            tool={tool}
            inspected={inspected}
            onPaint={paint}
            onInspect={inspect}
          />
          <p className="text-[0.74rem] text-faint">
            {tool === 'look'
              ? 'Click a cell to inspect it.'
              : `Click or drag on the grid to paint with the "${TOOLS.find((t) => t.value === tool)?.label}" tool. Changing the world restarts learning.`}
          </p>
        </div>

        <div className="min-w-0 space-y-5">
          <div className="space-y-2">
            <Segmented<ViewMode>
              label="What to show"
              size="sm"
              fit
              value={mode}
              onValueChange={setMode}
              options={MODES}
            />
            <p className="text-[0.76rem] leading-snug text-faint">{MODE_NOTE[mode]}</p>
          </div>
          <Switch
            label="Draw the route the agent would take now"
            checked={path}
            onCheckedChange={setPath}
          />
          <p className="text-[0.76rem] leading-snug text-muted">
            Dotted line: following the best move from every cell, the agent would{' '}
            {snapshot.path.outcome === 'goal'
              ? `reach the goal in ${snapshot.path.cells.length - 1} steps.`
              : snapshot.path.outcome === 'pit'
                ? 'fall into a pit.'
                : 'go round in circles and never arrive.'}
          </p>

          <div className="rounded-xl border border-line bg-raised/30 p-4" aria-label="Inspector">
            <p className="eyebrow mb-2">
              {cell === null ? 'Cell inspector' : `Cell (${cx}, ${cy}) · ${kind}`}
            </p>
            {cell === null || kind === 'wall' || kind === 'goal' || kind === 'pit' ? (
              <p className="text-[0.8rem] text-muted">
                {cell === null
                  ? 'Choose the Inspect tool and click a cell to see what the agent thinks of each move from there.'
                  : 'The episode ends here (or the cell is a wall): there are no moves to rate.'}
              </p>
            ) : (
              <>
                <ul className="space-y-1.5 font-mono text-[0.84rem]">
                  {Array.from({ length: ACTION_COUNT }, (_, a) => {
                    const v = snapshot.q[cell * ACTION_COUNT + a] as number;
                    const best = Math.max(
                      ...Array.from(
                        { length: ACTION_COUNT },
                        (_, b) => snapshot.q[cell * ACTION_COUNT + b] as number,
                      ),
                    );
                    return (
                      <li
                        key={a}
                        className={cn(
                          'flex items-center justify-between',
                          v === best && v !== 0 && 'text-fg',
                        )}
                      >
                        <span>
                          {ARROW[a]} {ACTIONS[a]}
                        </span>
                        <span className="tabular-nums">{v.toFixed(2)}</span>
                      </li>
                    );
                  })}
                </ul>
                <p className="mt-3 text-[0.76rem] text-faint">
                  The best possible value of this cell is{' '}
                  <span className="font-mono text-fg">
                    {(snapshot.optimal[cell] as number).toFixed(2)}
                  </span>{' '}
                  (computed exactly with value iteration, which knows the rules the agent has to
                  discover).
                </p>
              </>
            )}
          </div>

          <ul className="space-y-1.5 text-[0.78rem] text-muted" aria-label="Legend">
            <li className="flex items-center gap-2">
              <span className="size-3 rounded-full bg-brand" /> the agent
            </li>
            <li className="flex items-center gap-2">
              <span className="w-4 text-center text-fg">★</span> goal: +10, episode ends
            </li>
            <li className="flex items-center gap-2">
              <span className="w-4 text-center font-bold text-fg">✕</span> pit: −100, episode ends
            </li>
            <li className="flex items-center gap-2">
              <span className="w-4 text-center text-fg">▨</span> wall: blocks movement
            </li>
          </ul>
        </div>
      </div>
    </div>
  );
}
